/**
 * Admin-only endpoints for operational management.
 * Provides USSD audit, project management, and farmer management.
 */
import {
  Controller,
  Get,
  Post,
  Put,
  Body,
  Param,
  Query,
  HttpException,
  HttpStatus,
} from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { BlockchainService } from '../blockchain/blockchain.service';
import {
  CreateProjectDto,
  ReportHarvestDto,
  ListProjectsQueryDto,
  ListFarmersQueryDto,
  CreateVendorDto,
  WhitelistVendorDto,
  ListVendorsQueryDto,
  UssdAuditQueryDto,
} from '../common/dto';

@Controller('v1/admin')
export class AdminController {
  constructor(
    private prisma: PrismaService,
    private blockchain: BlockchainService,
  ) {}

  // ============ USSD Audit Endpoints ============

  @Get('ussd-audit/latest')
  async getLatestAudit(@Query() query: UssdAuditQueryDto) {
    const take = query.limit || 10;

    return this.prisma.ussdAudit.findMany({
      take,
      orderBy: { createdAt: 'desc' },
      select: {
        createdAt: true,
        requestId: true,
        sessionId: true,
        stateBefore: true,
        stateAfter: true,
        responseType: true,
      },
    });
  }

  // ============ Project Management Endpoints ============

  @Get('projects')
  async listProjects(@Query() query: ListProjectsQueryDto) {
    const take = query.limit || 20;
    const skip = query.offset || 0;

    const where = query.status ? { status: query.status } : {};

    const [projects, total] = await Promise.all([
      this.prisma.project.findMany({
        where,
        take,
        skip,
        orderBy: { createdAt: 'desc' },
        include: {
          user: {
            select: {
              id: true,
              phoneHash: true,
              walletAddress: true,
              kycStatus: true,
            },
          },
          _count: {
            select: { investments: true },
          },
        },
      }),
      this.prisma.project.count({ where }),
    ]);

    return {
      ok: true,
      data: projects,
      pagination: { total, limit: take, offset: skip },
    };
  }

  @Post('projects')
  async createProject(@Body() dto: CreateProjectDto) {
    // Validate farmer exists
    const farmer = await this.prisma.user.findUnique({
      where: { id: dto.farmerId },
    });

    if (!farmer) {
      throw new HttpException('Farmer not found', HttpStatus.NOT_FOUND);
    }

    if (!farmer.walletAddress) {
      throw new HttpException(
        'Farmer has no wallet address',
        HttpStatus.BAD_REQUEST,
      );
    }

    // Calculate harvest time
    const harvestTime = new Date(dto.harvestTime);
    if (harvestTime <= new Date()) {
      throw new HttpException(
        'Harvest time must be in the future',
        HttpStatus.BAD_REQUEST,
      );
    }

    const harvestTimestamp = Math.floor(harvestTime.getTime() / 1000);
    const farmerShareBps = dto.farmerShareBps || 3000; // Default 30%

    // Create project in database first
    const project = await this.prisma.project.create({
      data: {
        userId: dto.farmerId,
        commodity: dto.commodity,
        targetAmount: dto.targetAmount,
        amount: dto.targetAmount,
        farmerShareBps,
        harvestTime,
        ipfsMetadata: dto.ipfsMetadata,
        status: 'created',
      },
    });

    // Try to create on-chain (if blockchain service is configured)
    try {
      const result = await this.blockchain.createProject(
        farmer.walletAddress,
        dto.vendorAddress,
        this.blockchain.parseIDRX(dto.targetAmount.toString()),
        farmerShareBps,
        harvestTimestamp,
        dto.ipfsMetadata || '',
      );

      // Update project with chain ID
      await this.prisma.project.update({
        where: { id: project.id },
        data: {
          chainProjectId: result.projectId,
          status: 'confirmed',
        },
      });

      // Record transaction
      await this.prisma.transaction.create({
        data: {
          projectId: project.id,
          kind: 'create_project',
          status: 'success',
          txHash: result.txHash,
        },
      });

      return {
        ok: true,
        projectId: project.id,
        chainProjectId: result.projectId,
        txHash: result.txHash,
      };
    } catch (error) {
      // If blockchain fails, still return DB project
      return {
        ok: true,
        projectId: project.id,
        chainProjectId: null,
        warning: 'Blockchain transaction failed, project saved to database only',
        error: error.message,
      };
    }
  }

  @Put('projects/:id/disburse')
  async disburseToVendor(@Param('id') id: string) {
    const project = await this.prisma.project.findUnique({
      where: { id },
    });

    if (!project) {
      throw new HttpException('Project not found', HttpStatus.NOT_FOUND);
    }

    if (!project.chainProjectId) {
      throw new HttpException(
        'Project not deployed to blockchain',
        HttpStatus.BAD_REQUEST,
      );
    }

    try {
      const txHash = await this.blockchain.disburseToVendor(
        project.chainProjectId,
      );

      await this.prisma.project.update({
        where: { id },
        data: { status: 'active' },
      });

      await this.prisma.transaction.create({
        data: {
          projectId: id,
          kind: 'disburse',
          status: 'success',
          txHash,
        },
      });

      return { ok: true, txHash };
    } catch (error) {
      throw new HttpException(
        `Disbursement failed: ${error.message}`,
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  @Put('projects/:id/harvest')
  async reportHarvest(@Param('id') id: string, @Body() dto: ReportHarvestDto) {
    const project = await this.prisma.project.findUnique({
      where: { id },
    });

    if (!project) {
      throw new HttpException('Project not found', HttpStatus.NOT_FOUND);
    }

    if (!project.chainProjectId) {
      throw new HttpException(
        'Project not deployed to blockchain',
        HttpStatus.BAD_REQUEST,
      );
    }

    try {
      const txHash = await this.blockchain.reportHarvest(
        project.chainProjectId,
        this.blockchain.parseIDRX(dto.revenue.toString()),
      );

      await this.prisma.project.update({
        where: { id },
        data: {
          status: 'harvested',
          harvestRevenue: dto.revenue,
        },
      });

      await this.prisma.transaction.create({
        data: {
          projectId: id,
          kind: 'harvest',
          status: 'success',
          txHash,
          amount: dto.revenue,
        },
      });

      return { ok: true, txHash };
    } catch (error) {
      throw new HttpException(
        `Harvest report failed: ${error.message}`,
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  @Put('projects/:id/finalize')
  async finalizeHarvest(@Param('id') id: string) {
    const project = await this.prisma.project.findUnique({
      where: { id },
    });

    if (!project) {
      throw new HttpException('Project not found', HttpStatus.NOT_FOUND);
    }

    if (!project.chainProjectId) {
      throw new HttpException(
        'Project not deployed to blockchain',
        HttpStatus.BAD_REQUEST,
      );
    }

    try {
      const txHash = await this.blockchain.finalizeHarvest(
        project.chainProjectId,
      );

      await this.prisma.project.update({
        where: { id },
        data: { status: 'completed' },
      });

      await this.prisma.transaction.create({
        data: {
          projectId: id,
          kind: 'finalize',
          status: 'success',
          txHash,
        },
      });

      return { ok: true, txHash };
    } catch (error) {
      throw new HttpException(
        `Finalization failed: ${error.message}`,
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  // ============ Farmer Management Endpoints ============

  @Get('farmers')
  async listFarmers(@Query() query: ListFarmersQueryDto) {
    const take = query.limit || 20;
    const skip = query.offset || 0;

    const where = query.kycStatus ? { kycStatus: query.kycStatus } : {};

    const [farmers, total] = await Promise.all([
      this.prisma.user.findMany({
        where,
        take,
        skip,
        orderBy: { createdAt: 'desc' },
        select: {
          id: true,
          phoneHash: true,
          walletAddress: true,
          kycStatus: true,
          reputationScore: true,
          createdAt: true,
          _count: {
            select: { projects: true },
          },
        },
      }),
      this.prisma.user.count({ where }),
    ]);

    return {
      ok: true,
      data: farmers,
      pagination: { total, limit: take, offset: skip },
    };
  }

  @Put('farmers/:id/verify')
  async verifyFarmer(@Param('id') id: string) {
    const farmer = await this.prisma.user.findUnique({
      where: { id },
    });

    if (!farmer) {
      throw new HttpException('Farmer not found', HttpStatus.NOT_FOUND);
    }

    if (!farmer.walletAddress) {
      throw new HttpException(
        'Farmer has no wallet address',
        HttpStatus.BAD_REQUEST,
      );
    }

    try {
      // Verify on blockchain
      const txHash = await this.blockchain.verifyFarmer(farmer.walletAddress);

      // Update database
      await this.prisma.user.update({
        where: { id },
        data: { kycStatus: 'VERIFIED' },
      });

      return { ok: true, txHash };
    } catch (error) {
      // If blockchain fails, still update DB
      await this.prisma.user.update({
        where: { id },
        data: { kycStatus: 'VERIFIED' },
      });

      return {
        ok: true,
        warning: 'Blockchain verification failed, database updated only',
        error: error.message,
      };
    }
  }

  @Put('farmers/:id/reject')
  async rejectFarmer(@Param('id') id: string) {
    const farmer = await this.prisma.user.findUnique({
      where: { id },
    });

    if (!farmer) {
      throw new HttpException('Farmer not found', HttpStatus.NOT_FOUND);
    }

    await this.prisma.user.update({
      where: { id },
      data: { kycStatus: 'REJECTED' },
    });

    return { ok: true };
  }

  // ============ Vendor Management Endpoints ============

  @Get('vendors')
  async listVendors(@Query() query: ListVendorsQueryDto) {
    const where =
      query.whitelisted !== undefined
        ? { isWhitelisted: query.whitelisted === 'true' }
        : {};

    const vendors = await this.prisma.vendor.findMany({
      where,
      orderBy: { createdAt: 'desc' },
    });

    return { ok: true, data: vendors };
  }

  @Post('vendors')
  async createVendor(@Body() dto: CreateVendorDto) {
    const vendor = await this.prisma.vendor.create({
      data: {
        name: dto.name,
        walletAddress: dto.walletAddress,
        isWhitelisted: dto.isWhitelisted ?? false,
      },
    });

    return { ok: true, data: vendor };
  }

  @Put('vendors/:id/whitelist')
  async whitelistVendor(
    @Param('id') id: string,
    @Body() dto: WhitelistVendorDto,
  ) {
    const vendor = await this.prisma.vendor.update({
      where: { id },
      data: { isWhitelisted: dto.isWhitelisted },
    });

    return { ok: true, data: vendor };
  }
}
