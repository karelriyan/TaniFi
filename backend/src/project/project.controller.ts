/**
 * Public project endpoints.
 * Provides project listing, details, and investment functionality.
 */
import {
  Controller,
  Get,
  Post,
  Param,
  Query,
  Body,
  HttpException,
  HttpStatus,
} from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { BlockchainService } from '../blockchain/blockchain.service';
import {
  ListProjectsQueryDto,
  ListInvestorsQueryDto,
  RecordInvestmentDto,
} from '../common/dto';

@Controller('v1/projects')
export class ProjectController {
  constructor(
    private prisma: PrismaService,
    private blockchain: BlockchainService,
  ) {}

  /**
   * List all projects with optional filtering
   */
  @Get()
  async listProjects(@Query() query: ListProjectsQueryDto) {
    const take = query.limit || 20;
    const skip = query.offset || 0;

    // Build where clause
    const where: Record<string, unknown> = {};
    if (query.status) {
      where.status = query.status;
    }

    const [projects, total] = await Promise.all([
      this.prisma.project.findMany({
        where,
        take,
        skip,
        orderBy: { createdAt: 'desc' },
        select: {
          id: true,
          commodity: true,
          targetAmount: true,
          fundedAmount: true,
          farmerShareBps: true,
          status: true,
          harvestTime: true,
          chainProjectId: true,
          createdAt: true,
          _count: {
            select: { investments: true },
          },
        },
      }),
      this.prisma.project.count({ where }),
    ]);

    return {
      ok: true,
      data: projects.map((p) => ({
        ...p,
        investorShareBps: 10000 - p.farmerShareBps - 100, // 100 = platform fee
        fundingProgress:
          (p.targetAmount ?? 0) > 0
            ? (p.fundedAmount / (p.targetAmount ?? 1)) * 100
            : 0,
      })),
      pagination: { total, limit: take, offset: skip },
    };
  }

  /**
   * Get project details with on-chain data
   */
  @Get(':id')
  async getProject(@Param('id') id: string) {
    const project = await this.prisma.project.findUnique({
      where: { id },
      include: {
        user: {
          select: {
            id: true,
            walletAddress: true,
            reputationScore: true,
          },
        },
        investments: {
          select: {
            investorAddress: true,
            amount: true,
            createdAt: true,
          },
          orderBy: { createdAt: 'desc' },
        },
        txs: {
          orderBy: { createdAt: 'desc' },
          take: 5,
        },
      },
    });

    if (!project) {
      throw new HttpException('Project not found', HttpStatus.NOT_FOUND);
    }

    // Try to fetch on-chain data if available
    let onChainData: Awaited<ReturnType<typeof this.blockchain.getProject>> | null = null;
    if (project.chainProjectId !== null) {
      try {
        onChainData = await this.blockchain.getProject(project.chainProjectId);
      } catch {
        // Ignore on-chain errors, return DB data
      }
    }

    return {
      ok: true,
      data: {
        ...project,
        investorShareBps: 10000 - project.farmerShareBps - 100,
        fundingProgress:
          (project.targetAmount ?? 0) > 0
            ? (project.fundedAmount / (project.targetAmount ?? 1)) * 100
            : 0,
        onChain: onChainData,
      },
    };
  }

  /**
   * Get project status (legacy endpoint)
   */
  @Get(':id/status')
  async getStatus(@Param('id') id: string) {
    const project = await this.prisma.project.findUnique({
      where: { id },
      include: { txs: { orderBy: { createdAt: 'asc' } } },
    });

    if (!project) return { ok: false, message: 'Not found' };

    const tx = project.txs[0] ?? null;

    return {
      ok: true,
      projectId: project.id,
      projectStatus: project.status,
      tx: tx
        ? { id: tx.id, kind: tx.kind, status: tx.status, txHash: tx.txHash }
        : null,
    };
  }

  /**
   * Get project investors
   */
  @Get(':id/investors')
  async getInvestors(
    @Param('id') id: string,
    @Query() query: ListInvestorsQueryDto,
  ) {
    const take = query.limit || 20;
    const skip = query.offset || 0;

    const project = await this.prisma.project.findUnique({
      where: { id },
    });

    if (!project) {
      throw new HttpException('Project not found', HttpStatus.NOT_FOUND);
    }

    const [investments, total] = await Promise.all([
      this.prisma.investment.findMany({
        where: { projectId: id },
        take,
        skip,
        orderBy: { createdAt: 'desc' },
      }),
      this.prisma.investment.count({ where: { projectId: id } }),
    ]);

    return {
      ok: true,
      data: investments,
      pagination: { total, limit: take, offset: skip },
    };
  }

  /**
   * Record investment (for USSD/backend flow, not direct blockchain)
   * Note: Direct blockchain investments should use the frontend
   */
  @Post(':id/invest')
  async recordInvestment(
    @Param('id') id: string,
    @Body() dto: RecordInvestmentDto,
  ) {
    const project = await this.prisma.project.findUnique({
      where: { id },
    });

    if (!project) {
      throw new HttpException('Project not found', HttpStatus.NOT_FOUND);
    }

    if (project.status !== 'confirmed' && project.status !== 'created') {
      throw new HttpException(
        'Project not accepting investments',
        HttpStatus.BAD_REQUEST,
      );
    }

    // Check if would exceed target
    const newFundedAmount = project.fundedAmount + dto.amount;
    if (project.targetAmount && newFundedAmount > project.targetAmount) {
      throw new HttpException(
        'Investment would exceed target amount',
        HttpStatus.BAD_REQUEST,
      );
    }

    // Create or update investment record
    const investment = await this.prisma.investment.upsert({
      where: {
        projectId_investorAddress: {
          projectId: id,
          investorAddress: dto.investorAddress,
        },
      },
      update: {
        amount: { increment: dto.amount },
        txHash: dto.txHash,
      },
      create: {
        projectId: id,
        investorAddress: dto.investorAddress,
        amount: dto.amount,
        txHash: dto.txHash,
      },
    });

    // Update project funded amount
    await this.prisma.project.update({
      where: { id },
      data: {
        fundedAmount: newFundedAmount,
        status:
          project.targetAmount && newFundedAmount >= project.targetAmount
            ? 'funded'
            : project.status,
      },
    });

    // Record transaction if txHash provided
    if (dto.txHash) {
      await this.prisma.transaction.create({
        data: {
          projectId: id,
          kind: 'invest',
          status: 'success',
          txHash: dto.txHash,
          amount: dto.amount,
          fromAddr: dto.investorAddress,
        },
      });
    }

    return {
      ok: true,
      investment,
      newFundedAmount,
      fullyFunded:
        project.targetAmount && newFundedAmount >= project.targetAmount,
    };
  }

  /**
   * Get projects by on-chain status (fetches from blockchain)
   */
  @Get('chain/:chainProjectId')
  async getChainProject(@Param('chainProjectId') chainProjectId: string) {
    const id = Number(chainProjectId);
    if (isNaN(id)) {
      throw new HttpException('Invalid project ID', HttpStatus.BAD_REQUEST);
    }

    try {
      const project = await this.blockchain.getProject(id);
      if (!project) {
        throw new HttpException('Project not found on chain', HttpStatus.NOT_FOUND);
      }

      return {
        ok: true,
        data: {
          ...project,
          targetAmountFormatted: this.blockchain.formatIDRX(project.targetAmount),
          fundedAmountFormatted: this.blockchain.formatIDRX(project.fundedAmount),
        },
      };
    } catch (error) {
      throw new HttpException(
        `Failed to fetch project: ${error.message}`,
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }
}
