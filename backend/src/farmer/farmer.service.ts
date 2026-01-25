import { Injectable, HttpException, HttpStatus } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { createHash } from 'crypto';

interface RegisterFarmerDto {
  name: string;
  nik: string;
  phoneNumber: string;
  landSize: number;
  location: string;
  walletAddress?: string;
  registrationMethod: 'WEB' | 'USSD';
}

@Injectable()
export class FarmerService {
  constructor(private prisma: PrismaService) {}

  private hashPhoneNumber(phoneNumber: string): string {
    // Normalize phone number (remove +, 0, spaces)
    const normalized = phoneNumber.replace(/^\+?62|^0/, '').replace(/\s/g, '');
    return createHash('sha256').update(normalized).digest('hex');
  }

  async registerFarmer(dto: RegisterFarmerDto) {
    const phoneHash = this.hashPhoneNumber(dto.phoneNumber);

    try {
      // Check for duplicate NIK
      const existingByNIK = await this.prisma.user.findUnique({
        where: { farmerNIK: dto.nik },
      });

      if (existingByNIK) {
        throw new HttpException(
          'NIK already registered',
          HttpStatus.CONFLICT,
        );
      }

      // Check for duplicate phone number
      const existingByPhone = await this.prisma.user.findUnique({
        where: { phoneHash },
      });

      if (existingByPhone) {
        // Update existing record instead of creating duplicate
        const updated = await this.prisma.user.update({
          where: { id: existingByPhone.id },
          data: {
            farmerName: dto.name,
            farmerNIK: dto.nik,
            landSize: dto.landSize,
            location: dto.location,
            walletAddress: dto.walletAddress || existingByPhone.walletAddress,
            registrationMethod: dto.registrationMethod,
            role: 'FARMER',
            kycStatus: 'PENDING',
            updatedAt: new Date(),
          },
        });

        return {
          id: updated.id,
          name: updated.farmerName,
          nik: updated.farmerNIK,
          phoneHash: updated.phoneHash,
          landSize: updated.landSize,
          location: updated.location,
          kycStatus: updated.kycStatus,
          registrationMethod: updated.registrationMethod,
          createdAt: updated.createdAt,
        };
      }

      // Create new farmer record
      const farmer = await this.prisma.user.create({
        data: {
          phoneHash,
          farmerName: dto.name,
          farmerNIK: dto.nik,
          landSize: dto.landSize,
          location: dto.location,
          walletAddress: dto.walletAddress,
          registrationMethod: dto.registrationMethod,
          role: 'FARMER',
          kycStatus: 'PENDING',
        },
      });

      return {
        id: farmer.id,
        name: farmer.farmerName,
        nik: farmer.farmerNIK,
        phoneHash: farmer.phoneHash,
        landSize: farmer.landSize,
        location: farmer.location,
        kycStatus: farmer.kycStatus,
        registrationMethod: farmer.registrationMethod,
        createdAt: farmer.createdAt,
      };

    } catch (error) {
      if (error instanceof HttpException) {
        throw error;
      }

      console.error('Register farmer DB error:', error);
      throw new HttpException(
        'Database error during farmer registration',
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  async getFarmersByStatus(status: string) {
    try {
      const farmers = await this.prisma.user.findMany({
        where: {
          role: 'FARMER',
          kycStatus: status,
        },
        orderBy: {
          createdAt: 'desc',
        },
      });

      return farmers.map(farmer => ({
        id: farmer.id,
        farmerName: farmer.farmerName,
        farmerNIK: farmer.farmerNIK,
        phoneHash: farmer.phoneHash,
        landSize: farmer.landSize,
        location: farmer.location,
        kycStatus: farmer.kycStatus,
        registrationMethod: farmer.registrationMethod,
        createdAt: farmer.createdAt,
        walletAddress: farmer.walletAddress,
      }));
    } catch (error) {
      console.error('Get farmers error:', error);
      throw new HttpException(
        'Failed to fetch farmers',
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  async verifyFarmer(farmerId: string, cooperativeAddress: string) {
    try {
      const farmer = await this.prisma.user.update({
        where: { id: farmerId },
        data: {
          kycStatus: 'VERIFIED',
          verifiedAt: new Date(),
          verifiedBy: cooperativeAddress,
        },
      });

      return {
        id: farmer.id,
        name: farmer.farmerName,
        kycStatus: farmer.kycStatus,
        verifiedAt: farmer.verifiedAt,
      };
    } catch (error) {
      console.error('Verify farmer error:', error);
      throw new HttpException(
        'Failed to verify farmer',
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  async rejectFarmer(farmerId: string, cooperativeAddress: string) {
    try {
      const farmer = await this.prisma.user.update({
        where: { id: farmerId },
        data: {
          kycStatus: 'REJECTED',
          verifiedAt: new Date(),
          verifiedBy: cooperativeAddress,
        },
      });

      return {
        id: farmer.id,
        name: farmer.farmerName,
        kycStatus: farmer.kycStatus,
      };
    } catch (error) {
      console.error('Reject farmer error:', error);
      throw new HttpException(
        'Failed to reject farmer',
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }
}
