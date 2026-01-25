import { Controller, Post, Body, HttpException, HttpStatus } from '@nestjs/common';
import { FarmerService } from './farmer.service';

export class RegisterFarmerDto {
  name: string;
  nik: string;
  phoneNumber: string;
  landSize: number;
  location: string;
  walletAddress?: string;
  registrationMethod: 'WEB' | 'USSD';
}

@Controller('v1/farmers')
export class FarmerController {
  constructor(private readonly farmerService: FarmerService) {}

  @Post('register')
  async registerFarmer(@Body() dto: RegisterFarmerDto) {
    try {
      // Validate required fields
      if (!dto.name || !dto.nik || !dto.phoneNumber || !dto.landSize || !dto.location) {
        throw new HttpException(
          'Missing required fields',
          HttpStatus.BAD_REQUEST,
        );
      }

      // NIK validation (16 digits)
      if (!/^\d{16}$/.test(dto.nik)) {
        throw new HttpException(
          'NIK must be 16 digits',
          HttpStatus.BAD_REQUEST,
        );
      }

      // Phone number validation
      if (!/^(\+62|62|0)[0-9]{9,12}$/.test(dto.phoneNumber)) {
        throw new HttpException(
          'Invalid phone number format',
          HttpStatus.BAD_REQUEST,
        );
      }

      const farmer = await this.farmerService.registerFarmer(dto);

      return {
        message: 'Farmer registered successfully',
        farmer,
      };
    } catch (error) {
      if (error instanceof HttpException) {
        throw error;
      }

      console.error('Register farmer error:', error);
      throw new HttpException(
        error.message || 'Failed to register farmer',
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }
}
