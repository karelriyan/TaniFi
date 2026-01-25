/**
 * DTOs for Admin Controller endpoints.
 * Uses class-validator for request validation.
 */
import {
  IsString,
  IsNumber,
  IsOptional,
  IsBoolean,
  IsDateString,
  IsEthereumAddress,
  Min,
  Max,
  MinLength,
  MaxLength,
} from 'class-validator';
import { Type } from 'class-transformer';

// ============ Project DTOs ============

export class CreateProjectDto {
  @IsString()
  @MinLength(1)
  farmerId: string;

  @IsString()
  @IsEthereumAddress()
  vendorAddress: string;

  @IsNumber()
  @Min(100) // Minimum 100 IDRX
  @Max(1000000000) // Maximum 1 billion IDRX
  @Type(() => Number)
  targetAmount: number;

  @IsOptional()
  @IsNumber()
  @Min(1000) // Minimum 10% (1000 bps)
  @Max(5000) // Maximum 50% (5000 bps)
  @Type(() => Number)
  farmerShareBps?: number;

  @IsDateString()
  harvestTime: string;

  @IsOptional()
  @IsString()
  @MaxLength(50)
  commodity?: string;

  @IsOptional()
  @IsString()
  @MaxLength(100)
  ipfsMetadata?: string;
}

export class ReportHarvestDto {
  @IsNumber()
  @Min(0)
  @Type(() => Number)
  revenue: number;
}

export class ListProjectsQueryDto {
  @IsOptional()
  @IsString()
  status?: string;

  @IsOptional()
  @IsNumber()
  @Min(1)
  @Max(100)
  @Type(() => Number)
  limit?: number;

  @IsOptional()
  @IsNumber()
  @Min(0)
  @Type(() => Number)
  offset?: number;
}

// ============ Farmer DTOs ============

export class ListFarmersQueryDto {
  @IsOptional()
  @IsString()
  kycStatus?: string;

  @IsOptional()
  @IsString()
  status?: string; // Alias for kycStatus (for frontend compatibility)

  @IsOptional()
  @IsNumber()
  @Min(1)
  @Max(100)
  @Type(() => Number)
  limit?: number;

  @IsOptional()
  @IsNumber()
  @Min(0)
  @Type(() => Number)
  offset?: number;
}

// ============ Vendor DTOs ============

export class CreateVendorDto {
  @IsString()
  @MinLength(1)
  @MaxLength(100)
  name: string;

  @IsString()
  @IsEthereumAddress()
  walletAddress: string;

  @IsOptional()
  @IsBoolean()
  isWhitelisted?: boolean;
}

export class WhitelistVendorDto {
  @IsBoolean()
  isWhitelisted: boolean;
}

export class ListVendorsQueryDto {
  @IsOptional()
  @IsString()
  whitelisted?: string;
}

// ============ USSD Audit DTOs ============

export class UssdAuditQueryDto {
  @IsOptional()
  @IsNumber()
  @Min(1)
  @Max(50)
  @Type(() => Number)
  limit?: number;
}
