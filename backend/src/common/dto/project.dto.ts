/**
 * DTOs for Project Controller endpoints.
 * Uses class-validator for request validation.
 */
import {
  IsString,
  IsNumber,
  IsOptional,
  IsEthereumAddress,
  Min,
  Max,
} from 'class-validator';
import { Type } from 'class-transformer';

// ============ Query DTOs ============
// Note: ListProjectsQueryDto is defined in admin.dto.ts and re-exported from there

export class ListInvestorsQueryDto {
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

// ============ Investment DTOs ============

export class RecordInvestmentDto {
  @IsString()
  @IsEthereumAddress()
  investorAddress: string;

  @IsNumber()
  @Min(1) // Minimum 1 IDRX
  @Max(1000000000) // Maximum 1 billion IDRX
  @Type(() => Number)
  amount: number;

  @IsOptional()
  @IsString()
  txHash?: string;
}
