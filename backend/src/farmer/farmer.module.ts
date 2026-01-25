import { Module } from '@nestjs/common';
import { FarmerController } from './farmer.controller';
import { FarmerService } from './farmer.service';
import { PrismaModule } from '../prisma/prisma.module';

@Module({
  imports: [PrismaModule],
  controllers: [FarmerController],
  providers: [FarmerService],
  exports: [FarmerService],
})
export class FarmerModule {}
