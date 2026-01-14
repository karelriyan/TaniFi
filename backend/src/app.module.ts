/**
 * Root module for the backend API.
 * - Loads environment config globally.
 * - Configures rate limiting with ThrottlerModule.
 * - Wires Prisma, USSD, and Blockchain modules.
 * - Exposes admin and project HTTP controllers.
 */
import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { ThrottlerModule, ThrottlerGuard } from '@nestjs/throttler';
import { APP_GUARD } from '@nestjs/core';
import { PrismaModule } from './prisma/prisma.module';
import { UssdModule } from './ussd/ussd.module';
import { BlockchainModule } from './blockchain/blockchain.module';
import { AdminController } from './admin/admin.controller';
import { ProjectController } from './project/project.controller';
import { HealthController } from './health/health.controller';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),

    // Rate limiting configuration
    ThrottlerModule.forRoot([
      {
        name: 'short',
        ttl: 1000, // 1 second
        limit: 10, // 10 requests per second (for USSD)
      },
      {
        name: 'medium',
        ttl: 10000, // 10 seconds
        limit: 50, // 50 requests per 10 seconds
      },
      {
        name: 'long',
        ttl: 60000, // 1 minute
        limit: 200, // 200 requests per minute
      },
    ]),

    PrismaModule,
    UssdModule,
    BlockchainModule,
  ],
  controllers: [AdminController, ProjectController, HealthController],
  providers: [
    // Apply throttler guard globally
    {
      provide: APP_GUARD,
      useClass: ThrottlerGuard,
    },
  ],
})
export class AppModule {}
