import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { PrismaModule } from './prisma/prisma.module';
import { UssdModule } from './ussd/ussd.module';
import { AdminController } from './admin/admin.controller';
import { ProjectController } from './project/project.controller';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),
    PrismaModule,
    UssdModule,
  ],
  controllers: [AdminController, ProjectController],
})
export class AppModule {}
