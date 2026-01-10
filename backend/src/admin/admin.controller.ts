import { Controller, Get, Query } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';

@Controller('v1/admin')
export class AdminController {
  constructor(private prisma: PrismaService) {}

  @Get('ussd-audit/latest')
  async latest(@Query('limit') limit?: string) {
    const take = Math.min(Math.max(Number(limit || 10), 1), 50);

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
}
