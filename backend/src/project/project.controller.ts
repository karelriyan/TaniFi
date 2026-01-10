import { Controller, Get, Param } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';

@Controller('v1/projects')
export class ProjectController {
  constructor(private prisma: PrismaService) {}

  @Get(':id/status')
  async status(@Param('id') id: string) {
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
}
