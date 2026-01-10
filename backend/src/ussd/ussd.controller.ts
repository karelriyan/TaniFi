/**
 * USSD webhook entrypoint.
 * - Delegates menu/state handling to UssdService.
 * - Records audit logs with a hashed phone number.
 * - Returns plain-text USSD responses (CON/END).
 */
import { Body, Controller, Post, Res } from '@nestjs/common';
import type { Response } from 'express';
import { PrismaService } from '../prisma/prisma.service';
import { UssdService } from './ussd.service';
import * as crypto from 'crypto';

type UssdWebhookDto = {
  sessionId: string;
  phoneNumber: string;
  text?: string;
};

@Controller('v1/hooks')
export class UssdController {
  constructor(
    private readonly ussd: UssdService,
    private readonly prisma: PrismaService,
  ) {}

  @Post('ussd')
  async ussdWebhook(@Body() dto: UssdWebhookDto, @Res() res: Response) {
    const requestId = crypto.randomUUID();

    const result = await this.ussd.handle(dto);
    const responseType = result.message.startsWith('CON') ? 'CON' : 'END';

    const salt = process.env.PHONE_HASH_SALT || 'dev';
    const phoneHash = crypto.createHash('sha256').update(`${salt}:${dto.phoneNumber}`).digest('hex');

    await this.prisma.ussdAudit.create({
      data: {
        requestId,
        sessionId: dto.sessionId,
        phoneHash,
        text: dto.text ?? '',
        stateBefore: result.stateBefore,
        stateAfter: result.stateAfter,
        responseType,
        responseText: result.message,
      },
    });

    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.setHeader('X-Request-Id', requestId);
    return res.send(result.message);
  }
}
