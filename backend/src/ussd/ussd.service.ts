import { Injectable } from '@nestjs/common';
import Redis from 'ioredis';
import * as crypto from 'crypto';
import { PrismaService } from '../prisma/prisma.service';

type UssdInput = { sessionId: string; phoneNumber: string; text?: string };
type SessionData = { state: 'START' | 'AWAIT_AMOUNT' };

@Injectable()
export class UssdService {
  private redis: Redis;

  constructor(private prisma: PrismaService) {
    this.redis = new Redis({
      host: process.env.REDIS_HOST || '127.0.0.1',
      port: Number(process.env.REDIS_PORT || 6379),
      maxRetriesPerRequest: 2,
    });
  }

  private ttl(): number {
    return Number(process.env.SESSION_TTL_SECONDS || 120);
  }

  private phoneHash(phone: string): string {
    const salt = process.env.PHONE_HASH_SALT || 'dev';
    return crypto.createHash('sha256').update(`${salt}:${phone}`).digest('hex');
  }

  private menu(): string {
    return ['CON TaniFi', '1. Register', '2. Create Project', '00. Exit'].join('\n');
  }

  private exit(): string {
    return 'END Thank you for using TaniFi 🌱';
  }

  private amountPrompt(): string {
    return ['CON Input project amount (Rp)', 'Example: 5000000', '0. Back', '00. Exit'].join('\n');
  }

  private invalidAmount(): string {
    return ['CON Invalid amount.', 'Input number only (e.g., 5000000)', '0. Back', '00. Exit'].join('\n');
  }

  async handle(input: UssdInput): Promise<{ message: string; stateBefore: string; stateAfter: string }> {
    const key = `ussd:${input.sessionId}`;
    const text = (input.text ?? '').trim();
    const phoneHash = this.phoneHash(input.phoneNumber);

    const raw = await this.redis.get(key);
    const sess: SessionData = raw ? JSON.parse(raw) : { state: 'START' };
    const stateBefore = sess.state;

    // New session
    if (text === '') {
      await this.redis.set(key, JSON.stringify(sess), 'EX', this.ttl());
      return { message: this.menu(), stateBefore, stateAfter: sess.state };
    }

    const parts = text.split('*');
    const last = parts[parts.length - 1] ?? '';

    // START state
    if (sess.state === 'START') {
      if (last === '00') {
        await this.redis.del(key);
        return { message: this.exit(), stateBefore, stateAfter: 'END' };
      }

      if (last === '1') {
        await this.prisma.user.upsert({
          where: { phoneHash },
          update: {},
          create: { phoneHash },
        });
        await this.redis.del(key);
        return { message: 'END Registration successful ✅', stateBefore, stateAfter: 'END' };
      }

      if (last === '2') {
        sess.state = 'AWAIT_AMOUNT';
        await this.redis.set(key, JSON.stringify(sess), 'EX', this.ttl());
        return { message: this.amountPrompt(), stateBefore, stateAfter: sess.state };
      }

      await this.redis.del(key);
      return { message: 'END Invalid choice.', stateBefore, stateAfter: 'END' };
    }

    // AWAIT_AMOUNT state
    if (sess.state === 'AWAIT_AMOUNT') {
      if (last === '00') {
        await this.redis.del(key);
        return { message: this.exit(), stateBefore, stateAfter: 'END' };
      }

      if (last === '0') {
        sess.state = 'START';
        await this.redis.set(key, JSON.stringify(sess), 'EX', this.ttl());
        return { message: this.menu(), stateBefore, stateAfter: sess.state };
      }

      const amount = Number(last);
      if (!Number.isFinite(amount) || amount <= 0) {
        await this.redis.set(key, JSON.stringify(sess), 'EX', this.ttl());
        return { message: this.invalidAmount(), stateBefore, stateAfter: sess.state };
      }

      const user = await this.prisma.user.upsert({
        where: { phoneHash },
        update: {},
        create: { phoneHash },
      });

      const txHash = `0x${crypto.randomBytes(32).toString('hex')}`;

const project = await this.prisma.project.create({
  data: {
    userId: user.id,
    amount: Math.floor(amount),
    status: 'created',
    txs: {
      create: {
        kind: 'create_project',
        status: 'pending',
        txHash,
      },
    },
  },
  include: {
    txs: true,
  },
});

// Simulate chain confirmation: after 10 seconds mark success
setTimeout(async () => {
  try {
    await this.prisma.transaction.update({
      where: { id: project.txs[0].id },
      data: { status: 'success' },
    });

    await this.prisma.project.update({
      where: { id: project.id },
      data: { status: 'confirmed' },
    });
  } catch {
    // ignore for demo
  }
}, 10000);


      await this.redis.del(key);
      const success = [
          'END Project created 🎉',
          `ID: ${project.id}`,
          `Amount: Rp ${project.amount.toLocaleString('id-ID')}`,
          `Tx: ${txHash}`,
          'Status: pending (auto-confirm in ~10s)',
      ].join('\n');
      return { message: success, stateBefore, stateAfter: 'END' };
    }

    await this.redis.del(key);
    return { message: 'END Session reset. Please try again.', stateBefore, stateAfter: 'END' };
  }
}
