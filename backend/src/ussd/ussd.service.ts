/**
 * USSD state machine and session persistence.
 * - Implements complete TaniFi menu flow as per LLD specification.
 * - Stores per-session state in Redis with a short TTL.
 * - Supports registration, project creation, status check, and account info.
 * - Persists user/project/tx records via Prisma.
 * - Hashes phone numbers before storage for privacy.
 */
import { Injectable } from '@nestjs/common';
import Redis from 'ioredis';
import * as crypto from 'crypto';
import { PrismaService } from '../prisma/prisma.service';

type UssdInput = { sessionId: string; phoneNumber: string; text?: string };

// Extended state machine for complete flow
type UssdState =
  | 'START'
  | 'AWAIT_PIN_SETUP'
  | 'AWAIT_PIN_CONFIRM'
  | 'AWAIT_PIN_VERIFY'
  | 'SELECT_COMMODITY'
  | 'AWAIT_AMOUNT'
  | 'CONFIRM_PROJECT'
  | 'VIEW_STATUS'
  | 'VIEW_ACCOUNT';

interface SessionData {
  state: UssdState;
  pin?: string;           // Temporary PIN storage during setup
  commodity?: string;     // Selected commodity
  amount?: number;        // Project amount
  projectId?: string;     // For status check
}

@Injectable()
export class UssdService {
  private redis: Redis;

  // Supported commodities with max funding
  private readonly commodities = [
    { id: 'gula_semut', name: 'Gula Semut', maxFund: 2000000 },
    { id: 'padi', name: 'Padi', maxFund: 5000000 },
    { id: 'jagung', name: 'Jagung', maxFund: 3000000 },
  ];

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

  private hashPin(pin: string): string {
    return crypto.createHash('sha256').update(pin).digest('hex');
  }

  // ============ Menu Builders ============

  private mainMenu(balance: number = 0): string {
    const lines = [
      'CON Selamat Datang di TaniFi',
      `Saldo: Rp ${balance.toLocaleString('id-ID')}`,
      '',
      '1. Ajukan Modal',
      '2. Cek Status Proyek',
      '3. Info Akun',
      '00. Keluar',
    ];
    return lines.join('\n');
  }

  private registrationMenu(): string {
    return [
      'CON Selamat Datang di TaniFi!',
      'Anda belum terdaftar.',
      '',
      '1. Daftar Sekarang',
      '00. Keluar',
    ].join('\n');
  }

  private pinSetupPrompt(): string {
    return [
      'CON Buat PIN 6 digit untuk',
      'mengamankan akun Anda.',
      '',
      'Masukkan PIN baru:',
    ].join('\n');
  }

  private pinConfirmPrompt(): string {
    return 'CON Konfirmasi PIN Anda:';
  }

  private pinVerifyPrompt(): string {
    return 'CON Masukkan PIN Anda:';
  }

  private commodityMenu(): string {
    const lines = [
      'CON Pilih Komoditas:',
      '',
    ];
    this.commodities.forEach((c, i) => {
      lines.push(`${i + 1}. ${c.name} (Maks Rp ${(c.maxFund / 1000000).toFixed(0)}jt)`);
    });
    lines.push('');
    lines.push('0. Kembali');
    return lines.join('\n');
  }

  private amountPrompt(commodity: string): string {
    const com = this.commodities.find(c => c.id === commodity);
    const maxStr = com ? `Maks Rp ${com.maxFund.toLocaleString('id-ID')}` : '';
    return [
      'CON Masukkan jumlah modal',
      `yang dibutuhkan (${maxStr}):`,
      '',
      'Contoh: 1000000',
      '',
      '0. Kembali',
    ].join('\n');
  }

  private confirmationPrompt(commodity: string, amount: number): string {
    const com = this.commodities.find(c => c.id === commodity);
    const name = com?.name || commodity;
    return [
      'CON Konfirmasi Pengajuan Modal:',
      '',
      `Komoditas: ${name}`,
      `Jumlah: Rp ${amount.toLocaleString('id-ID')}`,
      `Nisbah: 30% Petani / 69% Investor`,
      `Fee Platform: 1%`,
      '',
      '1. Konfirmasi',
      '2. Batal',
    ].join('\n');
  }

  private accountInfo(user: any, projectCount: number): string {
    const verified = user?.kycStatus === 'VERIFIED' ? 'Terverifikasi' : 'Belum Verifikasi';
    const balance = user?.balance || 0;
    return [
      'CON Info Akun TaniFi',
      '',
      `Status KYC: ${verified}`,
      `Saldo IDRX: Rp ${balance.toLocaleString('id-ID')}`,
      `Total Proyek: ${projectCount}`,
      `Skor Reputasi: ${user?.reputationScore || 100}`,
      '',
      '0. Kembali',
    ].join('\n');
  }

  private projectStatusList(projects: any[]): string {
    if (projects.length === 0) {
      return [
        'CON Status Proyek',
        '',
        'Belum ada proyek aktif.',
        '',
        '0. Kembali',
      ].join('\n');
    }

    const lines = ['CON Proyek Anda:', ''];
    projects.slice(0, 3).forEach((p, i) => {
      const status = this.translateStatus(p.status);
      const amt = (p.amount / 1000000).toFixed(1);
      lines.push(`${i + 1}. Rp ${amt}jt - ${status}`);
    });
    if (projects.length > 3) {
      lines.push(`...dan ${projects.length - 3} lainnya`);
    }
    lines.push('');
    lines.push('0. Kembali');
    return lines.join('\n');
  }

  private translateStatus(status: string): string {
    const map: Record<string, string> = {
      created: 'Menunggu',
      confirmed: 'Disetujui',
      funded: 'Terdanai',
      active: 'Berjalan',
      harvested: 'Panen',
      completed: 'Selesai',
      failed: 'Gagal',
    };
    return map[status] || status;
  }

  private exit(): string {
    return 'END Terima kasih telah menggunakan TaniFi. Selamat bertani!';
  }

  private success(message: string): string {
    return `END ${message}`;
  }

  private error(message: string): string {
    return `END Error: ${message}`;
  }

  // ============ Main Handler ============

  async handle(input: UssdInput): Promise<{ message: string; stateBefore: string; stateAfter: string }> {
    const key = `ussd:${input.sessionId}`;
    const text = (input.text ?? '').trim();
    const phoneHash = this.phoneHash(input.phoneNumber);

    // Get or initialize session
    const raw = await this.redis.get(key);
    const sess: SessionData = raw ? JSON.parse(raw) : { state: 'START' };
    const stateBefore = sess.state;

    // Get user if exists
    const user = await this.prisma.user.findUnique({
      where: { phoneHash },
    });

    // Parse input - get the last selection in multi-level input
    const parts = text.split('*');
    const last = parts[parts.length - 1] ?? '';

    let result: { message: string; stateAfter: UssdState | 'END' };

    try {
      result = await this.processState(sess, last, phoneHash, user, key);
    } catch (err) {
      result = { message: this.error('Terjadi kesalahan. Coba lagi.'), stateAfter: 'END' };
    }

    // Update session if not ended
    if (result.stateAfter !== 'END') {
      sess.state = result.stateAfter as UssdState;
      await this.redis.set(key, JSON.stringify(sess), 'EX', this.ttl());
    } else {
      await this.redis.del(key);
    }

    return {
      message: result.message,
      stateBefore,
      stateAfter: result.stateAfter,
    };
  }

  private async processState(
    sess: SessionData,
    input: string,
    phoneHash: string,
    user: any,
    redisKey: string,
  ): Promise<{ message: string; stateAfter: UssdState | 'END' }> {

    // Global exit
    if (input === '00') {
      return { message: this.exit(), stateAfter: 'END' };
    }

    switch (sess.state) {
      case 'START':
        return this.handleStart(sess, input, phoneHash, user);

      case 'AWAIT_PIN_SETUP':
        return this.handlePinSetup(sess, input, phoneHash);

      case 'AWAIT_PIN_CONFIRM':
        return this.handlePinConfirm(sess, input, phoneHash);

      case 'AWAIT_PIN_VERIFY':
        return this.handlePinVerify(sess, input, phoneHash, user);

      case 'SELECT_COMMODITY':
        return this.handleCommoditySelect(sess, input);

      case 'AWAIT_AMOUNT':
        return this.handleAmountInput(sess, input);

      case 'CONFIRM_PROJECT':
        return this.handleProjectConfirm(sess, input, phoneHash);

      case 'VIEW_STATUS':
        return this.handleViewStatus(sess, input, phoneHash);

      case 'VIEW_ACCOUNT':
        return this.handleViewAccount(sess, input, phoneHash);

      default:
        return { message: this.mainMenu(), stateAfter: 'START' };
    }
  }

  // ============ State Handlers ============

  private async handleStart(
    sess: SessionData,
    input: string,
    phoneHash: string,
    user: any,
  ): Promise<{ message: string; stateAfter: UssdState | 'END' }> {

    // New session - show menu
    if (input === '') {
      if (!user) {
        return { message: this.registrationMenu(), stateAfter: 'START' };
      }
      const balance = user.balance || 0;
      return { message: this.mainMenu(balance), stateAfter: 'START' };
    }

    // If not registered and selecting Register
    if (!user && input === '1') {
      return { message: this.pinSetupPrompt(), stateAfter: 'AWAIT_PIN_SETUP' };
    }

    // If not registered - block other options
    if (!user) {
      return { message: this.registrationMenu(), stateAfter: 'START' };
    }

    // Registered user menu options
    switch (input) {
      case '1': // Ajukan Modal - requires PIN
        sess.state = 'AWAIT_PIN_VERIFY';
        return { message: this.pinVerifyPrompt(), stateAfter: 'AWAIT_PIN_VERIFY' };

      case '2': // Cek Status
        return this.showProjectStatus(phoneHash);

      case '3': // Info Akun
        return this.showAccountInfo(phoneHash);

      default:
        return { message: this.mainMenu(user.balance || 0), stateAfter: 'START' };
    }
  }

  private async handlePinSetup(
    sess: SessionData,
    input: string,
    phoneHash: string,
  ): Promise<{ message: string; stateAfter: UssdState | 'END' }> {
    // Validate PIN format
    if (!/^\d{6}$/.test(input)) {
      return {
        message: 'CON PIN harus 6 digit angka.\n\nMasukkan PIN:',
        stateAfter: 'AWAIT_PIN_SETUP',
      };
    }

    sess.pin = input;
    return { message: this.pinConfirmPrompt(), stateAfter: 'AWAIT_PIN_CONFIRM' };
  }

  private async handlePinConfirm(
    sess: SessionData,
    input: string,
    phoneHash: string,
  ): Promise<{ message: string; stateAfter: UssdState | 'END' }> {
    if (input !== sess.pin) {
      return {
        message: 'CON PIN tidak cocok.\n\nMasukkan PIN baru:',
        stateAfter: 'AWAIT_PIN_SETUP',
      };
    }

    // Create user with hashed PIN
    const pinHash = this.hashPin(input);
    await this.prisma.user.create({
      data: {
        phoneHash,
        encryptedPin: pinHash,
        kycStatus: 'PENDING',
        reputationScore: 100,
        balance: 0,
      },
    });

    delete sess.pin;
    return {
      message: this.success('Pendaftaran berhasil! Silakan login kembali dengan *777#'),
      stateAfter: 'END',
    };
  }

  private async handlePinVerify(
    sess: SessionData,
    input: string,
    phoneHash: string,
    user: any,
  ): Promise<{ message: string; stateAfter: UssdState | 'END' }> {
    if (input === '0') {
      return { message: this.mainMenu(user?.balance || 0), stateAfter: 'START' };
    }

    const pinHash = this.hashPin(input);
    if (!user?.encryptedPin || user.encryptedPin !== pinHash) {
      return {
        message: 'CON PIN salah. Coba lagi:\n\n0. Kembali',
        stateAfter: 'AWAIT_PIN_VERIFY',
      };
    }

    // PIN verified - show commodity selection
    return { message: this.commodityMenu(), stateAfter: 'SELECT_COMMODITY' };
  }

  private async handleCommoditySelect(
    sess: SessionData,
    input: string,
  ): Promise<{ message: string; stateAfter: UssdState | 'END' }> {
    if (input === '0') {
      return { message: this.mainMenu(), stateAfter: 'START' };
    }

    const idx = parseInt(input) - 1;
    if (idx < 0 || idx >= this.commodities.length) {
      return { message: this.commodityMenu(), stateAfter: 'SELECT_COMMODITY' };
    }

    sess.commodity = this.commodities[idx].id;
    return {
      message: this.amountPrompt(sess.commodity),
      stateAfter: 'AWAIT_AMOUNT',
    };
  }

  private async handleAmountInput(
    sess: SessionData,
    input: string,
  ): Promise<{ message: string; stateAfter: UssdState | 'END' }> {
    if (input === '0') {
      return { message: this.commodityMenu(), stateAfter: 'SELECT_COMMODITY' };
    }

    const amount = parseInt(input);
    const commodity = this.commodities.find(c => c.id === sess.commodity);

    if (isNaN(amount) || amount <= 0) {
      return {
        message: 'CON Jumlah tidak valid.\n\nMasukkan angka (contoh: 1000000):',
        stateAfter: 'AWAIT_AMOUNT',
      };
    }

    if (commodity && amount > commodity.maxFund) {
      return {
        message: `CON Jumlah melebihi batas.\nMaksimal: Rp ${commodity.maxFund.toLocaleString('id-ID')}\n\nMasukkan jumlah:`,
        stateAfter: 'AWAIT_AMOUNT',
      };
    }

    sess.amount = amount;
    return {
      message: this.confirmationPrompt(sess.commodity!, amount),
      stateAfter: 'CONFIRM_PROJECT',
    };
  }

  private async handleProjectConfirm(
    sess: SessionData,
    input: string,
    phoneHash: string,
  ): Promise<{ message: string; stateAfter: UssdState | 'END' }> {
    if (input === '2') {
      // Cancel
      return { message: this.mainMenu(), stateAfter: 'START' };
    }

    if (input !== '1') {
      return {
        message: this.confirmationPrompt(sess.commodity!, sess.amount!),
        stateAfter: 'CONFIRM_PROJECT',
      };
    }

    // Create project
    const user = await this.prisma.user.findUnique({
      where: { phoneHash },
    });

    if (!user) {
      return { message: this.error('User tidak ditemukan.'), stateAfter: 'END' };
    }

    const txHash = `0x${crypto.randomBytes(32).toString('hex')}`;
    const commodity = this.commodities.find(c => c.id === sess.commodity);

    const project = await this.prisma.project.create({
      data: {
        userId: user.id,
        amount: sess.amount!,
        commodity: sess.commodity,
        status: 'created',
        txs: {
          create: {
            kind: 'create_project',
            status: 'pending',
            txHash,
          },
        },
      },
      include: { txs: true },
    });

    // Simulate blockchain confirmation after 10 seconds
    this.simulateConfirmation(project.id, project.txs[0]?.id);

    const successMsg = [
      'Pengajuan Modal Berhasil!',
      '',
      `Komoditas: ${commodity?.name}`,
      `Jumlah: Rp ${sess.amount!.toLocaleString('id-ID')}`,
      `ID Proyek: ${project.id.slice(0, 8)}...`,
      '',
      'Status: Menunggu investor',
      'Anda akan menerima SMS konfirmasi.',
    ].join('\n');

    return { message: this.success(successMsg), stateAfter: 'END' };
  }

  private async showProjectStatus(
    phoneHash: string,
  ): Promise<{ message: string; stateAfter: UssdState | 'END' }> {
    const user = await this.prisma.user.findUnique({
      where: { phoneHash },
      include: {
        projects: {
          orderBy: { createdAt: 'desc' },
          take: 5,
        },
      },
    });

    return {
      message: this.projectStatusList(user?.projects || []),
      stateAfter: 'VIEW_STATUS',
    };
  }

  private async handleViewStatus(
    sess: SessionData,
    input: string,
    phoneHash: string,
  ): Promise<{ message: string; stateAfter: UssdState | 'END' }> {
    if (input === '0') {
      return { message: this.mainMenu(), stateAfter: 'START' };
    }
    return this.showProjectStatus(phoneHash);
  }

  private async showAccountInfo(
    phoneHash: string,
  ): Promise<{ message: string; stateAfter: UssdState | 'END' }> {
    const user = await this.prisma.user.findUnique({
      where: { phoneHash },
      include: {
        _count: { select: { projects: true } },
      },
    });

    return {
      message: this.accountInfo(user, user?._count?.projects || 0),
      stateAfter: 'VIEW_ACCOUNT',
    };
  }

  private async handleViewAccount(
    sess: SessionData,
    input: string,
    phoneHash: string,
  ): Promise<{ message: string; stateAfter: UssdState | 'END' }> {
    if (input === '0') {
      return { message: this.mainMenu(), stateAfter: 'START' };
    }
    return this.showAccountInfo(phoneHash);
  }

  // ============ Helpers ============

  private simulateConfirmation(projectId: string, txId?: string): void {
    setTimeout(async () => {
      try {
        if (txId) {
          await this.prisma.transaction.update({
            where: { id: txId },
            data: { status: 'success' },
          });
        }
        await this.prisma.project.update({
          where: { id: projectId },
          data: { status: 'confirmed' },
        });
      } catch {
        // Ignore errors in demo simulation
      }
    }, 10000);
  }
}
