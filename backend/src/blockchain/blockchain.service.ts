import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { ethers, Contract, Wallet, Provider, JsonRpcProvider } from 'ethers';
import {
  TANI_VAULT_ABI,
  IDRX_ABI,
  FARMER_REGISTRY_ABI,
  CONTRACT_ADDRESSES,
  NETWORK_CONFIG,
} from './abis';

// Types for contract interactions
export interface ProjectData {
  id: bigint;
  farmer: string;
  cooperative: string;
  approvedVendor: string;
  targetAmount: bigint;
  fundedAmount: bigint;
  farmerShareBps: bigint;
  investorShareBps: bigint;
  startTime: bigint;
  harvestTime: bigint;
  harvestRevenue: bigint;
  state: number;
  ipfsMetadata: string;
}

export interface FarmerProfile {
  tokenId: bigint;
  walletAddress: string;
  phoneHash: string;
  reputationScore: bigint;
  isKYCVerified: boolean;
  registeredAt: bigint;
  completedProjects: bigint;
  failedProjects: bigint;
  metadataURI: string;
}

export enum ProjectState {
  FUNDRAISING = 0,
  ACTIVE = 1,
  HARVESTED = 2,
  FAILED = 3,
  COMPLETED = 4,
}

@Injectable()
export class BlockchainService implements OnModuleInit {
  private readonly logger = new Logger(BlockchainService.name);

  private provider: JsonRpcProvider;
  private operatorWallet: Wallet;
  private taniVault: Contract;
  private idrx: Contract;
  private farmerRegistry: Contract;

  // Read-only contract instances (no signer needed)
  private taniVaultReadOnly: Contract;
  private idrxReadOnly: Contract;
  private farmerRegistryReadOnly: Contract;

  constructor(private configService: ConfigService) {}

  async onModuleInit() {
    await this.initializeProvider();
    await this.initializeContracts();
  }

  private async initializeProvider() {
    const rpcUrl = this.configService.get<string>(
      'BASE_RPC_URL',
      NETWORK_CONFIG.BASE_SEPOLIA.rpcUrl,
    );

    this.provider = new JsonRpcProvider(rpcUrl);

    // Verify connection
    try {
      const network = await this.provider.getNetwork();
      this.logger.log(
        `Connected to network: ${network.name} (chainId: ${network.chainId})`,
      );
    } catch (error) {
      this.logger.error('Failed to connect to blockchain network', error);
    }

    // Initialize operator wallet if private key is provided
    const privateKey = this.configService.get<string>('OPERATOR_PRIVATE_KEY');
    if (privateKey) {
      this.operatorWallet = new Wallet(privateKey, this.provider);
      this.logger.log(`Operator wallet: ${this.operatorWallet.address}`);
    } else {
      this.logger.warn(
        'OPERATOR_PRIVATE_KEY not set - write operations will fail',
      );
    }
  }

  private async initializeContracts() {
    const addresses = CONTRACT_ADDRESSES.BASE_SEPOLIA;

    // Read-only instances
    this.taniVaultReadOnly = new Contract(
      addresses.TANI_VAULT,
      TANI_VAULT_ABI,
      this.provider,
    );
    this.idrxReadOnly = new Contract(addresses.IDRX, IDRX_ABI, this.provider);
    this.farmerRegistryReadOnly = new Contract(
      addresses.FARMER_REGISTRY,
      FARMER_REGISTRY_ABI,
      this.provider,
    );

    // Write-capable instances (if wallet available)
    if (this.operatorWallet) {
      this.taniVault = new Contract(
        addresses.TANI_VAULT,
        TANI_VAULT_ABI,
        this.operatorWallet,
      );
      this.idrx = new Contract(addresses.IDRX, IDRX_ABI, this.operatorWallet);
      this.farmerRegistry = new Contract(
        addresses.FARMER_REGISTRY,
        FARMER_REGISTRY_ABI,
        this.operatorWallet,
      );
    }

    this.logger.log('Contracts initialized');
  }

  // ============ IDRX Token Functions ============

  /**
   * Get IDRX balance for an address
   */
  async getIDRXBalance(address: string): Promise<bigint> {
    try {
      return await this.idrxReadOnly.balanceOf(address);
    } catch (error) {
      this.logger.error(`Failed to get IDRX balance for ${address}`, error);
      throw error;
    }
  }

  /**
   * Get IDRX balance in human-readable format (with 2 decimals)
   */
  async getIDRXBalanceFormatted(address: string): Promise<string> {
    const balance = await this.getIDRXBalance(address);
    return ethers.formatUnits(balance, 2);
  }

  /**
   * Check allowance for TaniVault
   */
  async getIDRXAllowance(owner: string, spender: string): Promise<bigint> {
    return await this.idrxReadOnly.allowance(owner, spender);
  }

  /**
   * Mint IDRX tokens to an address (operator only)
   */
  async mintIDRX(to: string, amount: bigint): Promise<string> {
    if (!this.idrx) throw new Error('Operator wallet not configured');

    const tx = await this.idrx.mint(to, amount);
    await tx.wait();

    this.logger.log(`Minted ${amount} IDRX to ${to}, tx: ${tx.hash}`);
    return tx.hash;
  }

  // ============ TaniVault Functions ============

  /**
   * Get project details by ID
   */
  async getProject(projectId: number): Promise<ProjectData | null> {
    try {
      const project = await this.taniVaultReadOnly.getProject(projectId);
      return {
        id: project.id,
        farmer: project.farmer,
        cooperative: project.cooperative,
        approvedVendor: project.approvedVendor,
        targetAmount: project.targetAmount,
        fundedAmount: project.fundedAmount,
        farmerShareBps: project.farmerShareBps,
        investorShareBps: project.investorShareBps,
        startTime: project.startTime,
        harvestTime: project.harvestTime,
        harvestRevenue: project.harvestRevenue,
        state: Number(project.state),
        ipfsMetadata: project.ipfsMetadata,
      };
    } catch (error) {
      this.logger.error(`Failed to get project ${projectId}`, error);
      return null;
    }
  }

  /**
   * Get total project count
   */
  async getProjectCount(): Promise<number> {
    const count = await this.taniVaultReadOnly.projectCount();
    return Number(count);
  }

  /**
   * Get investment amount for an investor in a project
   */
  async getInvestment(projectId: number, investor: string): Promise<bigint> {
    return await this.taniVaultReadOnly.getInvestment(projectId, investor);
  }

  /**
   * Calculate expected returns for an investor
   */
  async calculateExpectedReturns(
    projectId: number,
    investor: string,
    expectedRevenue: bigint,
  ): Promise<bigint> {
    return await this.taniVaultReadOnly.calculateExpectedReturns(
      projectId,
      investor,
      expectedRevenue,
    );
  }

  /**
   * Get vault total balance
   */
  async getVaultBalance(): Promise<bigint> {
    return await this.taniVaultReadOnly.getVaultBalance();
  }

  /**
   * Create a new agricultural project (operator/cooperative only)
   */
  async createProject(
    farmer: string,
    approvedVendor: string,
    targetAmount: bigint,
    farmerShareBps: number,
    harvestTime: number,
    ipfsMetadata: string,
  ): Promise<{ projectId: number; txHash: string }> {
    if (!this.taniVault) throw new Error('Operator wallet not configured');

    const tx = await this.taniVault.createProject(
      farmer,
      approvedVendor,
      targetAmount,
      farmerShareBps,
      harvestTime,
      ipfsMetadata,
    );

    const receipt = await tx.wait();

    // Parse ProjectCreated event to get projectId
    const event = receipt.logs.find(
      (log: any) =>
        log.topics[0] ===
        ethers.id(
          'ProjectCreated(uint256,address,address,uint256)',
        ),
    );

    let projectId = 0;
    if (event) {
      projectId = Number(BigInt(event.topics[1]));
    }

    this.logger.log(`Project created: ${projectId}, tx: ${tx.hash}`);
    return { projectId, txHash: tx.hash };
  }

  /**
   * Invest in a project (requires user's wallet - typically via frontend)
   */
  async invest(
    projectId: number,
    amount: bigint,
    investorWallet: Wallet,
  ): Promise<string> {
    const vaultWithSigner = this.taniVaultReadOnly.connect(
      investorWallet,
    ) as Contract;
    const tx = await vaultWithSigner.invest(projectId, amount);
    await tx.wait();

    this.logger.log(
      `Investment of ${amount} in project ${projectId}, tx: ${tx.hash}`,
    );
    return tx.hash;
  }

  /**
   * Disburse funds to vendor (cooperative/operator only)
   */
  async disburseToVendor(projectId: number): Promise<string> {
    if (!this.taniVault) throw new Error('Operator wallet not configured');

    const tx = await this.taniVault.disburseToVendor(projectId);
    await tx.wait();

    this.logger.log(`Funds disbursed for project ${projectId}, tx: ${tx.hash}`);
    return tx.hash;
  }

  /**
   * Report harvest revenue (cooperative only)
   */
  async reportHarvest(projectId: number, revenue: bigint): Promise<string> {
    if (!this.taniVault) throw new Error('Operator wallet not configured');

    const tx = await this.taniVault.reportHarvest(projectId, revenue);
    await tx.wait();

    this.logger.log(
      `Harvest reported for project ${projectId}: ${revenue}, tx: ${tx.hash}`,
    );
    return tx.hash;
  }

  /**
   * Finalize harvest and distribute profits
   */
  async finalizeHarvest(projectId: number): Promise<string> {
    if (!this.taniVault) throw new Error('Operator wallet not configured');

    const tx = await this.taniVault.finalizeHarvest(projectId);
    await tx.wait();

    this.logger.log(`Harvest finalized for project ${projectId}, tx: ${tx.hash}`);
    return tx.hash;
  }

  /**
   * Mark project as failed
   */
  async markProjectFailed(projectId: number, reason: string): Promise<string> {
    if (!this.taniVault) throw new Error('Operator wallet not configured');

    const tx = await this.taniVault.markProjectFailed(projectId, reason);
    await tx.wait();

    this.logger.log(
      `Project ${projectId} marked as failed: ${reason}, tx: ${tx.hash}`,
    );
    return tx.hash;
  }

  // ============ Farmer Registry Functions ============

  /**
   * Check if a farmer is registered
   */
  async isFarmerRegistered(address: string): Promise<boolean> {
    return await this.farmerRegistryReadOnly.isRegistered(address);
  }

  /**
   * Check if a farmer is KYC verified
   */
  async isFarmerVerified(address: string): Promise<boolean> {
    return await this.farmerRegistryReadOnly.isVerified(address);
  }

  /**
   * Get farmer's reputation score
   */
  async getFarmerReputation(address: string): Promise<number> {
    const score = await this.farmerRegistryReadOnly.getReputation(address);
    return Number(score);
  }

  /**
   * Get full farmer profile
   */
  async getFarmerProfile(address: string): Promise<FarmerProfile | null> {
    try {
      const profile = await this.farmerRegistryReadOnly.getFarmerProfile(address);
      return {
        tokenId: profile.tokenId,
        walletAddress: profile.walletAddress,
        phoneHash: profile.phoneHash,
        reputationScore: profile.reputationScore,
        isKYCVerified: profile.isKYCVerified,
        registeredAt: profile.registeredAt,
        completedProjects: profile.completedProjects,
        failedProjects: profile.failedProjects,
        metadataURI: profile.metadataURI,
      };
    } catch (error) {
      this.logger.error(`Failed to get farmer profile for ${address}`, error);
      return null;
    }
  }

  /**
   * Get wallet address from phone hash
   */
  async getWalletFromPhoneHash(phoneHash: string): Promise<string> {
    return await this.farmerRegistryReadOnly.phoneHashToWallet(phoneHash);
  }

  /**
   * Register a new farmer (KYC admin only)
   */
  async registerFarmer(
    farmerAddress: string,
    phoneHash: string,
    metadataURI: string,
  ): Promise<{ tokenId: number; txHash: string }> {
    if (!this.farmerRegistry)
      throw new Error('Operator wallet not configured');

    const tx = await this.farmerRegistry.registerFarmer(
      farmerAddress,
      phoneHash,
      metadataURI,
    );

    const receipt = await tx.wait();

    // Parse FarmerRegistered event
    const event = receipt.logs.find(
      (log: any) =>
        log.topics[0] ===
        ethers.id('FarmerRegistered(uint256,address,bytes32)'),
    );

    let tokenId = 0;
    if (event) {
      tokenId = Number(BigInt(event.topics[1]));
    }

    this.logger.log(`Farmer registered: ${farmerAddress}, tokenId: ${tokenId}`);
    return { tokenId, txHash: tx.hash };
  }

  /**
   * Verify farmer's KYC
   */
  async verifyFarmer(farmerAddress: string): Promise<string> {
    if (!this.farmerRegistry)
      throw new Error('Operator wallet not configured');

    const tx = await this.farmerRegistry.verifyFarmer(farmerAddress);
    await tx.wait();

    this.logger.log(`Farmer verified: ${farmerAddress}, tx: ${tx.hash}`);
    return tx.hash;
  }

  // ============ Utility Functions ============

  /**
   * Generate phone hash (keccak256)
   */
  hashPhoneNumber(phoneNumber: string): string {
    // Normalize phone number (remove spaces, dashes, etc.)
    const normalized = phoneNumber.replace(/[\s\-\(\)]/g, '');
    return ethers.keccak256(ethers.toUtf8Bytes(normalized));
  }

  /**
   * Parse IDRX amount (2 decimals)
   */
  parseIDRX(amount: string | number): bigint {
    return ethers.parseUnits(amount.toString(), 2);
  }

  /**
   * Format IDRX amount (2 decimals)
   */
  formatIDRX(amount: bigint): string {
    return ethers.formatUnits(amount, 2);
  }

  /**
   * Get current block timestamp
   */
  async getCurrentTimestamp(): Promise<number> {
    const block = await this.provider.getBlock('latest');
    return block?.timestamp || Math.floor(Date.now() / 1000);
  }

  /**
   * Wait for transaction confirmation
   */
  async waitForTransaction(
    txHash: string,
    confirmations: number = 1,
  ): Promise<boolean> {
    try {
      const receipt = await this.provider.waitForTransaction(
        txHash,
        confirmations,
      );
      return receipt?.status === 1;
    } catch (error) {
      this.logger.error(`Transaction ${txHash} failed`, error);
      return false;
    }
  }

  /**
   * Get transaction status
   */
  async getTransactionStatus(
    txHash: string,
  ): Promise<'pending' | 'success' | 'failed' | 'not_found'> {
    try {
      const receipt = await this.provider.getTransactionReceipt(txHash);
      if (!receipt) return 'pending';
      return receipt.status === 1 ? 'success' : 'failed';
    } catch {
      return 'not_found';
    }
  }

  /**
   * Get provider for external use
   */
  getProvider(): Provider {
    return this.provider;
  }

  /**
   * Get operator address
   */
  getOperatorAddress(): string | null {
    return this.operatorWallet?.address || null;
  }
}
