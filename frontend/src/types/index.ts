// Project state enum matching smart contract
export enum ProjectState {
  FUNDRAISING = 0,
  ACTIVE = 1,
  HARVESTED = 2,
  FAILED = 3,
  COMPLETED = 4,
}

export interface Project {
  id: number;
  farmer: string;
  cooperative: string;
  approvedVendor: string;
  targetAmount: bigint;
  fundedAmount: bigint;
  farmerShareBps: number;
  investorShareBps: number;
  startTime: number;
  harvestTime: number;
  harvestRevenue: bigint;
  state: ProjectState;
  ipfsMetadata: string;
}

export interface ProjectDisplay {
  id: number;
  farmer: string;
  cooperative: string;
  targetAmount: string;
  fundedAmount: string;
  fundingProgress: number;
  farmerShare: number;
  investorShare: number;
  harvestDate: string;
  state: ProjectState;
  stateName: string;
  commodity?: string;
}

export interface Investment {
  projectId: number;
  amount: bigint;
  expectedReturn: bigint;
}

export interface WalletState {
  address: string | null;
  balance: bigint;
  isConnected: boolean;
  chainId: number | null;
}
