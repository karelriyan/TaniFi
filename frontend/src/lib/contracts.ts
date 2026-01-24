// Contract ABIs and addresses for TaniFi Protocol

export const TANI_VAULT_ABI = [
  'event ProjectCreated(uint256 indexed projectId, address indexed farmer, address cooperative, uint256 targetAmount)',
  'event InvestmentReceived(uint256 indexed projectId, address indexed investor, uint256 amount)',
  'function projectCount() view returns (uint256)',
  'function getProject(uint256 _projectId) view returns (tuple(uint256 id, address farmer, address cooperative, address approvedVendor, uint256 targetAmount, uint256 fundedAmount, uint256 farmerShareBps, uint256 investorShareBps, uint256 startTime, uint256 harvestTime, uint256 harvestRevenue, uint8 state, string ipfsMetadata))',
  'function getInvestment(uint256 _projectId, address _investor) view returns (uint256)',
  'function invest(uint256 _projectId, uint256 _amount)',
  'function withdrawReturns(uint256 _projectId)',
  'function calculateExpectedReturns(uint256 _projectId, address _investor, uint256 _expectedRevenue) view returns (uint256)',
  'function stablecoin() view returns (address)',
] as const;

export const IDRX_ABI = [
  'function name() view returns (string)',
  'function symbol() view returns (string)',
  'function decimals() view returns (uint8)',
  'function balanceOf(address account) view returns (uint256)',
  'function allowance(address owner, address spender) view returns (uint256)',
  'function approve(address spender, uint256 amount) returns (bool)',
  'function transfer(address to, uint256 amount) returns (bool)',
  'function faucet()',
] as const;

// Contract addresses on Base Sepolia (Deployed January 24, 2026)
export const CONTRACT_ADDRESSES = {
  TANI_VAULT: process.env.NEXT_PUBLIC_TANI_VAULT_ADDRESS || '0xEAD7D9095e16fA298d5d66ab129d28638a1deb50',
  IDRX: process.env.NEXT_PUBLIC_IDRX_ADDRESS || '0xe22c8b828A60c95F9Ca3ad9275B30C3F58Bd0110',
  FARMER_REGISTRY: process.env.NEXT_PUBLIC_FARMER_REGISTRY_ADDRESS || '0x0fc35d36cAE59077739f93B513F9a5f5a52E4409',
};

// Network configuration for Base Sepolia
export const BASE_SEPOLIA = {
  chainId: 84532,
  chainIdHex: '0x14a34',
  chainName: 'Base Sepolia',
  nativeCurrency: {
    name: 'Ethereum',
    symbol: 'ETH',
    decimals: 18,
  },
  rpcUrls: ['https://sepolia.base.org'],
  blockExplorerUrls: ['https://sepolia.basescan.org'],
};

// Legacy: Lisk Sepolia (deprecated - keeping for reference)
export const LISK_SEPOLIA = {
  chainId: 4202,
  chainIdHex: '0x106a',
  chainName: 'Lisk Sepolia',
  nativeCurrency: {
    name: 'Sepolia ETH',
    symbol: 'ETH',
    decimals: 18,
  },
  rpcUrls: ['https://rpc.sepolia-api.lisk.com'],
  blockExplorerUrls: ['https://sepolia-blockscout.lisk.com'],
};
