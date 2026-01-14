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

// Contract addresses on Lisk Sepolia (Deployed January 2026)
export const CONTRACT_ADDRESSES = {
  TANI_VAULT: process.env.NEXT_PUBLIC_TANI_VAULT_ADDRESS || '0xB39c94B718A75c3005F06f977224cF52AD7cAe49',
  IDRX: process.env.NEXT_PUBLIC_IDRX_ADDRESS || '0x01653fA9F9e9411ac3028f6b4A54f39D68edEA44',
  FARMER_REGISTRY: process.env.NEXT_PUBLIC_FARMER_REGISTRY_ADDRESS || '0x01A0789ae050370AC87d38Fd42b5371Ea0128bA4',
};

// Network configuration
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
