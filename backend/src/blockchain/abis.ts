// Contract ABIs for TaniFi Protocol
// Auto-generated from Solidity contracts

export const TANI_VAULT_ABI = [
  // Events
  'event ProjectCreated(uint256 indexed projectId, address indexed farmer, address cooperative, uint256 targetAmount)',
  'event InvestmentReceived(uint256 indexed projectId, address indexed investor, uint256 amount)',
  'event FundsDisbursed(uint256 indexed projectId, address indexed vendor, uint256 amount)',
  'event HarvestReported(uint256 indexed projectId, uint256 revenue)',
  'event ProfitDistributed(uint256 indexed projectId, uint256 farmerShare, uint256 investorPool, uint256 platformFee)',
  'event InvestorWithdrawal(uint256 indexed projectId, address indexed investor, uint256 amount)',
  'event ProjectFailed(uint256 indexed projectId, string reason)',

  // Read Functions
  'function projectCount() view returns (uint256)',
  'function getProject(uint256 _projectId) view returns (tuple(uint256 id, address farmer, address cooperative, address approvedVendor, uint256 targetAmount, uint256 fundedAmount, uint256 farmerShareBps, uint256 investorShareBps, uint256 startTime, uint256 harvestTime, uint256 harvestRevenue, uint8 state, string ipfsMetadata))',
  'function getInvestment(uint256 _projectId, address _investor) view returns (uint256)',
  'function getVaultBalance() view returns (uint256)',
  'function investments(uint256, address) view returns (uint256)',
  'function totalInvested(uint256) view returns (uint256)',
  'function isCooperative(address _addr) view returns (bool)',
  'function isOperator(address _addr) view returns (bool)',
  'function calculateExpectedReturns(uint256 _projectId, address _investor, uint256 _expectedRevenue) view returns (uint256)',
  'function stablecoin() view returns (address)',
  'function treasury() view returns (address)',
  'function owner() view returns (address)',
  'function paused() view returns (bool)',

  // Write Functions
  'function createProject(address _farmer, address _approvedVendor, uint256 _targetAmount, uint256 _farmerShareBps, uint256 _harvestTime, string calldata _ipfsMetadata) returns (uint256)',
  'function invest(uint256 _projectId, uint256 _amount)',
  'function disburseToVendor(uint256 _projectId)',
  'function reportHarvest(uint256 _projectId, uint256 _revenue)',
  'function finalizeHarvest(uint256 _projectId)',
  'function withdrawReturns(uint256 _projectId)',
  'function markProjectFailed(uint256 _projectId, string calldata _reason)',
  'function refundInvestors(uint256 _projectId)',

  // Admin Functions
  'function addCooperative(address _cooperative)',
  'function removeCooperative(address _cooperative)',
  'function addOperator(address _operator)',
  'function removeOperator(address _operator)',
  'function pause()',
  'function unpause()',
] as const;

export const IDRX_ABI = [
  // Events
  'event Transfer(address indexed from, address indexed to, uint256 value)',
  'event Approval(address indexed owner, address indexed spender, uint256 value)',

  // Read Functions
  'function name() view returns (string)',
  'function symbol() view returns (string)',
  'function decimals() view returns (uint8)',
  'function totalSupply() view returns (uint256)',
  'function balanceOf(address account) view returns (uint256)',
  'function allowance(address owner, address spender) view returns (uint256)',

  // Write Functions
  'function transfer(address to, uint256 amount) returns (bool)',
  'function approve(address spender, uint256 amount) returns (bool)',
  'function transferFrom(address from, address to, uint256 amount) returns (bool)',
  'function mint(address to, uint256 amount)',
  'function burn(uint256 amount)',
  'function faucet()',
  'function faucetTo(address to, uint256 amount)',
] as const;

export const FARMER_REGISTRY_ABI = [
  // Events
  'event Transfer(address indexed from, address indexed to, uint256 indexed tokenId)',
  'event FarmerRegistered(uint256 indexed tokenId, address indexed farmer, bytes32 phoneHash)',
  'event FarmerVerified(address indexed farmer, uint256 tokenId)',
  'event ReputationUpdated(address indexed farmer, uint256 oldScore, uint256 newScore, string reason)',
  'event MetadataUpdated(uint256 indexed tokenId, string newURI)',

  // Read Functions
  'function name() view returns (string)',
  'function symbol() view returns (string)',
  'function totalSupply() view returns (uint256)',
  'function ownerOf(uint256 tokenId) view returns (address)',
  'function tokenOfOwner(address owner) view returns (uint256)',
  'function balanceOf(address owner) view returns (uint256)',
  'function tokenURI(uint256 tokenId) view returns (string)',
  'function getFarmerProfile(address farmer) view returns (tuple(uint256 tokenId, address walletAddress, bytes32 phoneHash, uint256 reputationScore, bool isKYCVerified, uint256 registeredAt, uint256 completedProjects, uint256 failedProjects, string metadataURI))',
  'function getReputation(address farmer) view returns (uint256)',
  'function isRegistered(address farmer) view returns (bool)',
  'function isVerified(address farmer) view returns (bool)',
  'function phoneHashToWallet(bytes32 phoneHash) view returns (address)',

  // Write Functions
  'function registerFarmer(address farmer, bytes32 phoneHash, string calldata metadataURI) returns (uint256)',
  'function verifyFarmer(address farmer)',
  'function revokeVerification(address farmer)',
  'function recordProjectCompletion(address farmer, bool success)',
  'function adjustReputation(address farmer, uint256 newScore, string calldata reason)',
  'function updateMetadata(address farmer, string calldata newURI)',

  // Admin Functions
  'function setTaniVault(address taniVault)',
  'function addKYCAdmin(address admin)',
  'function removeKYCAdmin(address admin)',
] as const;

// Contract addresses on Lisk Sepolia (Deployed January 2026)
export const CONTRACT_ADDRESSES = {
  LISK_SEPOLIA: {
    TANI_VAULT: process.env.TANI_VAULT_ADDRESS || '0xB39c94B718A75c3005F06f977224cF52AD7cAe49',
    IDRX: process.env.IDRX_ADDRESS || '0x01653fA9F9e9411ac3028f6b4A54f39D68edEA44',
    FARMER_REGISTRY: process.env.FARMER_REGISTRY_ADDRESS || '0x01A0789ae050370AC87d38Fd42b5371Ea0128bA4',
  },
} as const;

// Network configuration
export const NETWORK_CONFIG = {
  LISK_SEPOLIA: {
    chainId: 4202,
    name: 'Lisk Sepolia',
    rpcUrl: process.env.LISK_RPC_URL || 'https://rpc.sepolia-api.lisk.com',
    blockExplorer: 'https://sepolia-blockscout.lisk.com',
  },
} as const;
