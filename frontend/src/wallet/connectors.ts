// Web3-React Connector Configuration for TaniFi
// Supports MetaMask and other injected wallets

import { InjectedConnector } from '@web3-react/injected-connector';

// Base Sepolia Chain ID
const BASE_SEPOLIA_CHAIN_ID = 84532;

// Supported networks for this dApp
export const SUPPORTED_CHAIN_IDS = [
  BASE_SEPOLIA_CHAIN_ID, // Base Sepolia (primary)
  1, // Ethereum Mainnet (for testing)
  5, // Goerli (for testing)
];

// MetaMask Injected Connector
export const injectedConnector = new InjectedConnector({
  supportedChainIds: SUPPORTED_CHAIN_IDS,
});
