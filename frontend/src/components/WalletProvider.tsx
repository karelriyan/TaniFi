'use client';

import React, { createContext, useContext, ReactNode, useState, useCallback } from 'react';
import { useWeb3React } from '@web3-react/core';
import { injectedConnector } from '@/wallet/connectors';
import { BASE_SEPOLIA } from '@/lib/contracts';
import { BrowserProvider, JsonRpcSigner } from 'ethers';

interface WalletContextType {
  address: string | null;
  chainId: number | null;
  isConnected: boolean;
  isCorrectNetwork: boolean;
  isConnecting: boolean;
  error: string | null;
  signer: JsonRpcSigner | null;
  provider: BrowserProvider | null;
  connect: () => Promise<void>;
  disconnect: () => void;
  switchNetwork: () => Promise<void>;
}

const WalletContext = createContext<WalletContextType | undefined>(undefined);

export function WalletProvider({ children }: { children: ReactNode }) {
  const { active, account, chainId, library, activate, deactivate } = useWeb3React();
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [signer, setSigner] = useState<JsonRpcSigner | null>(null);

  // Connect to MetaMask
  const connect = useCallback(async () => {
    if (typeof window === 'undefined' || !window.ethereum) {
      setError('Please install MetaMask browser extension');
      return;
    }

    if (!window.ethereum.isMetaMask) {
      setError('MetaMask not detected. Please install MetaMask.');
      return;
    }

    setIsConnecting(true);
    setError(null);

    try {
      await activate(injectedConnector);

      // Get signer after activation
      if (library) {
        const signer = await library.getSigner();
        setSigner(signer);
      }

      console.log('✅ Wallet connected via web3-react');
    } catch (err: any) {
      console.error('Connection error:', err);

      if (err.code === 4001) {
        setError('Connection rejected. Please approve in MetaMask.');
      } else if (err.code === -32002) {
        setError('Connection request pending. Check MetaMask.');
      } else {
        setError(err.message || 'Failed to connect wallet');
      }
    } finally {
      setIsConnecting(false);
    }
  }, [activate, library]);

  // Disconnect wallet
  const disconnect = useCallback(() => {
    try {
      deactivate();
      setSigner(null);
      setError(null);
      console.log('✅ Wallet disconnected');
    } catch (err: any) {
      console.error('Disconnect error:', err);
    }
  }, [deactivate]);

  // Switch to Base Sepolia network
  const switchNetwork = useCallback(async () => {
    if (!window.ethereum) return;

    try {
      await window.ethereum.request({
        method: 'wallet_switchEthereumChain',
        params: [{ chainId: BASE_SEPOLIA.chainIdHex }],
      });
    } catch (switchError: any) {
      // Chain not added, try to add it
      if (switchError.code === 4902) {
        try {
          await window.ethereum.request({
            method: 'wallet_addEthereumChain',
            params: [
              {
                chainId: BASE_SEPOLIA.chainIdHex,
                chainName: BASE_SEPOLIA.chainName,
                nativeCurrency: BASE_SEPOLIA.nativeCurrency,
                rpcUrls: BASE_SEPOLIA.rpcUrls,
                blockExplorerUrls: BASE_SEPOLIA.blockExplorerUrls,
              },
            ],
          });
        } catch (addError) {
          setError('Failed to add Base Sepolia network');
        }
      }
    }
  }, []);

  // Update signer when library changes
  React.useEffect(() => {
    if (active && library) {
      library.getSigner().then(setSigner).catch(console.error);
    } else {
      setSigner(null);
    }
  }, [active, library]);

  const value: WalletContextType = {
    address: account || null,
    chainId: chainId || null,
    isConnected: active,
    isCorrectNetwork: chainId === BASE_SEPOLIA.chainId,
    isConnecting,
    error,
    signer,
    provider: library || null,
    connect,
    disconnect,
    switchNetwork,
  };

  return (
    <WalletContext.Provider value={value}>
      {children}
    </WalletContext.Provider>
  );
}

export function useWalletContext() {
  const context = useContext(WalletContext);
  if (context === undefined) {
    throw new Error('useWalletContext must be used within a WalletProvider');
  }
  return context;
}

// Extend Window interface for ethereum
declare global {
  interface Window {
    ethereum?: any;
  }
}
