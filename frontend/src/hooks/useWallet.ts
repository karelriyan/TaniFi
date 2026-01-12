'use client';

import { useState, useEffect, useCallback } from 'react';
import { ethers, BrowserProvider, JsonRpcSigner } from 'ethers';
import { LISK_SEPOLIA } from '@/lib/contracts';

interface WalletState {
  address: string | null;
  chainId: number | null;
  isConnected: boolean;
  isCorrectNetwork: boolean;
  signer: JsonRpcSigner | null;
  provider: BrowserProvider | null;
}

export function useWallet() {
  const [wallet, setWallet] = useState<WalletState>({
    address: null,
    chainId: null,
    isConnected: false,
    isCorrectNetwork: false,
    signer: null,
    provider: null,
  });
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const checkConnection = useCallback(async () => {
    if (typeof window === 'undefined' || !window.ethereum) return;

    try {
      const provider = new BrowserProvider(window.ethereum);
      const accounts = await provider.listAccounts();

      if (accounts.length > 0) {
        const signer = await provider.getSigner();
        const address = await signer.getAddress();
        const network = await provider.getNetwork();
        const chainId = Number(network.chainId);

        setWallet({
          address,
          chainId,
          isConnected: true,
          isCorrectNetwork: chainId === LISK_SEPOLIA.chainId,
          signer,
          provider,
        });
      }
    } catch (err) {
      console.error('Failed to check wallet connection:', err);
    }
  }, []);

  useEffect(() => {
    checkConnection();

    if (typeof window !== 'undefined' && window.ethereum) {
      const handleAccountsChanged = (accounts: string[]) => {
        if (accounts.length === 0) {
          setWallet({
            address: null,
            chainId: null,
            isConnected: false,
            isCorrectNetwork: false,
            signer: null,
            provider: null,
          });
        } else {
          checkConnection();
        }
      };

      const handleChainChanged = () => {
        checkConnection();
      };

      window.ethereum.on('accountsChanged', handleAccountsChanged);
      window.ethereum.on('chainChanged', handleChainChanged);

      return () => {
        window.ethereum.removeListener('accountsChanged', handleAccountsChanged);
        window.ethereum.removeListener('chainChanged', handleChainChanged);
      };
    }
  }, [checkConnection]);

  const connect = useCallback(async () => {
    if (typeof window === 'undefined' || !window.ethereum) {
      setError('Please install MetaMask or another Web3 wallet');
      return;
    }

    setIsConnecting(true);
    setError(null);

    try {
      const provider = new BrowserProvider(window.ethereum);
      await provider.send('eth_requestAccounts', []);

      const signer = await provider.getSigner();
      const address = await signer.getAddress();
      const network = await provider.getNetwork();
      const chainId = Number(network.chainId);

      setWallet({
        address,
        chainId,
        isConnected: true,
        isCorrectNetwork: chainId === LISK_SEPOLIA.chainId,
        signer,
        provider,
      });
    } catch (err: any) {
      setError(err.message || 'Failed to connect wallet');
    } finally {
      setIsConnecting(false);
    }
  }, []);

  const disconnect = useCallback(() => {
    setWallet({
      address: null,
      chainId: null,
      isConnected: false,
      isCorrectNetwork: false,
      signer: null,
      provider: null,
    });
  }, []);

  const switchNetwork = useCallback(async () => {
    if (!window.ethereum) return;

    try {
      await window.ethereum.request({
        method: 'wallet_switchEthereumChain',
        params: [{ chainId: LISK_SEPOLIA.chainIdHex }],
      });
    } catch (switchError: any) {
      // Chain not added, try to add it
      if (switchError.code === 4902) {
        try {
          await window.ethereum.request({
            method: 'wallet_addEthereumChain',
            params: [
              {
                chainId: LISK_SEPOLIA.chainIdHex,
                chainName: LISK_SEPOLIA.chainName,
                nativeCurrency: LISK_SEPOLIA.nativeCurrency,
                rpcUrls: LISK_SEPOLIA.rpcUrls,
                blockExplorerUrls: LISK_SEPOLIA.blockExplorerUrls,
              },
            ],
          });
        } catch (addError) {
          setError('Failed to add Lisk Sepolia network');
        }
      }
    }
  }, []);

  return {
    ...wallet,
    isConnecting,
    error,
    connect,
    disconnect,
    switchNetwork,
  };
}

// Extend Window interface for ethereum
declare global {
  interface Window {
    ethereum?: any;
  }
}
