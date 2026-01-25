'use client';

import React, { ReactNode } from 'react';
import { Web3ReactProvider } from '@web3-react/core';
import { ethers } from 'ethers';
import { WalletProvider } from './WalletProvider';

// Function to get ethers library from provider (for web3-react v6)
function getLibrary(provider: any) {
  const library = new ethers.BrowserProvider(provider);
  return library;
}

export function Providers({ children }: { children: ReactNode }) {
  return (
    <Web3ReactProvider getLibrary={getLibrary}>
      <WalletProvider>
        {children}
      </WalletProvider>
    </Web3ReactProvider>
  );
}
