'use client';

import Link from 'next/link';
import Image from 'next/image';
import { useWalletContext } from './WalletProvider';

export function Navbar() {
  const {
    address,
    isConnected,
    isCorrectNetwork,
    isConnecting,
    error,
    connect,
    disconnect,
    switchNetwork,
  } = useWalletContext();

  const formatAddress = (addr: string) => {
    return `${addr.slice(0, 6)}...${addr.slice(-4)}`;
  };

  return (
    <nav className="glass-navbar shadow-lg sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-3">
            <Image
              src="/tanifi-logo.png"
              alt="TaniFi Logo"
              width={40}
              height={40}
              className="object-contain"
            />
            <span className="text-xl font-bold text-primary-700">TaniFi</span>
          </Link>

          {/* Navigation Links */}
          <div className="hidden md:flex items-center space-x-8">
            <Link
              href="/"
              className="text-gray-600 hover:text-primary-600 transition-colors"
            >
              Projects
            </Link>
            <Link
              href="/dashboard"
              className="text-gray-600 hover:text-primary-600 transition-colors"
            >
              My Investments
            </Link>
            <Link
              href="/onboarding"
              className="text-gray-600 hover:text-primary-600 transition-colors"
            >
              Onboarding
            </Link>
            <Link
              href="/faucet"
              className="text-gray-600 hover:text-primary-600 transition-colors"
            >
              Faucet
            </Link>
          </div>

          {/* Wallet Connection */}
          <div className="flex items-center space-x-4">
            {error && (
              <div className="text-sm text-red-600 bg-red-50 px-3 py-2 rounded-lg max-w-xs">
                {error}
              </div>
            )}
            {isConnected ? (
              <>
                {!isCorrectNetwork && (
                  <button
                    onClick={switchNetwork}
                    className="px-4 py-2 text-sm glass-button text-amber-700 font-medium rounded-lg"
                  >
                    Switch Network
                  </button>
                )}
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-700 glass px-3 py-1 rounded-full font-medium">
                    {formatAddress(address!)}
                  </span>
                  <button
                    onClick={disconnect}
                    className="px-4 py-2 text-sm text-red-600 hover:text-red-700 font-medium"
                  >
                    Disconnect
                  </button>
                </div>
              </>
            ) : (
              <button
                onClick={connect}
                disabled={isConnecting}
                className="px-6 py-2 glass-strong text-primary-700 font-semibold rounded-lg disabled:opacity-50 transition-all"
              >
                {isConnecting ? 'Connecting...' : 'Connect Wallet'}
              </button>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
