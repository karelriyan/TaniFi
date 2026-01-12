'use client';

import Link from 'next/link';
import { useWalletContext } from './WalletProvider';

export function Navbar() {
  const {
    address,
    isConnected,
    isCorrectNetwork,
    isConnecting,
    connect,
    disconnect,
    switchNetwork,
  } = useWalletContext();

  const formatAddress = (addr: string) => {
    return `${addr.slice(0, 6)}...${addr.slice(-4)}`;
  };

  return (
    <nav className="bg-white shadow-md">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-2">
            <span className="text-2xl">🌾</span>
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
              href="/faucet"
              className="text-gray-600 hover:text-primary-600 transition-colors"
            >
              Faucet
            </Link>
          </div>

          {/* Wallet Connection */}
          <div className="flex items-center space-x-4">
            {isConnected ? (
              <>
                {!isCorrectNetwork && (
                  <button
                    onClick={switchNetwork}
                    className="px-4 py-2 text-sm bg-amber-500 text-white rounded-lg hover:bg-amber-600"
                  >
                    Switch Network
                  </button>
                )}
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-600 bg-gray-100 px-3 py-1 rounded-full">
                    {formatAddress(address!)}
                  </span>
                  <button
                    onClick={disconnect}
                    className="px-4 py-2 text-sm text-red-600 hover:text-red-700"
                  >
                    Disconnect
                  </button>
                </div>
              </>
            ) : (
              <button
                onClick={connect}
                disabled={isConnecting}
                className="px-6 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 transition-colors"
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
