'use client';

import { useState, useEffect } from 'react';
import { ethers, Contract } from 'ethers';
import { useWalletContext } from '@/components/WalletProvider';
import { IDRX_ABI, CONTRACT_ADDRESSES, LISK_SEPOLIA } from '@/lib/contracts';

export default function FaucetPage() {
  const { address, isConnected, isCorrectNetwork, signer } = useWalletContext();

  const [balance, setBalance] = useState<string>('0');
  const [loading, setLoading] = useState(false);
  const [claiming, setClaiming] = useState(false);
  const [txHash, setTxHash] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isConnected && address && isCorrectNetwork) {
      fetchBalance();
    }
  }, [isConnected, address, isCorrectNetwork]);

  const fetchBalance = async () => {
    if (!address) return;

    try {
      setLoading(true);
      const provider = new ethers.JsonRpcProvider(LISK_SEPOLIA.rpcUrls[0]);
      const idrx = new Contract(CONTRACT_ADDRESSES.IDRX, IDRX_ABI, provider);

      const bal = await idrx.balanceOf(address);
      setBalance(ethers.formatUnits(bal, 2));
    } catch (err) {
      console.error('Failed to fetch balance:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleClaimFaucet = async () => {
    if (!signer) return;

    try {
      setClaiming(true);
      setError(null);
      setTxHash(null);

      const idrx = new Contract(CONTRACT_ADDRESSES.IDRX, IDRX_ABI, signer);
      const tx = await idrx.faucet();
      await tx.wait();

      setTxHash(tx.hash);
      await fetchBalance();
    } catch (err: any) {
      console.error('Faucet claim failed:', err);
      setError(err.message || 'Failed to claim from faucet');
    } finally {
      setClaiming(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div className="text-center mb-8">
        <span className="text-6xl mb-4 block">💧</span>
        <h1 className="text-3xl font-bold text-gray-800 mb-2">IDRX Test Faucet</h1>
        <p className="text-gray-600">
          Get free test IDRX tokens to try out TaniFi on Lisk Sepolia.
        </p>
      </div>

      <div className="bg-white rounded-xl shadow-md p-8">
        {!isConnected ? (
          <div className="text-center">
            <p className="text-gray-600 mb-4">
              Connect your wallet to claim test tokens.
            </p>
          </div>
        ) : !isCorrectNetwork ? (
          <div className="text-center">
            <p className="text-amber-600 mb-4">
              Please switch to Lisk Sepolia network.
            </p>
          </div>
        ) : (
          <>
            {/* Current Balance */}
            <div className="mb-6 p-4 bg-primary-50 rounded-lg text-center">
              <p className="text-sm text-primary-600 mb-1">Your IDRX Balance</p>
              <p className="text-3xl font-bold text-primary-700">
                {loading ? '...' : `Rp ${Number(balance).toLocaleString('id-ID')}`}
              </p>
            </div>

            {/* Faucet Info */}
            <div className="mb-6 p-4 bg-gray-50 rounded-lg">
              <h3 className="font-semibold text-gray-800 mb-2">Faucet Details</h3>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>Each claim gives you 10,000,000 IDRX</li>
                <li>For testing purposes only on Lisk Sepolia</li>
                <li>No cooldown - claim as many times as needed</li>
              </ul>
            </div>

            {/* Error Display */}
            {error && (
              <div className="mb-4 p-3 bg-red-50 text-red-600 rounded-lg text-sm">
                {error}
              </div>
            )}

            {/* Success Display */}
            {txHash && (
              <div className="mb-4 p-3 bg-green-50 text-green-600 rounded-lg">
                <p className="font-medium">Tokens claimed successfully!</p>
                <a
                  href={`${LISK_SEPOLIA.blockExplorerUrls[0]}/tx/${txHash}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary-600 underline text-sm"
                >
                  View transaction on Explorer
                </a>
              </div>
            )}

            {/* Claim Button */}
            <button
              onClick={handleClaimFaucet}
              disabled={claiming}
              className="w-full py-3 bg-primary-600 text-white text-lg font-semibold rounded-lg hover:bg-primary-700 disabled:opacity-50 transition-colors"
            >
              {claiming ? (
                <span className="flex items-center justify-center">
                  <svg
                    className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  Claiming...
                </span>
              ) : (
                'Claim 10M Test IDRX'
              )}
            </button>
          </>
        )}
      </div>

      {/* Additional Info */}
      <div className="mt-8 text-center text-sm text-gray-500">
        <p>
          Need ETH for gas? Use the{' '}
          <a
            href="https://sepolia-faucet.lisk.com/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary-600 underline"
          >
            Lisk Sepolia Faucet
          </a>
        </p>
      </div>
    </div>
  );
}
