'use client';

import { useState, useEffect } from 'react';
import { ethers, Contract } from 'ethers';
import Link from 'next/link';
import { useWalletContext } from '@/components/WalletProvider';
import { ProjectState } from '@/types';
import { TANI_VAULT_ABI, IDRX_ABI, CONTRACT_ADDRESSES, BASE_SEPOLIA } from '@/lib/contracts';

interface InvestmentItem {
  projectId: number;
  amount: string;
  state: ProjectState;
  stateName: string;
  canWithdraw: boolean;
}

export default function DashboardPage() {
  const { address, isConnected, isCorrectNetwork, signer } = useWalletContext();

  const [investments, setInvestments] = useState<InvestmentItem[]>([]);
  const [idrxBalance, setIdrxBalance] = useState<string>('0');
  const [loading, setLoading] = useState(true);
  const [withdrawing, setWithdrawing] = useState<number | null>(null);

  useEffect(() => {
    if (isConnected && address && isCorrectNetwork) {
      fetchUserData();
    } else {
      setLoading(false);
    }
  }, [isConnected, address, isCorrectNetwork]);

  const fetchUserData = async () => {
    if (!address) return;

    try {
      setLoading(true);
      const provider = new ethers.JsonRpcProvider(BASE_SEPOLIA.rpcUrls[0]);
      const vault = new Contract(CONTRACT_ADDRESSES.TANI_VAULT, TANI_VAULT_ABI, provider);
      const idrx = new Contract(CONTRACT_ADDRESSES.IDRX, IDRX_ABI, provider);

      // Get IDRX balance
      const balance = await idrx.balanceOf(address);
      setIdrxBalance(ethers.formatUnits(balance, 2));

      // Get project count and check investments
      const count = await vault.projectCount();
      const projectCount = Number(count);

      const userInvestments: InvestmentItem[] = [];

      for (let i = 0; i < projectCount; i++) {
        const investment = await vault.getInvestment(i, address);
        if (investment > 0n) {
          const project = await vault.getProject(i);
          const state = Number(project.state) as ProjectState;

          userInvestments.push({
            projectId: i,
            amount: ethers.formatUnits(investment, 2),
            state,
            stateName: ProjectState[state],
            canWithdraw: state === ProjectState.COMPLETED,
          });
        }
      }

      setInvestments(userInvestments);
    } catch (err) {
      console.error('Failed to fetch user data:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleWithdraw = async (projectId: number) => {
    if (!signer) return;

    try {
      setWithdrawing(projectId);
      const vault = new Contract(CONTRACT_ADDRESSES.TANI_VAULT, TANI_VAULT_ABI, signer);

      const tx = await vault.withdrawReturns(projectId);
      await tx.wait();

      // Refresh data
      await fetchUserData();
    } catch (err) {
      console.error('Withdrawal failed:', err);
    } finally {
      setWithdrawing(null);
    }
  };

  if (!isConnected) {
    return (
      <div className="max-w-4xl mx-auto text-center py-20">
        <span className="text-6xl mb-6 block drop-shadow-lg">🔒</span>
        <h1 className="text-2xl font-bold gradient-text mb-4">
          Connect Your Wallet
        </h1>
        <p className="text-gray-700 font-medium">
          Please connect your wallet to view your investments.
        </p>
      </div>
    );
  }

  if (!isCorrectNetwork) {
    return (
      <div className="max-w-4xl mx-auto text-center py-20">
        <span className="text-6xl mb-6 block drop-shadow-lg">🔄</span>
        <h1 className="text-2xl font-bold gradient-text mb-4">
          Wrong Network
        </h1>
        <p className="text-gray-700 font-medium">
          Please switch to Base Sepolia network to view your investments.
        </p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold gradient-text mb-8">My Dashboard</h1>

      {/* Balance Card */}
      <div className="glass-card rounded-2xl shadow-xl p-6 mb-8">
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <p className="text-sm text-gray-700 mb-2 font-medium">Wallet Address</p>
            <p className="font-mono text-sm glass p-3 rounded-lg border-primary-200">
              {address}
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-700 mb-2 font-medium">IDRX Balance</p>
            <p className="text-2xl font-bold gradient-text">
              Rp {Number(idrxBalance).toLocaleString('id-ID')}
            </p>
          </div>
        </div>
      </div>

      {/* Investments */}
      <h2 className="text-xl font-bold gradient-text mb-4">My Investments</h2>

      {loading ? (
        <div className="flex justify-center py-12">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-primary-600"></div>
        </div>
      ) : investments.length === 0 ? (
        <div className="glass-card rounded-2xl shadow-xl p-8 text-center">
          <span className="text-5xl mb-4 block drop-shadow-lg">📊</span>
          <p className="text-gray-700 mb-4 font-medium">You haven't made any investments yet.</p>
          <Link
            href="/"
            className="inline-block px-6 py-2.5 bg-gradient-to-r from-primary-600 to-primary-700 text-white font-semibold rounded-lg hover:from-primary-700 hover:to-primary-800 transition-all duration-300 shadow-lg hover:shadow-xl"
          >
            Browse Projects
          </Link>
        </div>
      ) : (
        <div className="space-y-4">
          {investments.map((inv) => (
            <div
              key={inv.projectId}
              className="glass-card rounded-2xl shadow-lg p-6 flex justify-between items-center hover:shadow-2xl transition-all"
            >
              <div>
                <Link
                  href={`/project/${inv.projectId}`}
                  className="text-lg font-semibold gradient-text hover:opacity-80 transition-opacity"
                >
                  Project #{inv.projectId}
                </Link>
                <p className="text-sm text-gray-600 font-medium">Status: {inv.stateName}</p>
              </div>
              <div className="text-right">
                <p className="text-lg font-bold gradient-text">
                  Rp {Number(inv.amount).toLocaleString('id-ID')}
                </p>
                {inv.canWithdraw && (
                  <button
                    onClick={() => handleWithdraw(inv.projectId)}
                    disabled={withdrawing === inv.projectId}
                    className="mt-2 px-4 py-1.5 bg-gradient-to-r from-primary-500 to-primary-600 text-white text-sm font-semibold rounded-lg hover:from-primary-600 hover:to-primary-700 disabled:opacity-50 shadow-md hover:shadow-lg transition-all"
                  >
                    {withdrawing === inv.projectId ? 'Withdrawing...' : 'Withdraw Returns'}
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
