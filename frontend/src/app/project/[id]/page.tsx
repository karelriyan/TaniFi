'use client';

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { ethers, Contract } from 'ethers';
import { useWalletContext } from '@/components/WalletProvider';
import { ProjectDisplay, ProjectState } from '@/types';
import { TANI_VAULT_ABI, IDRX_ABI, CONTRACT_ADDRESSES, BASE_SEPOLIA } from '@/lib/contracts';

const STATE_COLORS: Record<ProjectState, string> = {
  [ProjectState.FUNDRAISING]: 'glass text-blue-700 border-blue-300',
  [ProjectState.ACTIVE]: 'glass text-primary-700 border-primary-400',
  [ProjectState.HARVESTED]: 'glass text-amber-700 border-amber-400',
  [ProjectState.FAILED]: 'glass text-red-700 border-red-400',
  [ProjectState.COMPLETED]: 'glass text-gray-700 border-gray-400',
};

const STATE_NAMES: Record<ProjectState, string> = {
  [ProjectState.FUNDRAISING]: 'Fundraising',
  [ProjectState.ACTIVE]: 'Active',
  [ProjectState.HARVESTED]: 'Harvested',
  [ProjectState.FAILED]: 'Failed',
  [ProjectState.COMPLETED]: 'Completed',
};

export default function ProjectDetailPage() {
  const params = useParams();
  const router = useRouter();
  const projectId = Number(params.id);

  const { address, isConnected, isCorrectNetwork, signer, provider } = useWalletContext();

  const [project, setProject] = useState<ProjectDisplay | null>(null);
  const [myInvestment, setMyInvestment] = useState<string>('0');
  const [idrxBalance, setIdrxBalance] = useState<string>('0');
  const [investAmount, setInvestAmount] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [investing, setInvesting] = useState(false);
  const [approving, setApproving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [txHash, setTxHash] = useState<string | null>(null);

  useEffect(() => {
    if (!isNaN(projectId)) {
      fetchProjectDetails();
    }
  }, [projectId]);

  useEffect(() => {
    if (isConnected && address && isCorrectNetwork) {
      fetchUserData();
    }
  }, [isConnected, address, isCorrectNetwork]);

  const fetchProjectDetails = async () => {
    try {
      setLoading(true);
      const rpcProvider = new ethers.JsonRpcProvider(BASE_SEPOLIA.rpcUrls[0]);
      const vault = new Contract(CONTRACT_ADDRESSES.TANI_VAULT, TANI_VAULT_ABI, rpcProvider);

      const projectData = await vault.getProject(projectId);

      const targetAmount = Number(ethers.formatUnits(projectData.targetAmount, 2));
      const fundedAmount = Number(ethers.formatUnits(projectData.fundedAmount, 2));

      setProject({
        id: Number(projectData.id),
        farmer: projectData.farmer,
        cooperative: projectData.cooperative,
        targetAmount: targetAmount.toLocaleString('id-ID'),
        fundedAmount: fundedAmount.toLocaleString('id-ID'),
        fundingProgress: targetAmount > 0 ? (fundedAmount / targetAmount) * 100 : 0,
        farmerShare: Number(projectData.farmerShareBps) / 100,
        investorShare: Number(projectData.investorShareBps) / 100,
        harvestDate: new Date(Number(projectData.harvestTime) * 1000).toLocaleDateString('id-ID'),
        state: Number(projectData.state) as ProjectState,
        stateName: STATE_NAMES[Number(projectData.state) as ProjectState],
      });
    } catch (err) {
      console.error('Failed to fetch project:', err);
      setError('Failed to load project details');
    } finally {
      setLoading(false);
    }
  };

  const fetchUserData = async () => {
    if (!address || !provider) return;

    try {
      const rpcProvider = new ethers.JsonRpcProvider(BASE_SEPOLIA.rpcUrls[0]);
      const vault = new Contract(CONTRACT_ADDRESSES.TANI_VAULT, TANI_VAULT_ABI, rpcProvider);
      const idrx = new Contract(CONTRACT_ADDRESSES.IDRX, IDRX_ABI, rpcProvider);

      // Get user's investment in this project
      const investment = await vault.getInvestment(projectId, address);
      setMyInvestment(ethers.formatUnits(investment, 2));

      // Get user's IDRX balance
      const balance = await idrx.balanceOf(address);
      setIdrxBalance(ethers.formatUnits(balance, 2));
    } catch (err) {
      console.error('Failed to fetch user data:', err);
    }
  };

  const handleApprove = async () => {
    if (!signer || !investAmount) return;

    try {
      setApproving(true);
      setError(null);

      const idrx = new Contract(CONTRACT_ADDRESSES.IDRX, IDRX_ABI, signer);
      const amount = ethers.parseUnits(investAmount, 2);

      const tx = await idrx.approve(CONTRACT_ADDRESSES.TANI_VAULT, amount);
      await tx.wait();

      setTxHash(tx.hash);
    } catch (err: any) {
      console.error('Approval failed:', err);
      setError(err.message || 'Failed to approve IDRX');
    } finally {
      setApproving(false);
    }
  };

  const handleInvest = async () => {
    if (!signer || !investAmount) return;

    try {
      setInvesting(true);
      setError(null);

      const vault = new Contract(CONTRACT_ADDRESSES.TANI_VAULT, TANI_VAULT_ABI, signer);
      const amount = ethers.parseUnits(investAmount, 2);

      const tx = await vault.invest(projectId, amount);
      await tx.wait();

      setTxHash(tx.hash);
      setInvestAmount('');

      // Refresh data
      await fetchProjectDetails();
      await fetchUserData();
    } catch (err: any) {
      console.error('Investment failed:', err);
      setError(err.message || 'Failed to invest');
    } finally {
      setInvesting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center py-20">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="text-center py-20">
        <p className="text-red-700 mb-4 font-semibold">Project not found</p>
        <button
          onClick={() => router.push('/')}
          className="px-6 py-2.5 bg-gradient-to-r from-primary-600 to-primary-700 text-white font-semibold rounded-lg hover:from-primary-700 hover:to-primary-800 shadow-lg hover:shadow-xl transition-all"
        >
          Back to Projects
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto">
      {/* Back Button */}
      <button
        onClick={() => router.push('/')}
        className="mb-6 glass-strong px-4 py-2 rounded-lg text-primary-700 hover:text-primary-800 font-semibold flex items-center gap-2 transition-all hover:shadow-lg"
      >
        <span>&larr;</span>
        <span>Back to Projects</span>
      </button>

      <div className="grid md:grid-cols-3 gap-8">
        {/* Main Content */}
        <div className="md:col-span-2">
          {/* Header Card */}
          <div className="glass-card rounded-2xl shadow-xl overflow-hidden mb-6">
            <div className="h-48 bg-gradient-to-br from-primary-400 via-primary-500 to-primary-600 flex items-center justify-center relative overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-t from-primary-700/20 to-transparent"></div>
              <span className="text-8xl drop-shadow-lg relative z-10">🌾</span>
            </div>
            <div className="p-6">
              <div className="flex justify-between items-start mb-4">
                <h1 className="text-2xl font-bold gradient-text">
                  Project #{project.id}
                </h1>
                <span className={`px-3 py-1 text-sm font-semibold rounded-full ${STATE_COLORS[project.state]}`}>
                  {project.stateName}
                </span>
              </div>

              {/* Funding Progress */}
              <div className="mb-6">
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-700 font-medium">Funding Progress</span>
                  <span className="font-semibold gradient-text">{project.fundingProgress.toFixed(1)}%</span>
                </div>
                <div className="w-full glass-dark rounded-full h-3 overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-primary-500 via-primary-600 to-primary-700 h-3 rounded-full transition-all duration-500 shadow-lg"
                    style={{ width: `${Math.min(project.fundingProgress, 100)}%` }}
                  />
                </div>
                <div className="flex justify-between text-sm text-gray-600 mt-2 font-medium">
                  <span>Rp {project.fundedAmount}</span>
                  <span>Target: Rp {project.targetAmount}</span>
                </div>
              </div>

              {/* Details Grid */}
              <div className="grid grid-cols-2 gap-4">
                <div className="glass rounded-lg p-4 border-primary-200">
                  <p className="text-sm text-gray-600 font-medium">Investor Share</p>
                  <p className="text-xl font-bold gradient-text">{project.investorShare}%</p>
                </div>
                <div className="glass rounded-lg p-4 border-primary-200">
                  <p className="text-sm text-gray-600 font-medium">Farmer Share</p>
                  <p className="text-xl font-bold text-tanifi-gold">{project.farmerShare}%</p>
                </div>
                <div className="glass rounded-lg p-4 border-primary-200">
                  <p className="text-sm text-gray-600 font-medium">Expected Harvest</p>
                  <p className="text-lg font-semibold text-gray-800">{project.harvestDate}</p>
                </div>
                <div className="glass rounded-lg p-4 border-primary-200">
                  <p className="text-sm text-gray-600 font-medium">Cooperative</p>
                  <p className="text-sm font-mono truncate text-gray-800">{project.cooperative}</p>
                </div>
              </div>
            </div>
          </div>

          {/* About Section */}
          <div className="glass-card rounded-2xl shadow-xl p-6">
            <h2 className="text-xl font-bold gradient-text mb-4">About This Project</h2>
            <p className="text-gray-700 mb-4 font-medium">
              This agricultural project follows the Musyarakah (profit-sharing) model compliant
              with Islamic finance principles. Investors provide capital to farmers, and profits
              from the harvest are shared according to the agreed ratio.
            </p>
            <ul className="space-y-2 text-gray-700">
              <li>• Platform fee: 1%</li>
              <li>• Investor returns are proportional to investment amount</li>
              <li>• Funds disbursed directly to approved vendors</li>
            </ul>
          </div>
        </div>

        {/* Investment Sidebar */}
        <div className="md:col-span-1">
          <div className="glass-card rounded-2xl shadow-xl p-6 sticky top-4">
            <h3 className="text-lg font-bold gradient-text mb-4">Invest</h3>

            {!isConnected ? (
              <p className="text-gray-700 text-sm font-medium">
                Connect your wallet to invest in this project.
              </p>
            ) : !isCorrectNetwork ? (
              <p className="text-amber-700 text-sm font-medium">
                Please switch to Base Sepolia network.
              </p>
            ) : project.state !== ProjectState.FUNDRAISING ? (
              <p className="text-gray-700 text-sm font-medium">
                This project is no longer accepting investments.
              </p>
            ) : (
              <>
                {/* User Balance */}
                <div className="mb-4 p-3 glass rounded-lg border-primary-200">
                  <p className="text-sm text-gray-600 font-medium">Your IDRX Balance</p>
                  <p className="text-lg font-semibold text-gray-800">Rp {Number(idrxBalance).toLocaleString('id-ID')}</p>
                </div>

                {/* Current Investment */}
                {Number(myInvestment) > 0 && (
                  <div className="mb-4 p-3 glass-strong rounded-lg border-primary-300">
                    <p className="text-sm text-primary-700 font-medium">Your Investment</p>
                    <p className="text-lg font-semibold gradient-text">
                      Rp {Number(myInvestment).toLocaleString('id-ID')}
                    </p>
                  </div>
                )}

                {/* Investment Input */}
                <div className="mb-4">
                  <label className="block text-sm text-gray-700 mb-2 font-medium">
                    Investment Amount (IDRX)
                  </label>
                  <input
                    type="number"
                    value={investAmount}
                    onChange={(e) => setInvestAmount(e.target.value)}
                    placeholder="0"
                    className="w-full px-4 py-3 glass border-primary-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-400 transition-all"
                  />
                </div>

                {/* Error Display */}
                {error && (
                  <div className="mb-4 p-3 glass border-red-300 rounded-lg">
                    <p className="text-red-700 text-sm font-medium">{error}</p>
                  </div>
                )}

                {/* Success Display */}
                {txHash && (
                  <div className="mb-4 p-3 glass border-primary-300 rounded-lg">
                    <p className="text-primary-800 font-semibold text-sm mb-1">Transaction submitted!</p>
                    <a
                      href={`${BASE_SEPOLIA.blockExplorerUrls[0]}/tx/${txHash}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block text-primary-600 underline text-sm font-medium"
                    >
                      View on Explorer
                    </a>
                  </div>
                )}

                {/* Action Buttons */}
                <div className="space-y-3">
                  <button
                    onClick={handleApprove}
                    disabled={!investAmount || approving}
                    className="w-full py-2.5 glass-strong text-gray-800 font-semibold rounded-lg hover:shadow-lg disabled:opacity-50 transition-all border-primary-200"
                  >
                    {approving ? 'Approving...' : '1. Approve IDRX'}
                  </button>
                  <button
                    onClick={handleInvest}
                    disabled={!investAmount || investing}
                    className="w-full py-2.5 bg-gradient-to-r from-primary-600 to-primary-700 text-white font-semibold rounded-lg hover:from-primary-700 hover:to-primary-800 disabled:opacity-50 shadow-lg hover:shadow-xl transition-all"
                  >
                    {investing ? 'Investing...' : '2. Invest'}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
