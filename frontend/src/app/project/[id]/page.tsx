'use client';

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { ethers, Contract } from 'ethers';
import { useWalletContext } from '@/components/WalletProvider';
import { ProjectDisplay, ProjectState } from '@/types';
import { TANI_VAULT_ABI, IDRX_ABI, CONTRACT_ADDRESSES, BASE_SEPOLIA } from '@/lib/contracts';

const STATE_COLORS: Record<ProjectState, string> = {
  [ProjectState.FUNDRAISING]: 'bg-blue-100 text-blue-800',
  [ProjectState.ACTIVE]: 'bg-green-100 text-green-800',
  [ProjectState.HARVESTED]: 'bg-amber-100 text-amber-800',
  [ProjectState.FAILED]: 'bg-red-100 text-red-800',
  [ProjectState.COMPLETED]: 'bg-gray-100 text-gray-800',
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
        <p className="text-red-600 mb-4">Project not found</p>
        <button
          onClick={() => router.push('/')}
          className="px-4 py-2 bg-primary-600 text-white rounded-lg"
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
        className="mb-6 text-primary-600 hover:text-primary-700 flex items-center"
      >
        &larr; Back to Projects
      </button>

      <div className="grid md:grid-cols-3 gap-8">
        {/* Main Content */}
        <div className="md:col-span-2">
          {/* Header Card */}
          <div className="bg-white rounded-xl shadow-md overflow-hidden mb-6">
            <div className="h-48 bg-gradient-to-r from-primary-400 to-primary-600 flex items-center justify-center">
              <span className="text-8xl">🌾</span>
            </div>
            <div className="p-6">
              <div className="flex justify-between items-start mb-4">
                <h1 className="text-2xl font-bold text-gray-800">
                  Project #{project.id}
                </h1>
                <span className={`px-3 py-1 text-sm font-medium rounded-full ${STATE_COLORS[project.state]}`}>
                  {project.stateName}
                </span>
              </div>

              {/* Funding Progress */}
              <div className="mb-6">
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-600">Funding Progress</span>
                  <span className="font-medium">{project.fundingProgress.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className="bg-primary-500 h-3 rounded-full transition-all"
                    style={{ width: `${Math.min(project.fundingProgress, 100)}%` }}
                  />
                </div>
                <div className="flex justify-between text-sm text-gray-500 mt-2">
                  <span>Rp {project.fundedAmount}</span>
                  <span>Target: Rp {project.targetAmount}</span>
                </div>
              </div>

              {/* Details Grid */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-sm text-gray-600">Investor Share</p>
                  <p className="text-xl font-bold text-primary-600">{project.investorShare}%</p>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-sm text-gray-600">Farmer Share</p>
                  <p className="text-xl font-bold text-tanifi-gold">{project.farmerShare}%</p>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-sm text-gray-600">Expected Harvest</p>
                  <p className="text-lg font-semibold">{project.harvestDate}</p>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-sm text-gray-600">Cooperative</p>
                  <p className="text-sm font-mono truncate">{project.cooperative}</p>
                </div>
              </div>
            </div>
          </div>

          {/* About Section */}
          <div className="bg-white rounded-xl shadow-md p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">About This Project</h2>
            <p className="text-gray-600 mb-4">
              This agricultural project follows the Musyarakah (profit-sharing) model compliant
              with Islamic finance principles. Investors provide capital to farmers, and profits
              from the harvest are shared according to the agreed ratio.
            </p>
            <ul className="space-y-2 text-gray-600">
              <li>Platform fee: 1%</li>
              <li>Investor returns are proportional to investment amount</li>
              <li>Funds disbursed directly to approved vendors</li>
            </ul>
          </div>
        </div>

        {/* Investment Sidebar */}
        <div className="md:col-span-1">
          <div className="bg-white rounded-xl shadow-md p-6 sticky top-4">
            <h3 className="text-lg font-bold text-gray-800 mb-4">Invest</h3>

            {!isConnected ? (
              <p className="text-gray-600 text-sm">
                Connect your wallet to invest in this project.
              </p>
            ) : !isCorrectNetwork ? (
              <p className="text-amber-600 text-sm">
                Please switch to Base Sepolia network.
              </p>
            ) : project.state !== ProjectState.FUNDRAISING ? (
              <p className="text-gray-600 text-sm">
                This project is no longer accepting investments.
              </p>
            ) : (
              <>
                {/* User Balance */}
                <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-600">Your IDRX Balance</p>
                  <p className="text-lg font-semibold">Rp {Number(idrxBalance).toLocaleString('id-ID')}</p>
                </div>

                {/* Current Investment */}
                {Number(myInvestment) > 0 && (
                  <div className="mb-4 p-3 bg-primary-50 rounded-lg">
                    <p className="text-sm text-primary-600">Your Investment</p>
                    <p className="text-lg font-semibold text-primary-700">
                      Rp {Number(myInvestment).toLocaleString('id-ID')}
                    </p>
                  </div>
                )}

                {/* Investment Input */}
                <div className="mb-4">
                  <label className="block text-sm text-gray-600 mb-2">
                    Investment Amount (IDRX)
                  </label>
                  <input
                    type="number"
                    value={investAmount}
                    onChange={(e) => setInvestAmount(e.target.value)}
                    placeholder="0"
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  />
                </div>

                {/* Error Display */}
                {error && (
                  <div className="mb-4 p-3 bg-red-50 text-red-600 text-sm rounded-lg">
                    {error}
                  </div>
                )}

                {/* Success Display */}
                {txHash && (
                  <div className="mb-4 p-3 bg-green-50 text-green-600 text-sm rounded-lg">
                    Transaction submitted!
                    <a
                      href={`${BASE_SEPOLIA.blockExplorerUrls[0]}/tx/${txHash}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block text-primary-600 underline mt-1"
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
                    className="w-full py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 disabled:opacity-50"
                  >
                    {approving ? 'Approving...' : '1. Approve IDRX'}
                  </button>
                  <button
                    onClick={handleInvest}
                    disabled={!investAmount || investing}
                    className="w-full py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50"
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
