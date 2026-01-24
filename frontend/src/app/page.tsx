'use client';

import { useState, useEffect } from 'react';
import { ethers, Contract } from 'ethers';
import { ProjectCard } from '@/components/ProjectCard';
import { useWalletContext } from '@/components/WalletProvider';
import { ProjectDisplay, ProjectState } from '@/types';
import { TANI_VAULT_ABI, CONTRACT_ADDRESSES, BASE_SEPOLIA } from '@/lib/contracts';

export default function HomePage() {
  const { isConnected, isCorrectNetwork } = useWalletContext();
  const [projects, setProjects] = useState<ProjectDisplay[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchProjects();
  }, []);

  const fetchProjects = async () => {
    try {
      setLoading(true);
      setError(null);

      // Use public RPC to fetch projects
      const provider = new ethers.JsonRpcProvider(BASE_SEPOLIA.rpcUrls[0]);
      const vault = new Contract(
        CONTRACT_ADDRESSES.TANI_VAULT,
        TANI_VAULT_ABI,
        provider
      );

      const count = await vault.projectCount();
      const projectCount = Number(count);

      const projectList: ProjectDisplay[] = [];

      for (let i = 0; i < projectCount; i++) {
        try {
          const project = await vault.getProject(i);

          const targetAmount = Number(ethers.formatUnits(project.targetAmount, 2));
          const fundedAmount = Number(ethers.formatUnits(project.fundedAmount, 2));
          const fundingProgress = targetAmount > 0 ? (fundedAmount / targetAmount) * 100 : 0;

          projectList.push({
            id: Number(project.id),
            farmer: project.farmer,
            cooperative: project.cooperative,
            targetAmount: targetAmount.toLocaleString('id-ID'),
            fundedAmount: fundedAmount.toLocaleString('id-ID'),
            fundingProgress,
            farmerShare: Number(project.farmerShareBps) / 100,
            investorShare: Number(project.investorShareBps) / 100,
            harvestDate: new Date(Number(project.harvestTime) * 1000).toLocaleDateString('id-ID'),
            state: Number(project.state) as ProjectState,
            stateName: ProjectState[Number(project.state)],
          });
        } catch (err) {
          console.error(`Failed to fetch project ${i}:`, err);
        }
      }

      setProjects(projectList);
    } catch (err: any) {
      console.error('Failed to fetch projects:', err);
      setError('Failed to load projects. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* Hero Section */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-800 mb-4">
          Invest in Indonesian Agriculture
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          TaniFi connects investors with smallholder farmers through Sharia-compliant
          profit sharing. Earn returns while supporting sustainable agriculture.
        </p>
      </div>

      {/* Stats Bar */}
      <div className="grid grid-cols-3 gap-6 mb-12">
        <div className="bg-white rounded-xl p-6 shadow-md text-center">
          <p className="text-3xl font-bold text-primary-600">{projects.length}</p>
          <p className="text-gray-600">Active Projects</p>
        </div>
        <div className="bg-white rounded-xl p-6 shadow-md text-center">
          <p className="text-3xl font-bold text-primary-600">70%</p>
          <p className="text-gray-600">Investor Share</p>
        </div>
        <div className="bg-white rounded-xl p-6 shadow-md text-center">
          <p className="text-3xl font-bold text-primary-600">IDRX</p>
          <p className="text-gray-600">Stablecoin</p>
        </div>
      </div>

      {/* Connection Warning */}
      {!isConnected && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 mb-8">
          <p className="text-amber-800">
            Connect your wallet to invest in projects.
          </p>
        </div>
      )}

      {isConnected && !isCorrectNetwork && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-8">
          <p className="text-red-800">
            Please switch to Base Sepolia network to interact with projects.
          </p>
        </div>
      )}

      {/* Projects Grid */}
      <h2 className="text-2xl font-bold text-gray-800 mb-6">Available Projects</h2>

      {loading ? (
        <div className="flex justify-center items-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
        </div>
      ) : error ? (
        <div className="text-center py-12">
          <p className="text-red-600 mb-4">{error}</p>
          <button
            onClick={fetchProjects}
            className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
          >
            Retry
          </button>
        </div>
      ) : projects.length === 0 ? (
        <div className="text-center py-12 bg-white rounded-xl shadow-md">
          <span className="text-6xl mb-4 block">🌱</span>
          <p className="text-gray-600">No projects available yet.</p>
          <p className="text-sm text-gray-500 mt-2">
            Check back soon for new investment opportunities!
          </p>
        </div>
      ) : (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {projects.map((project) => (
            <ProjectCard key={project.id} project={project} />
          ))}
        </div>
      )}
    </div>
  );
}
