'use client';

import Link from 'next/link';
import { ProjectDisplay, ProjectState } from '@/types';

interface ProjectCardProps {
  project: ProjectDisplay;
}

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

export function ProjectCard({ project }: ProjectCardProps) {
  return (
    <div className="bg-white rounded-xl shadow-md overflow-hidden hover:shadow-lg transition-shadow">
      {/* Project Image/Header */}
      <div className="h-32 bg-gradient-to-r from-primary-400 to-primary-600 flex items-center justify-center">
        <span className="text-6xl">
          {project.commodity === 'gula_semut' ? '🍯' :
           project.commodity === 'padi' ? '🌾' :
           project.commodity === 'jagung' ? '🌽' : '🌱'}
        </span>
      </div>

      {/* Project Info */}
      <div className="p-5">
        <div className="flex justify-between items-start mb-3">
          <h3 className="text-lg font-semibold text-gray-800">
            Project #{project.id}
          </h3>
          <span className={`px-2 py-1 text-xs font-medium rounded-full ${STATE_COLORS[project.state]}`}>
            {STATE_NAMES[project.state]}
          </span>
        </div>

        {/* Commodity */}
        <p className="text-sm text-gray-600 mb-3">
          Commodity: {project.commodity || 'Agricultural'}
        </p>

        {/* Funding Progress */}
        <div className="mb-4">
          <div className="flex justify-between text-sm mb-1">
            <span className="text-gray-600">Funded</span>
            <span className="font-medium">{project.fundingProgress.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-primary-500 h-2 rounded-full transition-all"
              style={{ width: `${Math.min(project.fundingProgress, 100)}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>Rp {project.fundedAmount}</span>
            <span>Rp {project.targetAmount}</span>
          </div>
        </div>

        {/* Profit Share */}
        <div className="flex justify-between text-sm mb-4">
          <div>
            <span className="text-gray-600">Investor Share:</span>
            <span className="font-medium ml-1 text-primary-600">{project.investorShare}%</span>
          </div>
          <div>
            <span className="text-gray-600">Farmer Share:</span>
            <span className="font-medium ml-1">{project.farmerShare}%</span>
          </div>
        </div>

        {/* Harvest Date */}
        <p className="text-xs text-gray-500 mb-4">
          Expected Harvest: {project.harvestDate}
        </p>

        {/* Action Button */}
        <Link
          href={`/project/${project.id}`}
          className="block w-full text-center py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
        >
          {project.state === ProjectState.FUNDRAISING ? 'Invest Now' : 'View Details'}
        </Link>
      </div>
    </div>
  );
}
