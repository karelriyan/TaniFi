'use client';

import Link from 'next/link';
import { ProjectDisplay, ProjectState } from '@/types';

interface ProjectCardProps {
  project: ProjectDisplay;
}

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

export function ProjectCard({ project }: ProjectCardProps) {
  return (
    <div className="glass-card rounded-xl overflow-hidden hover:shadow-2xl transition-all duration-300 transform hover:scale-105">
      {/* Project Image/Header */}
      <div className="h-32 bg-gradient-to-br from-primary-400 via-primary-500 to-primary-600 flex items-center justify-center relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-t from-primary-700/20 to-transparent"></div>
        <span className="text-6xl relative z-10 drop-shadow-lg">
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
            <span className="font-semibold gradient-text">{project.fundingProgress.toFixed(1)}%</span>
          </div>
          <div className="w-full glass-dark rounded-full h-2.5 overflow-hidden">
            <div
              className="bg-gradient-to-r from-primary-500 via-primary-600 to-primary-700 h-2.5 rounded-full transition-all duration-500 shadow-lg"
              style={{ width: `${Math.min(project.fundingProgress, 100)}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-gray-600 mt-1.5 font-medium">
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
          className="block w-full text-center py-2.5 px-4 bg-gradient-to-r from-primary-600 to-primary-700 text-white font-semibold rounded-lg hover:from-primary-700 hover:to-primary-800 transition-all duration-300 shadow-md hover:shadow-xl transform hover:-translate-y-0.5"
        >
          {project.state === ProjectState.FUNDRAISING ? 'Invest Now' : 'View Details'}
        </Link>
      </div>
    </div>
  );
}
