'use client';

import { useState } from 'react';
import Image from 'next/image';
import { FarmerRegistrationForm } from '@/components/FarmerRegistrationForm';
import { CooperativeDashboard } from '@/components/CooperativeDashboard';

type ViewType = 'select' | 'farmer' | 'cooperative';

export default function OnboardingPage() {
  const [currentView, setCurrentView] = useState<ViewType>('select');

  if (currentView === 'farmer') {
    return (
      <div className="min-h-screen py-12 px-4">
        <div className="mb-6">
          <button
            onClick={() => setCurrentView('select')}
            className="glass-strong px-4 py-2 rounded-lg text-primary-700 hover:text-primary-800 font-semibold flex items-center gap-2 transition-all hover:shadow-lg"
          >
            ← Kembali
          </button>
        </div>
        <FarmerRegistrationForm />
      </div>
    );
  }

  if (currentView === 'cooperative') {
    return (
      <div className="min-h-screen py-12 px-4">
        <div className="mb-6">
          <button
            onClick={() => setCurrentView('select')}
            className="glass-strong px-4 py-2 rounded-lg text-primary-700 hover:text-primary-800 font-semibold flex items-center gap-2 transition-all hover:shadow-lg"
          >
            ← Kembali
          </button>
        </div>
        <CooperativeDashboard />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center py-12 px-4">
      <div className="max-w-5xl w-full">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex justify-center mb-6">
            <Image
              src="/tanifi-logo.png"
              alt="TaniFi Logo"
              width={120}
              height={120}
              className="object-contain drop-shadow-lg"
            />
          </div>
          <h1 className="text-4xl font-bold gradient-text-animated mb-3">
            Selamat Datang di TaniFi
          </h1>
          <p className="text-lg text-gray-700 font-medium">
            Platform Pembiayaan Usaha Tani Berbasis Blockchain
          </p>
        </div>

        {/* Role Selection */}
        <div className="grid md:grid-cols-2 gap-8">
          {/* Farmer Card */}
          <button
            onClick={() => setCurrentView('farmer')}
            className="glass-card rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 hover:scale-105 text-left group"
          >
            <div className="text-6xl mb-4 drop-shadow-lg">🌾</div>
            <h2 className="text-2xl font-bold gradient-text mb-3">
              Saya Petani
            </h2>
            <p className="text-gray-700 mb-6 font-medium">
              Daftar untuk mendapatkan pembiayaan usaha tani dengan sistem bagi hasil yang adil
            </p>
            <div className="space-y-2 text-sm text-gray-700">
              <div className="flex items-start gap-2">
                <span className="text-primary-600 mt-0.5 font-bold">✓</span>
                <span>Proses pendaftaran mudah</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-primary-600 mt-0.5 font-bold">✓</span>
                <span>Akses via Web atau USSD (*123#)</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-primary-600 mt-0.5 font-bold">✓</span>
                <span>Verifikasi oleh koperasi terpercaya</span>
              </div>
            </div>
            <div className="mt-6 flex items-center gap-2 text-primary-700 font-semibold">
              <span>Daftar Sekarang</span>
              <span className="group-hover:translate-x-2 transition-transform duration-300">→</span>
            </div>
          </button>

          {/* Cooperative Card */}
          <button
            onClick={() => setCurrentView('cooperative')}
            className="glass-card rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 hover:scale-105 text-left group"
          >
            <div className="text-6xl mb-4 drop-shadow-lg">🏛️</div>
            <h2 className="text-2xl font-bold gradient-text mb-3">
              Saya Koperasi
            </h2>
            <p className="text-gray-700 mb-6 font-medium">
              Kelola dan verifikasi pendaftaran petani, pantau proyek pembiayaan
            </p>
            <div className="space-y-2 text-sm text-gray-700">
              <div className="flex items-start gap-2">
                <span className="text-primary-600 mt-0.5 font-bold">✓</span>
                <span>Dashboard verifikasi petani</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-primary-600 mt-0.5 font-bold">✓</span>
                <span>Data real-time dari Web & USSD</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-primary-600 mt-0.5 font-bold">✓</span>
                <span>Manajemen proyek terpadu</span>
              </div>
            </div>
            <div className="mt-6 flex items-center gap-2 text-primary-700 font-semibold">
              <span>Akses Dashboard</span>
              <span className="group-hover:translate-x-2 transition-transform duration-300">→</span>
            </div>
          </button>
        </div>

        {/* Info Section */}
        <div className="mt-12 glass-card rounded-2xl shadow-xl p-6">
          <h3 className="font-bold gradient-text mb-4 flex items-center gap-2 text-lg">
            <span>💡</span>
            <span>Tentang TaniFi</span>
          </h3>
          <div className="grid md:grid-cols-3 gap-6 text-sm text-gray-700">
            <div>
              <p className="font-semibold text-primary-700 mb-1">Berbasis Blockchain</p>
              <p>Smart contract di Base Sepolia menjamin transparansi dan keamanan</p>
            </div>
            <div>
              <p className="font-semibold text-primary-700 mb-1">Syariah Compliant</p>
              <p>Sistem Musyarakah dengan bagi hasil 70/30 sesuai prinsip Islam</p>
            </div>
            <div>
              <p className="font-semibold text-primary-700 mb-1">Inklusif</p>
              <p>Akses via smartphone (Web) atau feature phone (USSD)</p>
            </div>
          </div>
        </div>

        {/* USSD Info */}
        <div className="mt-6 glass-strong rounded-2xl p-6 border border-primary-300 shadow-lg">
          <div className="flex items-start gap-4">
            <div className="text-4xl drop-shadow-lg">📱</div>
            <div>
              <h4 className="font-bold gradient-text mb-2 text-lg">
                Petani Tanpa Smartphone?
              </h4>
              <p className="text-gray-700 text-sm mb-2 font-medium">
                Kami juga melayani pendaftaran via USSD untuk petani yang hanya memiliki handphone biasa.
              </p>
              <p className="text-gray-700 text-sm">
                Dial: <span className="font-mono font-bold text-primary-700 text-base">*123#</span> (Coming Soon)
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
