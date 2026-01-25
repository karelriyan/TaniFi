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
      <div className="min-h-screen bg-gradient-to-br from-primary-50 to-green-50 py-12 px-4">
        <div className="mb-6">
          <button
            onClick={() => setCurrentView('select')}
            className="text-primary-600 hover:text-primary-700 font-medium flex items-center gap-2"
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
      <div className="min-h-screen bg-gradient-to-br from-primary-50 to-blue-50 py-12 px-4">
        <div className="mb-6">
          <button
            onClick={() => setCurrentView('select')}
            className="text-primary-600 hover:text-primary-700 font-medium flex items-center gap-2"
          >
            ← Kembali
          </button>
        </div>
        <CooperativeDashboard />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-green-50 to-blue-50 flex items-center justify-center py-12 px-4">
      <div className="max-w-5xl w-full">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex justify-center mb-6">
            <Image
              src="/tanifi-logo.png"
              alt="TaniFi Logo"
              width={120}
              height={120}
              className="object-contain"
            />
          </div>
          <h1 className="text-4xl font-bold text-gray-800 mb-3">
            Selamat Datang di TaniFi
          </h1>
          <p className="text-lg text-gray-600">
            Platform Pembiayaan Usaha Tani Berbasis Blockchain
          </p>
        </div>

        {/* Role Selection */}
        <div className="grid md:grid-cols-2 gap-8">
          {/* Farmer Card */}
          <button
            onClick={() => setCurrentView('farmer')}
            className="bg-white rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-all transform hover:-translate-y-1 text-left group"
          >
            <div className="text-6xl mb-4">🌾</div>
            <h2 className="text-2xl font-bold text-gray-800 mb-3 group-hover:text-primary-600 transition-colors">
              Saya Petani
            </h2>
            <p className="text-gray-600 mb-6">
              Daftar untuk mendapatkan pembiayaan usaha tani dengan sistem bagi hasil yang adil
            </p>
            <div className="space-y-2 text-sm text-gray-700">
              <div className="flex items-start gap-2">
                <span className="text-green-500 mt-0.5">✓</span>
                <span>Proses pendaftaran mudah</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-500 mt-0.5">✓</span>
                <span>Akses via Web atau USSD (*123#)</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-500 mt-0.5">✓</span>
                <span>Verifikasi oleh koperasi terpercaya</span>
              </div>
            </div>
            <div className="mt-6 flex items-center gap-2 text-primary-600 font-medium">
              <span>Daftar Sekarang</span>
              <span className="group-hover:translate-x-1 transition-transform">→</span>
            </div>
          </button>

          {/* Cooperative Card */}
          <button
            onClick={() => setCurrentView('cooperative')}
            className="bg-white rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-all transform hover:-translate-y-1 text-left group"
          >
            <div className="text-6xl mb-4">🏛️</div>
            <h2 className="text-2xl font-bold text-gray-800 mb-3 group-hover:text-blue-600 transition-colors">
              Saya Koperasi
            </h2>
            <p className="text-gray-600 mb-6">
              Kelola dan verifikasi pendaftaran petani, pantau proyek pembiayaan
            </p>
            <div className="space-y-2 text-sm text-gray-700">
              <div className="flex items-start gap-2">
                <span className="text-blue-500 mt-0.5">✓</span>
                <span>Dashboard verifikasi petani</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-blue-500 mt-0.5">✓</span>
                <span>Data real-time dari Web & USSD</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-blue-500 mt-0.5">✓</span>
                <span>Manajemen proyek terpadu</span>
              </div>
            </div>
            <div className="mt-6 flex items-center gap-2 text-blue-600 font-medium">
              <span>Akses Dashboard</span>
              <span className="group-hover:translate-x-1 transition-transform">→</span>
            </div>
          </button>
        </div>

        {/* Info Section */}
        <div className="mt-12 bg-white rounded-xl shadow-lg p-6">
          <h3 className="font-bold text-gray-800 mb-3 flex items-center gap-2">
            <span>💡</span>
            <span>Tentang TaniFi</span>
          </h3>
          <div className="grid md:grid-cols-3 gap-6 text-sm text-gray-600">
            <div>
              <p className="font-medium text-gray-800 mb-1">Berbasis Blockchain</p>
              <p>Smart contract di Base Sepolia menjamin transparansi dan keamanan</p>
            </div>
            <div>
              <p className="font-medium text-gray-800 mb-1">Syariah Compliant</p>
              <p>Sistem Musyarakah dengan bagi hasil 70/30 sesuai prinsip Islam</p>
            </div>
            <div>
              <p className="font-medium text-gray-800 mb-1">Inklusif</p>
              <p>Akses via smartphone (Web) atau feature phone (USSD)</p>
            </div>
          </div>
        </div>

        {/* USSD Info */}
        <div className="mt-6 bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl p-6 border border-purple-200">
          <div className="flex items-start gap-4">
            <div className="text-4xl">📱</div>
            <div>
              <h4 className="font-bold text-gray-800 mb-2">
                Petani Tanpa Smartphone?
              </h4>
              <p className="text-gray-700 text-sm mb-2">
                Kami juga melayani pendaftaran via USSD untuk petani yang hanya memiliki handphone biasa.
              </p>
              <p className="text-gray-600 text-sm">
                Dial: <span className="font-mono font-bold text-purple-700">*123#</span> (Coming Soon)
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
