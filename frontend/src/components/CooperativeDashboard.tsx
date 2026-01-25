'use client';

import { useState, useEffect } from 'react';
import { useWalletContext } from './WalletProvider';

interface Farmer {
  id: string;
  farmerName: string;
  farmerNIK: string;
  phoneHash: string;
  landSize: number;
  location: string;
  registrationMethod: string;
  kycStatus: string;
  createdAt: string;
  walletAddress?: string;
}

export function CooperativeDashboard() {
  const { address, isConnected } = useWalletContext();
  const [farmers, setFarmers] = useState<Farmer[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedStatus, setSelectedStatus] = useState<'PENDING' | 'VERIFIED' | 'REJECTED'>('PENDING');
  const [processingId, setProcessingId] = useState<string | null>(null);

  useEffect(() => {
    if (isConnected) {
      fetchFarmers();
    }
  }, [isConnected, selectedStatus]);

  const fetchFarmers = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/farmers/pending?status=${selectedStatus}`);

      if (!response.ok) {
        throw new Error('Failed to fetch farmers');
      }

      const data = await response.json();
      setFarmers(data.farmers || data || []);

    } catch (err: any) {
      console.error('Fetch farmers error:', err);
      setError(err.message || 'Gagal memuat data petani');
      setFarmers([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleValidate = async (farmerId: string, action: 'VERIFY' | 'REJECT') => {
    if (!address) {
      alert('Harap connect wallet terlebih dahulu');
      return;
    }

    if (!confirm(`Apakah Anda yakin ingin ${action === 'VERIFY' ? 'memverifikasi' : 'menolak'} petani ini?`)) {
      return;
    }

    setProcessingId(farmerId);

    try {
      const response = await fetch('/api/farmers/validate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          farmerId,
          cooperativeAddress: address,
          action,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || errorData.message || 'Validasi gagal');
      }

      const data = await response.json();
      console.log('Validation successful:', data);

      alert(`✅ Petani berhasil ${action === 'VERIFY' ? 'diverifikasi' : 'ditolak'}`);

      // Refresh the list
      fetchFarmers();

    } catch (err: any) {
      console.error('Validation error:', err);
      alert(`❌ Error: ${err.message}`);
    } finally {
      setProcessingId(null);
    }
  };

  if (!isConnected) {
    return (
      <div className="max-w-6xl mx-auto glass-card rounded-2xl shadow-2xl p-8">
        <div className="text-center py-12">
          <div className="text-6xl mb-4">🔒</div>
          <h2 className="text-2xl font-bold gradient-text mb-2">
            Autentikasi Diperlukan
          </h2>
          <p className="text-gray-700 mb-6">
            Hubungkan wallet Anda untuk mengakses dashboard koperasi
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto glass-card rounded-2xl shadow-2xl p-8">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold gradient-text mb-2">
          Dashboard Koperasi 🏛️
        </h2>
        <p className="text-gray-700">
          Kelola dan verifikasi pendaftaran petani
        </p>
        <p className="text-sm text-gray-600 mt-1 font-medium">
          Connected as: {address?.slice(0, 6)}...{address?.slice(-4)}
        </p>
      </div>

      {/* Status Filter */}
      <div className="mb-6 flex gap-3">
        <button
          onClick={() => setSelectedStatus('PENDING')}
          className={`px-5 py-2.5 rounded-lg font-semibold transition-all duration-300 ${
            selectedStatus === 'PENDING'
              ? 'bg-gradient-to-r from-amber-500 to-amber-600 text-white shadow-lg'
              : 'glass text-gray-700 hover:glass-strong'
          }`}
        >
          Pending
        </button>
        <button
          onClick={() => setSelectedStatus('VERIFIED')}
          className={`px-5 py-2.5 rounded-lg font-semibold transition-all duration-300 ${
            selectedStatus === 'VERIFIED'
              ? 'bg-gradient-to-r from-primary-500 to-primary-600 text-white shadow-lg'
              : 'glass text-gray-700 hover:glass-strong'
          }`}
        >
          Verified
        </button>
        <button
          onClick={() => setSelectedStatus('REJECTED')}
          className={`px-5 py-2.5 rounded-lg font-semibold transition-all duration-300 ${
            selectedStatus === 'REJECTED'
              ? 'bg-gradient-to-r from-red-500 to-red-600 text-white shadow-lg'
              : 'glass text-gray-700 hover:glass-strong'
          }`}
        >
          Rejected
        </button>
      </div>

      {/* Error Message */}
      {error && (
        <div className="mb-6 p-4 glass border-red-300 rounded-lg">
          <p className="text-red-800 font-medium">❌ {error}</p>
        </div>
      )}

      {/* Loading State */}
      {isLoading && (
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto"></div>
          <p className="text-gray-700 mt-4 font-medium">Memuat data...</p>
        </div>
      )}

      {/* Farmers Table */}
      {!isLoading && farmers.length === 0 && (
        <div className="text-center py-12">
          <div className="text-6xl mb-4">📋</div>
          <p className="text-gray-700 font-medium">Tidak ada petani dengan status {selectedStatus}</p>
        </div>
      )}

      {!isLoading && farmers.length > 0 && (
        <div className="overflow-x-auto glass-dark rounded-xl p-1">
          <table className="w-full">
            <thead>
              <tr className="glass-strong border-b border-primary-200">
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-800">Nama</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-800">NIK</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-800">Luas Lahan</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-800">Lokasi</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-800">Metode</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-800">Tanggal</th>
                {selectedStatus === 'PENDING' && (
                  <th className="px-4 py-3 text-left text-sm font-semibold text-gray-800">Aksi</th>
                )}
              </tr>
            </thead>
            <tbody className="divide-y divide-primary-100">
              {farmers.map((farmer) => (
                <tr key={farmer.id} className="hover:glass-strong transition-all">
                  <td className="px-4 py-4 text-sm text-gray-800">
                    {farmer.farmerName || 'N/A'}
                  </td>
                  <td className="px-4 py-4 text-sm text-gray-600 font-mono">
                    {farmer.farmerNIK || 'N/A'}
                  </td>
                  <td className="px-4 py-4 text-sm text-gray-800">
                    {farmer.landSize ? `${farmer.landSize} ha` : 'N/A'}
                  </td>
                  <td className="px-4 py-4 text-sm text-gray-600 max-w-xs truncate">
                    {farmer.location || 'N/A'}
                  </td>
                  <td className="px-4 py-4 text-sm">
                    <span className={`px-2.5 py-1 rounded-full text-xs font-semibold glass ${
                      farmer.registrationMethod === 'USSD'
                        ? 'border-purple-300 text-purple-800'
                        : 'border-primary-300 text-primary-800'
                    }`}>
                      {farmer.registrationMethod}
                    </span>
                  </td>
                  <td className="px-4 py-4 text-sm text-gray-600">
                    {new Date(farmer.createdAt).toLocaleDateString('id-ID')}
                  </td>
                  {selectedStatus === 'PENDING' && (
                    <td className="px-4 py-4 text-sm">
                      <div className="flex gap-2">
                        <button
                          onClick={() => handleValidate(farmer.id, 'VERIFY')}
                          disabled={processingId === farmer.id}
                          className="px-3 py-1.5 bg-gradient-to-r from-primary-500 to-primary-600 text-white rounded-lg hover:from-primary-600 hover:to-primary-700 disabled:opacity-50 disabled:cursor-not-allowed text-xs font-semibold shadow-md hover:shadow-lg transition-all"
                        >
                          {processingId === farmer.id ? '...' : '✓ Verify'}
                        </button>
                        <button
                          onClick={() => handleValidate(farmer.id, 'REJECT')}
                          disabled={processingId === farmer.id}
                          className="px-3 py-1.5 bg-gradient-to-r from-red-500 to-red-600 text-white rounded-lg hover:from-red-600 hover:to-red-700 disabled:opacity-50 disabled:cursor-not-allowed text-xs font-semibold shadow-md hover:shadow-lg transition-all"
                        >
                          {processingId === farmer.id ? '...' : '✗ Reject'}
                        </button>
                      </div>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Stats */}
      {!isLoading && farmers.length > 0 && (
        <div className="mt-6 p-4 glass rounded-lg border-primary-200">
          <p className="text-sm text-gray-700 font-medium">
            Total: <span className="font-bold gradient-text">{farmers.length}</span> petani dengan status{' '}
            <span className="font-bold text-primary-700">{selectedStatus}</span>
          </p>
        </div>
      )}
    </div>
  );
}
