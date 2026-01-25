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
      <div className="max-w-6xl mx-auto bg-white rounded-xl shadow-lg p-8">
        <div className="text-center py-12">
          <div className="text-6xl mb-4">🔒</div>
          <h2 className="text-2xl font-bold text-gray-800 mb-2">
            Autentikasi Diperlukan
          </h2>
          <p className="text-gray-600 mb-6">
            Hubungkan wallet Anda untuk mengakses dashboard koperasi
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto bg-white rounded-xl shadow-lg p-8">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          Dashboard Koperasi 🏛️
        </h2>
        <p className="text-gray-600">
          Kelola dan verifikasi pendaftaran petani
        </p>
        <p className="text-sm text-gray-500 mt-1">
          Connected as: {address?.slice(0, 6)}...{address?.slice(-4)}
        </p>
      </div>

      {/* Status Filter */}
      <div className="mb-6 flex gap-3">
        <button
          onClick={() => setSelectedStatus('PENDING')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            selectedStatus === 'PENDING'
              ? 'bg-amber-500 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          Pending
        </button>
        <button
          onClick={() => setSelectedStatus('VERIFIED')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            selectedStatus === 'VERIFIED'
              ? 'bg-green-500 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          Verified
        </button>
        <button
          onClick={() => setSelectedStatus('REJECTED')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            selectedStatus === 'REJECTED'
              ? 'bg-red-500 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          Rejected
        </button>
      </div>

      {/* Error Message */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-800">❌ {error}</p>
        </div>
      )}

      {/* Loading State */}
      {isLoading && (
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto"></div>
          <p className="text-gray-600 mt-4">Memuat data...</p>
        </div>
      )}

      {/* Farmers Table */}
      {!isLoading && farmers.length === 0 && (
        <div className="text-center py-12">
          <div className="text-6xl mb-4">📋</div>
          <p className="text-gray-600">Tidak ada petani dengan status {selectedStatus}</p>
        </div>
      )}

      {!isLoading && farmers.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-200">
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">Nama</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">NIK</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">Luas Lahan</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">Lokasi</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">Metode</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">Tanggal</th>
                {selectedStatus === 'PENDING' && (
                  <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">Aksi</th>
                )}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {farmers.map((farmer) => (
                <tr key={farmer.id} className="hover:bg-gray-50">
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
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      farmer.registrationMethod === 'USSD'
                        ? 'bg-purple-100 text-purple-800'
                        : 'bg-blue-100 text-blue-800'
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
                          className="px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed text-xs font-medium"
                        >
                          {processingId === farmer.id ? '...' : '✓ Verify'}
                        </button>
                        <button
                          onClick={() => handleValidate(farmer.id, 'REJECT')}
                          disabled={processingId === farmer.id}
                          className="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600 disabled:opacity-50 disabled:cursor-not-allowed text-xs font-medium"
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
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <p className="text-sm text-gray-600">
            Total: <span className="font-semibold">{farmers.length}</span> petani dengan status{' '}
            <span className="font-semibold">{selectedStatus}</span>
          </p>
        </div>
      )}
    </div>
  );
}
