'use client';

import { useState } from 'react';
import { useWalletContext } from './WalletProvider';

interface FormData {
  name: string;
  nik: string;
  phoneNumber: string;
  landSize: string;
  location: string;
}

export function FarmerRegistrationForm() {
  const { address, isConnected } = useWalletContext();
  const [formData, setFormData] = useState<FormData>({
    name: '',
    nik: '',
    phoneNumber: '',
    landSize: '',
    location: '',
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);
    setSuccess(false);

    try {
      // Validate
      if (!formData.name || !formData.nik || !formData.phoneNumber || !formData.landSize || !formData.location) {
        throw new Error('Semua field harus diisi');
      }

      if (!/^\d{16}$/.test(formData.nik)) {
        throw new Error('NIK harus 16 digit angka');
      }

      if (parseFloat(formData.landSize) <= 0) {
        throw new Error('Luas lahan harus lebih dari 0');
      }

      // Submit registration
      const response = await fetch('/api/farmers/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: formData.name,
          nik: formData.nik,
          phoneNumber: formData.phoneNumber,
          landSize: parseFloat(formData.landSize),
          location: formData.location,
          walletAddress: address || undefined,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || errorData.message || 'Pendaftaran gagal');
      }

      const data = await response.json();
      console.log('Registration successful:', data);

      setSuccess(true);
      setFormData({
        name: '',
        nik: '',
        phoneNumber: '',
        landSize: '',
        location: '',
      });

    } catch (err: any) {
      console.error('Registration error:', err);
      setError(err.message || 'Terjadi kesalahan saat pendaftaran');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto bg-white rounded-xl shadow-lg p-8">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          Pendaftaran Petani 🌾
        </h2>
        <p className="text-gray-600">
          Daftar untuk mendapatkan akses pembiayaan usaha tani
        </p>
      </div>

      {success && (
        <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
          <p className="text-green-800 font-medium">
            ✅ Pendaftaran berhasil! Data Anda sedang diverifikasi oleh koperasi.
          </p>
          <p className="text-green-600 text-sm mt-1">
            Anda akan dihubungi setelah verifikasi selesai.
          </p>
        </div>
      )}

      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-800 font-medium">❌ {error}</p>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Name */}
        <div>
          <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-2">
            Nama Lengkap <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            id="name"
            name="name"
            value={formData.name}
            onChange={handleChange}
            placeholder="Contoh: Budi Santoso"
            required
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>

        {/* NIK */}
        <div>
          <label htmlFor="nik" className="block text-sm font-medium text-gray-700 mb-2">
            NIK (Nomor Induk Kependudukan) <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            id="nik"
            name="nik"
            value={formData.nik}
            onChange={handleChange}
            placeholder="16 digit NIK"
            maxLength={16}
            required
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
          <p className="text-sm text-gray-500 mt-1">Harus 16 digit sesuai KTP</p>
        </div>

        {/* Phone Number */}
        <div>
          <label htmlFor="phoneNumber" className="block text-sm font-medium text-gray-700 mb-2">
            Nomor Telepon <span className="text-red-500">*</span>
          </label>
          <input
            type="tel"
            id="phoneNumber"
            name="phoneNumber"
            value={formData.phoneNumber}
            onChange={handleChange}
            placeholder="081234567890"
            required
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
          <p className="text-sm text-gray-500 mt-1">Format: 08xxx atau +62xxx</p>
        </div>

        {/* Land Size */}
        <div>
          <label htmlFor="landSize" className="block text-sm font-medium text-gray-700 mb-2">
            Luas Lahan (Hektar) <span className="text-red-500">*</span>
          </label>
          <input
            type="number"
            id="landSize"
            name="landSize"
            value={formData.landSize}
            onChange={handleChange}
            placeholder="Contoh: 2.5"
            step="0.1"
            min="0"
            required
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>

        {/* Location */}
        <div>
          <label htmlFor="location" className="block text-sm font-medium text-gray-700 mb-2">
            Lokasi Lahan <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            id="location"
            name="location"
            value={formData.location}
            onChange={handleChange}
            placeholder="Contoh: Desa Karangmangu, Kec. Baturaden, Kab. Banyumas"
            required
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>

        {/* Wallet Address (optional) */}
        {isConnected && (
          <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm text-blue-800">
              <span className="font-medium">Wallet Connected:</span>{' '}
              {address?.slice(0, 6)}...{address?.slice(-4)}
            </p>
            <p className="text-xs text-blue-600 mt-1">
              Alamat ini akan dikaitkan dengan akun petani Anda
            </p>
          </div>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          disabled={isSubmitting}
          className="w-full bg-primary-600 text-white font-medium py-3 px-6 rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isSubmitting ? 'Mendaftar...' : 'Daftar Sekarang'}
        </button>
      </form>

      {/* Info Section */}
      <div className="mt-8 p-4 bg-gray-50 rounded-lg">
        <h3 className="font-medium text-gray-800 mb-2">ℹ️ Informasi</h3>
        <ul className="text-sm text-gray-600 space-y-1">
          <li>• Data Anda akan diverifikasi oleh koperasi</li>
          <li>• Proses verifikasi memakan waktu 1-3 hari kerja</li>
          <li>• Anda juga bisa daftar via USSD dengan dial *123# (coming soon)</li>
        </ul>
      </div>
    </div>
  );
}
