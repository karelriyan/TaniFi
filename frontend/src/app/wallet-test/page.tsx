'use client';

import { useState } from 'react';
import { useWalletContext } from '@/components/WalletProvider';

export default function WalletTestPage() {
  const { address, isConnected, isConnecting, connect, disconnect, error } = useWalletContext();
  const [testResult, setTestResult] = useState<string>('');

  const testDirectConnection = async () => {
    try {
      setTestResult('Testing...');
      const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
      setTestResult(`✅ Success! Connected to: ${accounts[0]}`);
    } catch (err: any) {
      setTestResult(`❌ Error: ${err.message}`);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-2xl">
      <h1 className="text-3xl font-bold mb-8">Wallet Connection Test</h1>

      {/* MetaMask Detection */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-4">
        <h2 className="text-xl font-semibold mb-4">1. MetaMask Detection</h2>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span>MetaMask Installed:</span>
            <span className={typeof window !== 'undefined' && window.ethereum ? 'text-green-600' : 'text-red-600'}>
              {typeof window !== 'undefined' && window.ethereum ? '✅ Yes' : '❌ No'}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span>Is MetaMask:</span>
            <span className={typeof window !== 'undefined' && window.ethereum?.isMetaMask ? 'text-green-600' : 'text-red-600'}>
              {typeof window !== 'undefined' && window.ethereum?.isMetaMask ? '✅ Yes' : '❌ No'}
            </span>
          </div>
        </div>
      </div>

      {/* Direct Connection Test */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-4">
        <h2 className="text-xl font-semibold mb-4">2. Direct Connection Test</h2>
        <button
          onClick={testDirectConnection}
          className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 mb-2"
        >
          Test Direct MetaMask Connection
        </button>
        {testResult && (
          <div className="mt-2 p-3 bg-gray-100 rounded text-sm">
            {testResult}
          </div>
        )}
      </div>

      {/* Wallet Provider Test */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-4">
        <h2 className="text-xl font-semibold mb-4">3. Wallet Provider Test</h2>
        <div className="space-y-2 mb-4">
          <div className="flex items-center justify-between">
            <span>Connected:</span>
            <span className={isConnected ? 'text-green-600' : 'text-gray-600'}>
              {isConnected ? '✅ Yes' : '❌ No'}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span>Connecting:</span>
            <span>{isConnecting ? '⏳ Yes' : 'No'}</span>
          </div>
          {address && (
            <div className="flex items-center justify-between">
              <span>Address:</span>
              <span className="font-mono text-sm">{address}</span>
            </div>
          )}
          {error && (
            <div className="text-red-600 text-sm">
              Error: {error}
            </div>
          )}
        </div>
        <div className="space-y-2">
          <button
            onClick={connect}
            disabled={isConnecting || isConnected}
            className="w-full px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
          >
            {isConnecting ? 'Connecting...' : isConnected ? 'Already Connected' : 'Connect via Provider'}
          </button>
          {isConnected && (
            <button
              onClick={disconnect}
              className="w-full px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
            >
              Disconnect
            </button>
          )}
        </div>
      </div>

      {/* Instructions */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h3 className="font-semibold mb-2">Instructions:</h3>
        <ol className="list-decimal list-inside space-y-1 text-sm">
          <li>Check if MetaMask is detected (should show ✅)</li>
          <li>Click "Test Direct MetaMask Connection" - MetaMask popup should appear</li>
          <li>Try "Connect via Provider" to test the wallet hook</li>
        </ol>
      </div>
    </div>
  );
}
