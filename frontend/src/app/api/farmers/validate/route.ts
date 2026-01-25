// Cooperative Farmer Validation API
// Allows authorized cooperatives to verify pending farmers

import { NextRequest, NextResponse } from 'next/server';

interface ValidationRequest {
  farmerId: string;
  cooperativeAddress: string;
  action: 'VERIFY' | 'REJECT';
  reason?: string;
}

export async function POST(request: NextRequest) {
  try {
    const body: ValidationRequest = await request.json();

    const { farmerId, cooperativeAddress, action, reason } = body;

    // Validation
    if (!farmerId || !cooperativeAddress || !action) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      );
    }

    if (!['VERIFY', 'REJECT'].includes(action)) {
      return NextResponse.json(
        { error: 'Invalid action. Must be VERIFY or REJECT' },
        { status: 400 }
      );
    }

    // Forward to backend API
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:3001';

    const response = await fetch(`${backendUrl}/v1/admin/farmers/${farmerId}/${action.toLowerCase()}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        cooperativeAddress,
        reason,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(error, { status: response.status });
    }

    const data = await response.json();

    return NextResponse.json(data, { status: 200 });

  } catch (error: any) {
    console.error('Farmer validation error:', error);

    return NextResponse.json(
      {
        error: 'Validation failed',
        message: error.message
      },
      { status: 500 }
    );
  }
}
