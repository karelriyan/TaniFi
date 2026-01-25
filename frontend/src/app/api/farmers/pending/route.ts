// Get Pending Farmers API
// Returns list of farmers awaiting verification

import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const status = searchParams.get('status') || 'PENDING';

    // Forward to backend API
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:3001';

    const response = await fetch(`${backendUrl}/v1/admin/farmers?status=${status}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(error, { status: response.status });
    }

    const data = await response.json();

    return NextResponse.json(data, { status: 200 });

  } catch (error: any) {
    console.error('Fetch pending farmers error:', error);

    return NextResponse.json(
      {
        error: 'Failed to fetch farmers',
        message: error.message
      },
      { status: 500 }
    );
  }
}
