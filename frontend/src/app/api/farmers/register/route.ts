// Farmer Registration API - Web Interface
// Writes to the same database as USSD

import { NextRequest, NextResponse } from 'next/server';

interface FarmerRegistrationRequest {
  name: string;
  nik: string;
  phoneNumber: string;
  landSize: number;
  location: string;
  walletAddress?: string;
}

export async function POST(request: NextRequest) {
  try {
    const body: FarmerRegistrationRequest = await request.json();

    const { name, nik, phoneNumber, landSize, location, walletAddress } = body;

    // Validation
    if (!name || !nik || !phoneNumber || !landSize || !location) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      );
    }

    // NIK validation (should be 16 digits)
    if (!/^\d{16}$/.test(nik)) {
      return NextResponse.json(
        { error: 'NIK must be 16 digits' },
        { status: 400 }
      );
    }

    // Phone number validation (Indonesian format)
    if (!/^(\+62|62|0)[0-9]{9,12}$/.test(phoneNumber)) {
      return NextResponse.json(
        { error: 'Invalid phone number format' },
        { status: 400 }
      );
    }

    // Forward to backend API
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:3001';

    const response = await fetch(`${backendUrl}/v1/farmers/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        name,
        nik,
        phoneNumber,
        landSize,
        location,
        walletAddress,
        registrationMethod: 'WEB',
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(error, { status: response.status });
    }

    const data = await response.json();

    return NextResponse.json(data, { status: 201 });

  } catch (error: any) {
    console.error('Farmer registration error:', error);

    return NextResponse.json(
      {
        error: 'Registration failed',
        message: error.message
      },
      { status: 500 }
    );
  }
}
