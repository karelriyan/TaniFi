// USSD Webhook Handler - Receives requests from Africa's Talking or similar USSD gateway
// This endpoint writes to the same database as the web interface

import { NextRequest, NextResponse } from 'next/server';

// In production, you would use Prisma Client to connect to the backend database
// For now, we'll proxy to the backend API

interface USSDRequest {
  sessionId: string;
  serviceCode: string;
  phoneNumber: string;
  text: string;
}

export async function POST(request: NextRequest) {
  try {
    const body: USSDRequest = await request.json();

    const { sessionId, serviceCode, phoneNumber, text } = body;

    // Forward the USSD request to the backend
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:3001';

    const response = await fetch(`${backendUrl}/v1/ussd`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        sessionId,
        serviceCode,
        phoneNumber,
        text,
      }),
    });

    const data = await response.json();

    // Return the USSD response
    return NextResponse.json(data, { status: response.status });

  } catch (error: any) {
    console.error('USSD webhook error:', error);

    return NextResponse.json(
      {
        message: 'END Maaf, terjadi kesalahan. Silakan coba lagi.',
        error: error.message
      },
      { status: 500 }
    );
  }
}

// GET endpoint for testing
export async function GET() {
  return NextResponse.json({
    message: 'USSD Webhook Endpoint',
    status: 'active',
    endpoints: {
      POST: '/api/ussd-hook - Receive USSD requests from gateway',
    },
    example: {
      sessionId: 'ATUid_123456',
      serviceCode: '*123#',
      phoneNumber: '+6281234567890',
      text: '1*2*500000',
    },
  });
}
