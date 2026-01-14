/**
 * Health check endpoint for monitoring and load balancers.
 */
import { Controller, Get } from '@nestjs/common';
import { SkipThrottle } from '@nestjs/throttler';

@Controller()
@SkipThrottle() // Don't rate limit health checks
export class HealthController {
  @Get('health')
  check() {
    return {
      ok: true,
      status: 'healthy',
      timestamp: new Date().toISOString(),
      service: 'tanifi-api',
      version: process.env.npm_package_version || '1.0.0',
    };
  }

  @Get()
  root() {
    return {
      ok: true,
      message: 'TaniFi API - Sharia-Compliant Agricultural Finance Protocol',
      docs: '/api',
      health: '/health',
    };
  }
}
