/**
 * USSD module wiring controller and service providers.
 */
import { Module } from '@nestjs/common';
import { UssdController } from './ussd.controller';
import { UssdService } from './ussd.service';

@Module({
  controllers: [UssdController],
  providers: [UssdService]
})
export class UssdModule {}
