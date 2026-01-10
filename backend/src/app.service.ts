/**
 * Minimal app-level service used by the root controller.
 * Acts as a placeholder for shared or demo responses.
 */
import { Injectable } from '@nestjs/common';

@Injectable()
export class AppService {
  getHello(): string {
    return 'Hello World!';
  }
}
