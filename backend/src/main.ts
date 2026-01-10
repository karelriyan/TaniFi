/**
 * Application bootstrap for the HTTP API.
 * - Creates the Nest app from AppModule.
 * - Binds to PORT (default 3000) and listens on all interfaces.
 * - Logs the local URL once ready.
 */
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  const port = Number(process.env.PORT || 3000);
  await app.listen(port, '0.0.0.0');

  console.log(`Listening on http://localhost:${port}`);
}
bootstrap();
