-- CreateTable
CREATE TABLE "UssdAudit" (
    "id" TEXT NOT NULL,
    "requestId" TEXT NOT NULL,
    "sessionId" TEXT NOT NULL,
    "phoneHash" TEXT NOT NULL,
    "text" TEXT,
    "stateBefore" TEXT NOT NULL,
    "stateAfter" TEXT NOT NULL,
    "responseType" TEXT NOT NULL,
    "responseText" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "UssdAudit_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "UssdAudit_sessionId_idx" ON "UssdAudit"("sessionId");

-- CreateIndex
CREATE INDEX "UssdAudit_phoneHash_idx" ON "UssdAudit"("phoneHash");

-- CreateIndex
CREATE INDEX "UssdAudit_createdAt_idx" ON "UssdAudit"("createdAt");
