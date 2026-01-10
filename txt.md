# Challenges I Ran Into

Building **TaniFi** within a hackathon timeline was far from smooth. Below are the real technical hurdles I faced during development, and how I solved them.

---

## 1. Environment & Tooling Setup (WSL + Docker + Foundry)

### The Challenge
At the very beginning, the development environment itself became a blocker:

- Docker was installed but the daemon could not start
- WSL was outdated and incompatible with Docker Desktop
- Ubuntu installation via Microsoft Store repeatedly failed
- Disk space constraints on the system drive caused additional friction

These issues prevented even basic services (PostgreSQL, Redis, NestJS) from running.

### How I Solved It
- Updated WSL manually using `wsl --update --web-download`
- Migrated development fully into **WSL2 (Ubuntu)** instead of mixing Windows + Linux tooling
- Reconfigured Docker Desktop to use WSL backend
- Carefully chose a **non-destructive storage strategy** (avoiding data loss on external SSDs)

**Lesson learned:**  
A “broken” environment can look like a code problem. Fixing infra early saved massive time later.

---

## 2. Prisma ORM Breaking Changes (Prisma v7)

### The Challenge
Prisma introduced breaking changes in version 7:

- `datasource.url` is no longer allowed in `schema.prisma`
- Connection configuration moved to `prisma.config.ts`
- `PrismaClient` must now be constructed with explicit options

This caused multiple runtime and compile-time errors such as:
- `P1012: datasource property 'url' is no longer supported`
- `PrismaClientInitializationError`

### How I Solved It
- Carefully read updated Prisma documentation
- Migrated DB config to `prisma.config.ts`
- Refactored `PrismaService` to properly extend `PrismaClient`
- Regenerated Prisma Client and re-ran migrations

**Lesson learned:**  
Using “latest” tech gives advantages, but only if you are ready to adapt to breaking changes.

---

## 3. USSD State Management & Stateless HTTP

### The Challenge
USSD flows are **stateful**, but HTTP requests are **stateless**.

Problems encountered:
- Session resets on every request
- User inputs like `2*5000000` not parsed correctly
- Back / Exit logic (`0`, `00`) behaving inconsistently
- Edge cases causing unexpected session termination

### How I Solved It
- Introduced **Redis** as a session store with TTL
- Implemented an explicit **finite state machine**:
  - `START`
  - `AWAIT_AMOUNT`
- Normalized USSD input parsing using `text.split('*')`
- Carefully handled `Back` and `Exit` flows per state

**Lesson learned:**  
USSD is deceptively simple. Treat it like a protocol, not just input text.

---

## 4. Audit Logging vs Database Schema Drift

### The Challenge
While adding audit logging (`UssdAudit`), the backend crashed with:

```

The table `public.UssdAudit` does not exist

```

The Prisma client expected the model, but the database schema had not been migrated.

### How I Solved It
- Verified Prisma schema vs actual database tables
- Added the missing model
- Ran `prisma migrate dev` to sync schema
- Confirmed audit rows were being written correctly

**Lesson learned:**  
Schema drift is silent but deadly. Migrations are not optional.

---

## 5. Lisk Sepolia Gas & Token Confusion

### The Challenge
After receiving funds from faucet:
- Blockscout showed **0.1 LSK**
- MetaMask showed balance
- Foundry (`cast balance`) returned **0**

This caused deployment failures with:
```

insufficient funds for gas * price + value

```

### How I Solved It
- Learned that **Lisk Sepolia gas is paid in ETH**, not LSK
- Realized I had received tokens, but **no gas currency**
- Adjusted funding strategy accordingly
- Confirmed gas requirements via RPC and Foundry output

**Lesson learned:**  
Not all L2s use the same native gas semantics. Always verify what pays gas.

---

## 6. Smart Contract Verification on Blockscout

### The Challenge
Contract deployment succeeded, but verification repeatedly failed with:
- `Fail - Unable to verify`
- `missing field sources`
- Partial mismatches between compiler metadata

### How I Solved It
- Confirmed on-chain bytecode using `cast code`
- Identified that Blockscout requires **Standard JSON Input**
- Used Foundry’s `--show-standard-json-input`
- Uploaded the correct JSON via Blockscout’s UI
- Successfully achieved **Verified (Partial Match)** status

**Lesson learned:**  
Verification is not automatic. Tooling literacy matters as much as Solidity.

---

## Final Reflection

Most challenges in this project were not about writing code, but about:
- Tooling maturity
- Protocol understanding
- Infrastructure debugging
- Reading documentation carefully under time pressure

Overcoming these issues significantly strengthened both the technical robustness of TaniFi and my confidence as a builder.

```