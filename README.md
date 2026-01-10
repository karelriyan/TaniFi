# 🌾 TaniFi — The Trustless Bridge to Prosperity

![Lisk Builders](https://img.shields.io/badge/Lisk%20Builders-Round%203-blue)
![Status](https://img.shields.io/badge/status-MVP-orange)
![License](https://img.shields.io/badge/license-MIT-green)

**TaniFi** is a **Sharia-compliant, decentralized agricultural finance protocol** that enables **unbanked farmers** to access capital through **USSD (2G phones)** while ensuring **trustless, transparent accounting on blockchain**.

Built for regions with **limited internet access**, TaniFi bridges **Telecom infrastructure (Web2)** with **DeFi (Web3)** using a hybrid, offline-first architecture.

---

## 📖 Table of Contents

* [The Problem](#-the-problem)
* [The Solution](#-the-solution)
* [How It Works](#️-how-it-works)
* [Architecture](#-architecture)
* [Key Features](#-key-features)
* [Tech Stack](#-tech-stack)
* [Pilot Project: Banyumas](#-pilot-project-banyumas)
* [Getting Started](#-getting-started)
* [Smart Contract Addresses](#-smart-contract-addresses)
* [License](#-license)
* [Team](#-team)

---

## 🚩 The Problem

Agriculture contributes significantly to Indonesia’s economy, yet **smallholder farmers remain financially excluded** due to structural issues:

### 1. Digital Divide

Over **40% of rural areas** suffer from poor or no internet access. Most farmers still rely on **feature phones**, making Web3 wallets and mobile apps inaccessible.

### 2. Predatory Financing

Farmers often depend on *tengkulak* (informal lenders) charging **up to 20% monthly interest**, trapping them in cycles of debt.

### 3. Trust Deficit

Several centralized agritech platforms collapsed due to **opaque fund management**, damaging investor confidence and harming farmers.

---

## 💡 The Solution

TaniFi introduces a **Hybrid USSD–Blockchain Protocol** that is:

* **Accessible** — works on any phone, no internet required
* **Trustless** — transparent, immutable accounting
* **Sharia-Compliant** — profit-sharing instead of interest

### Core Principles

1. **Offline-First Access**
   Farmers interact via **USSD menus** (`*777#`) — no smartphone, no data plan.

2. **On-Chain Transparency**
   All financial events are recorded using **triple-entry accounting** on blockchain.

3. **Musyarakah-Based Financing**
   Capital is provided through **profit-sharing**, aligning incentives between farmers and investors.

---

## ⚙️ How It Works

### 👨‍🌾 Farmer Journey (USSD)

1. Farmer dials `*777#`
2. Selects financing option (e.g. “Create Project”)
3. Backend translates the USSD request into a blockchain transaction
4. Funds are released to verified vendors
5. Farmer receives SMS confirmation

### 💰 Financial Flow

1. Investors deposit funds into the **TaniVault**
2. Funds are disbursed via verified, closed-loop vendors
3. After harvest, revenue is deposited
4. Smart contract distributes profit automatically
   *(e.g. 70% Investor / 30% Farmer)*

---

## 🏗 Architecture

TaniFi bridges **Telecom (Web2)** and **Blockchain (Web3)** using Account Abstraction.

```mermaid
graph TD
    User[Farmer<br/>(Feature Phone)] -- USSD (*777#) --> Telco
    Telco -- HTTP Callback --> Gateway

    subgraph Off-Chain Layer
        Gateway --> Auth
        Gateway --> Redis[Session State]
        Gateway --> MPC[MPC / Wallet Abstraction]
    end

    subgraph Lisk Layer 2
        MPC --> Bundler
        Bundler --> EntryPoint
        EntryPoint --> SmartWallet
        SmartWallet --> Vault
    end

    Investor --> Vault
```

---

## 🚀 Key Features

* 📱 **USSD Interface** — Works on any phone, zero internet dependency
* ⛽ **Gasless Transactions** — ERC-4337 abstraction for farmers
* 🤝 **Musyarakah Vault** — Automated profit-sharing logic
* 🧾 **Audit Trail** — Every request logged in Postgres (USSD audit log)
* 🔁 **Transaction Lifecycle** — Pending → confirmed (mocked for MVP)
* 🇮🇩 **IDR-Based Accounting** — Minimizes volatility exposure

---

## 🛠 Tech Stack

### Backend

* **Node.js**, **NestJS**
* **PostgreSQL** (financial & audit data)
* **Redis** (USSD session state)
* **Prisma ORM**

### Blockchain

* **Lisk Layer 2**
* **Solidity**, **Foundry**
* **ERC-4337 Account Abstraction**

### Infrastructure

* **Docker**
* **USSD Gateway Simulator** (hackathon mode)

---

## 📍 Pilot Project Plan: Banyumas

**Commodity:** Coconut Sugar (Gula Semut)
**Location:** Banyumas, Central Java

**Use Case:**
Farmers receive working capital instantly when delivering sugar to cooperatives, enabling **invoice-based supply chain financing** via USSD.

---

## 💻 Getting Started

### Prerequisites

* Node.js v18+
* Docker
* PostgreSQL
* Redis

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/tanifi-protocol.git
cd tanifi-protocol
```

### 2. Backend Setup

```bash
cd backend
npm install
npm run start:dev
```

### 3. Test USSD Flow

```bash
Invoke-RestMethod -Uri http://localhost:3000/v1/hooks/ussd \
  -Method POST \
  -ContentType "application/json" \
  -Body '{"sessionId":"S1","phoneNumber":"+6281234567890","text":""}'
```

---

## 🔗 Smart Contract Addresses (Testnet)

| Contract  | Address |
| --------- | ------- |
| TaniVault | `0x...` |
| IDRX Mock | `0x...` |

---

## 📜 License

MIT License — see [LICENSE](LICENSE)

---

## 👤 Developer

Built for **Lisk Builders Challenge – Round 3**

Solo Developer: Karel Tsalasatir Riyan

Universitas Jenderal Soedirman

---

> *“Empowering the hands that feed the nation — one block at a time.”*

---
