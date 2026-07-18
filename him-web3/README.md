# HIM Web3 Applet

Full-stack blockchain tooling for **HIM** (agentic multi-mini-model being).

## Standard tooling

- **Node.js + npm** (required)
- Install web3 libs with **install_applet_package**:

```bash
cd him-web3
npm install
npm run install:applet -- ethers
npm run install:applet -- viem
# or
node scripts/install_applet_package.js ethers viem
```

## Security architecture

```
React UI  ──fetch──►  Express /api/*  ──RPC──►  Alchemy / Infura
   (no keys)              (.env secrets)
```

- API keys live only in `him-web3/.env` (gitignored)
- Frontend never imports `ethers`/`viem` with secrets
- Contract reads via `POST /api/chain/call`

## Setup

```bash
cd him-web3
cp .env.example .env
# edit ETH_RPC_URL=

npm run setup          # root + client deps
npm run build:client   # optional static UI
npm run server         # http://127.0.0.1:8787
```

Dev UI with proxy:

```bash
# terminal 1
npm run server
# terminal 2
npm run dev:client     # http://127.0.0.1:5173 → proxies /api
```

## API

| Route | Description |
|-------|-------------|
| `GET /api/health` | Server + RPC configured? |
| `GET /api/him` | HIM applet meta |
| `GET /api/chain/block-number` | Latest block (viem) |
| `GET /api/chain/balance/:address` | ETH balance (viem) |
| `GET /api/chain/block` | Latest block summary (ethers) |
| `POST /api/chain/call` | Read-only contract call |

## HIM (Python) integration

```python
from auro_native_llm.him.web3_tools import HimWeb3Tools
w = HimWeb3Tools()
print(w.install_applet_package(["ethers"]))
print(w.api_health())
```

```bash
python -m auro_native_llm.use --him "check web3 health and block number"
```
