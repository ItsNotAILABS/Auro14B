/**
 * HIM web3 server — full-stack secure API for blockchain.
 *
 * Frontend (React) talks only to /api/* on this server.
 * RPC keys stay in process.env — never in the browser bundle.
 */
import express from "express";
import cors from "cors";
import path from "node:path";
import { fileURLToPath } from "node:url";
import dotenv from "dotenv";
import {
  healthPayload,
  hasRpc,
  getBlockNumber,
  getBalance,
  getBlock,
  callContract,
} from "./rpc.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.resolve(__dirname, "../.env") });

const app = express();
const PORT = Number(process.env.PORT || 8787);
const HOST = process.env.HOST || "127.0.0.1";
const CORS_ORIGIN = process.env.CORS_ORIGIN || true;

app.use(cors({ origin: CORS_ORIGIN }));
app.use(express.json({ limit: "1mb" }));

// ---------- /api/* (secure server-side) ----------
app.get("/api/health", (_req, res) => {
  res.json(healthPayload());
});

app.get("/api/chain/block-number", async (_req, res) => {
  try {
    if (!hasRpc()) {
      return res.status(503).json({
        ok: false,
        error: "RPC not configured",
        hint: "Copy him-web3/.env.example → .env and set ETH_RPC_URL",
      });
    }
    const blockNumber = await getBlockNumber();
    res.json({ ok: true, blockNumber, via: "viem" });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e.message || e) });
  }
});

app.get("/api/chain/balance/:address", async (req, res) => {
  try {
    if (!hasRpc()) {
      return res.status(503).json({ ok: false, error: "RPC not configured" });
    }
    const data = await getBalance(req.params.address);
    res.json({ ok: true, ...data, via: "viem" });
  } catch (e) {
    res.status(400).json({ ok: false, error: String(e.message || e) });
  }
});

app.get("/api/chain/block", async (req, res) => {
  try {
    if (!hasRpc()) {
      return res.status(503).json({ ok: false, error: "RPC not configured" });
    }
    const tag = req.query.tag || "latest";
    const block = await getBlock(tag);
    res.json({ ok: true, block, via: "ethers" });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e.message || e) });
  }
});

/** Read-only contract call — ABI + address from client, RPC key never leaves server */
app.post("/api/chain/call", async (req, res) => {
  try {
    if (!hasRpc()) {
      return res.status(503).json({ ok: false, error: "RPC not configured" });
    }
    const { address, abi, functionName, args } = req.body || {};
    if (!address || !abi || !functionName) {
      return res.status(400).json({
        ok: false,
        error: "body requires address, abi, functionName",
      });
    }
    const result = await callContract({
      address,
      abi,
      functionName,
      args: args || [],
    });
    res.json({ ok: true, result, via: "ethers" });
  } catch (e) {
    res.status(400).json({ ok: false, error: String(e.message || e) });
  }
});

/** HIM agent meta — no secrets */
app.get("/api/him", (_req, res) => {
  res.json({
    ok: true,
    name: "HIM",
    architecture: "full-stack",
    frontend: "React → /api/* only",
    backend: "Express + ethers + viem",
    security: "API keys in server .env only",
    install: "npm run install:applet -- <package>",
  });
});

// Serve built React client if present
const clientDist = path.resolve(__dirname, "../client/dist");
app.use(express.static(clientDist));
app.get("*", (req, res, next) => {
  if (req.path.startsWith("/api")) return next();
  const index = path.join(clientDist, "index.html");
  res.sendFile(index, (err) => {
    if (err) {
      res
        .status(200)
        .type("html")
        .send(
          `<!doctype html><html><body style="font-family:system-ui;background:#0b0f14;color:#e8eefc;padding:2rem">
          <h1>HIM web3 API</h1>
          <p>Server is up. Build the React client: <code>cd client && npm install && npm run build</code></p>
          <p>API: <a href="/api/health" style="color:#6cf">/api/health</a> · <a href="/api/him" style="color:#6cf">/api/him</a></p>
          </body></html>`
        );
    }
  });
});

app.listen(PORT, HOST, () => {
  console.log(
    JSON.stringify({
      ok: true,
      service: "him-web3",
      listen: `http://${HOST}:${PORT}`,
      rpc_configured: hasRpc(),
      name: "HIM",
    })
  );
});
