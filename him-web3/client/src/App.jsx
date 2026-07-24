import React, { useEffect, useState } from "react";

/**
 * HIM React UI — talks ONLY to /api/* (never Alchemy/Infura directly).
 * Secrets stay on the Express server.
 */
async function api(path, opts) {
  const r = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  return r.json();
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [blockNumber, setBlockNumber] = useState(null);
  const [address, setAddress] = useState(
    "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
  ); // vitalik.eth example
  const [balance, setBalance] = useState(null);
  const [block, setBlock] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api("/api/health")
      .then(setHealth)
      .catch((e) => setError(String(e)));
    api("/api/him").catch(() => {});
  }, []);

  async function loadBlockNumber() {
    setBusy(true);
    setError("");
    try {
      const j = await api("/api/chain/block-number");
      if (!j.ok) throw new Error(j.error || j.hint || "failed");
      setBlockNumber(j.blockNumber);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  }

  async function loadBalance() {
    setBusy(true);
    setError("");
    try {
      const j = await api(`/api/chain/balance/${address}`);
      if (!j.ok) throw new Error(j.error || "failed");
      setBalance(j);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  }

  async function loadBlock() {
    setBusy(true);
    setError("");
    try {
      const j = await api("/api/chain/block?tag=latest");
      if (!j.ok) throw new Error(j.error || "failed");
      setBlock(j.block);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  }

  const rpcOk = health?.rpc_configured;

  return (
    <>
      <header>
        <h1>HIM · Web3 Applet</h1>
        <div>
          <span className={`badge ${rpcOk ? "ok" : ""}`}>
            {rpcOk ? "RPC configured (server)" : "RPC missing — set .env"}
          </span>
          <span className="badge">React → /api/* only</span>
        </div>
      </header>
      <main>
        <section className="card">
          <h2>Architecture</h2>
          <p className="note">
            HIM uses a full-stack pattern: this React UI never holds Alchemy/Infura
            keys. All chain calls go to secure Express routes that use{" "}
            <code>ethers</code> / <code>viem</code> server-side. Install packages
            with <code>npm run install:applet -- ethers viem</code>.
          </p>
          <pre>{JSON.stringify(health, null, 2)}</pre>
        </section>

        <section className="card">
          <h2>On-chain reads (via secure API)</h2>
          <div className="row">
            <button disabled={busy} onClick={loadBlockNumber}>
              Block number (viem)
            </button>
            <button className="secondary" disabled={busy} onClick={loadBlock}>
              Latest block (ethers)
            </button>
          </div>
          {blockNumber && (
            <p className="note">
              Block: <strong>{blockNumber}</strong>
            </p>
          )}
          {block && <pre>{JSON.stringify(block, null, 2)}</pre>}
        </section>

        <section className="card">
          <h2>Balance lookup</h2>
          <input
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            placeholder="0x address"
          />
          <button disabled={busy} onClick={loadBalance}>
            Get balance
          </button>
          {balance && <pre>{JSON.stringify(balance, null, 2)}</pre>}
        </section>

        {error && (
          <section className="card">
            <h2>Error</h2>
            <pre style={{ color: "var(--warn)" }}>{error}</pre>
            <p className="note">
              If RPC is missing: copy <code>him-web3/.env.example</code> to{" "}
              <code>.env</code> and set <code>ETH_RPC_URL</code> (Alchemy or Infura).
            </p>
          </section>
        )}
      </main>
    </>
  );
}
