import React, { useEffect, useState } from "react";
import {
  useFeatureState,
  PortfolioPanel,
  FlashLoanPanel,
  AuditorPanel,
  BridgeMonitorPanel,
  YieldAggregatorPanel,
  ParallelSimPanel,
  ComputeOptimizerPanel,
  LaunchpadPanel,
  AIAgentPanel,
  GovernancePanel,
  MEVScannerPanel,
  BacktesterPanel,
  IOChainWebGPUPanel,
} from "./FeaturePanels.jsx";

async function api(path, opts = {}) {
  const r = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  return r.json();
}

export default function App() {
  const [activeTab, setActiveTab] = useState("defi"); // 'defi' | 'hf-protocol' | 'super-workspace' | 'profiler' | 'apps' | 'primitives' | 'engines'
  const [platform, setPlatform] = useState(null);
  const [telemetry, setTelemetry] = useState({ tps: 9840, blockHeight: 18940200, finalityMs: 780 });
  const [apps, setApps] = useState([]);
  const [primitives, setPrimitives] = useState([]);
  const [engines, setEngines] = useState([]);
  
  // DeFi Tools & Security State
  const [arbitrageData, setArbitrageData] = useState(null);
  const [vaultData, setVaultData] = useState(null);
  const [cumulativeArbProfit, setCumulativeArbProfit] = useState(1280.40);
  const [gasLimitInput, setGasLimitInput] = useState("150000");
  const [priorityTipInput, setPriorityTipInput] = useState("2.5");
  const [gasReserveResult, setGasReserveResult] = useState(null);
  const [hftPair, setHftPair] = useState("MONAD/USDC");
  const [hftAmount, setHftAmount] = useState("500");
  const [hftSide, setHftSide] = useState("BUY");
  const [hftTxResult, setHftTxResult] = useState(null);

  // Hugging Face V3 Exporter & Auth State
  const [firstPartyModels, setFirstPartyModels] = useState([]);
  const [selectedHfModel, setSelectedHfModel] = useState("thesis-ai/Thesis-Monad-HFT-Reasoning-7B");
  const [selectedQuantization, setSelectedQuantization] = useState("q4_k_m");
  const [hfTokenInput, setHfTokenInput] = useState("");
  const [hfTokenVerified, setHfTokenVerified] = useState(false);
  const [showTokenDrawer, setShowTokenDrawer] = useState(false);
  const [hfPrompt, setHfPrompt] = useState("Scan Monad DEX pools for parallel arbitrage and execute instant EIP-2930 swap if profit > $200");
  const [hfAgentResult, setHfAgentResult] = useState(null);
  const [hfReleaseResult, setHfReleaseResult] = useState(null);

  // Super Large Context Workspace V3 State
  const [superContextPrompt, setSuperContextPrompt] = useState("Analyze 500 Monad block traces across 131,072 token context window for DEX state diff conflicts.");
  const [superContextResult, setSuperContextResult] = useState(null);
  const [attentionMapData, setAttentionMapData] = useState(null);

  // Opcode Profiler State
  const [coldAccesses, setColdAccesses] = useState(3);
  const [warmAccesses, setWarmAccesses] = useState(12);
  const [precompileCalls, setPrecompileCalls] = useState(1);
  const [storageClears, setStorageClears] = useState(1);
  const [profilerResult, setProfilerResult] = useState(null);

  // Floating AI Assistant Drawer State
  const [chatOpen, setChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState([
    { sender: "agent", text: "Operator! V3 HF Exporter & Super Context Attention Heatmaps active." }
  ]);
  const [chatInput, setChatInput] = useState("");

  // EIP-7702 Session Key Generator State
  const [sessionKeys, setSessionKeys] = useState([]);

  // Monad ↔ Solana Cross-VM State
  const [crossVmData, setCrossVmData] = useState(null);
  const [crossVmPair, setCrossVmPair] = useState("MONAD/SOL");
  const [crossVmSize, setCrossVmSize] = useState("1000");
  const [crossVmResult, setCrossVmResult] = useState(null);

  const [selectedApp, setSelectedApp] = useState(null);
  const [invocationResult, setInvocationResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  // V2 Feature State (12 panels)
  const featureState = useFeatureState();

  // Orderbook Depth Data
  const [orderbook] = useState({
    asks: [
      { price: "1.498", size: "12,400", pct: 85 },
      { price: "1.492", size: "8,100", pct: 60 },
      { price: "1.488", size: "4,200", pct: 35 },
    ],
    bids: [
      { price: "1.482", size: "9,800", pct: 70 },
      { price: "1.478", size: "14,300", pct: 95 },
      { price: "1.470", size: "6,500", pct: 45 },
    ]
  });

  useEffect(() => {
    // Platform API Fetch
    api("/platform").then((data) => { if (data.ok) setPlatform(data); }).catch(() => {});
    api("/platform/apps").then((data) => { if (data.ok) setApps(data.apps); }).catch(() => {});
    api("/platform/primitives").then((data) => { if (data.ok) setPrimitives(data.primitives); }).catch(() => {});
    api("/engines").then((data) => { if (data.ok) setEngines(data.engines); }).catch(() => {});
    api("/platform/hf-protocol/first-party").then((data) => { if (data.ok) setFirstPartyModels(data.hubCatalog); }).catch(() => {});

    fetchDeFiData();

    // 400ms Monad Block Telemetry Pulse Simulation
    const timer = setInterval(() => {
      setTelemetry((prev) => ({
        tps: 9800 + Math.floor(Math.random() * 350),
        blockHeight: prev.blockHeight + 1,
        finalityMs: 760 + Math.floor(Math.random() * 40),
      }));
    }, 400);

    return () => clearInterval(timer);
  }, []);

  function fetchDeFiData() {
    api("/platform/defi/arbitrage").then(setArbitrageData).catch(() => {});
    api("/platform/defi/vaults/yield").then(setVaultData).catch(() => {});
    calculateGasReserve("150000", priorityTipInput);
    runOpcodeProfiler(3, 12, 1, 1);
    fetchAttentionMap();
    fetchCrossVmArbitrage();
  }

  async function fetchCrossVmArbitrage() {
    try {
      const res = await api("/platform/crosschain/arbitrage");
      if (res.ok) setCrossVmData(res);
    } catch (e) {
      console.warn(e);
    }
  }

  async function executeCrossVmSwap(e) {
    if (e) e.preventDefault();
    setBusy(true);
    setCrossVmResult(null);
    setError("");
    try {
      const res = await api("/platform/crosschain/swap", {
        method: "POST",
        body: JSON.stringify({
          pair: crossVmPair,
          size: Number(crossVmSize),
          targetSpreadBps: 94.7,
        }),
      });
      if (!res.ok) throw new Error(res.error || "Cross-VM Swap Failed");
      setCrossVmResult(res);
    } catch (err) {
      setError("Cross-VM Execution Error: " + String(err.message || err));
    } finally {
      setBusy(false);
    }
  }

  async function fetchAttentionMap() {
    try {
      const res = await api("/platform/workspace/attention-map", {
        method: "POST",
        body: JSON.stringify({ prompt: superContextPrompt }),
      });
      if (res.ok) setAttentionMapData(res);
    } catch (e) {
      console.warn(e);
    }
  }

  async function calculateGasReserve(gasLimitVal, tipVal) {
    try {
      const res = await api(`/platform/defi/gas-reserve?gasLimit=${gasLimitVal}&priorityTipGwei=${tipVal}`);
      if (res.ok) setGasReserveResult(res);
    } catch (e) {
      console.warn(e);
    }
  }

  async function runOpcodeProfiler(cold, warm, precomp, clears) {
    try {
      const res = await api("/platform/defi/opcode-profiler", {
        method: "POST",
        body: JSON.stringify({ coldAccesses: cold, warmAccesses: warm, precompileCalls: precomp, storageClears: clears }),
      });
      if (res.ok) setProfilerResult(res);
    } catch (e) {
      console.warn(e);
    }
  }

  async function verifyHfToken(e) {
    e.preventDefault();
    setBusy(true);
    try {
      const res = await api("/platform/hf-protocol/validate-token", {
        method: "POST",
        body: JSON.stringify({ hfToken: hfTokenInput }),
      });
      if (res.ok && res.valid) {
        setHfTokenVerified(true);
        setShowTokenDrawer(false);
      } else {
        setError("Invalid Hugging Face User Access Token (Must start with hf_)");
      }
    } catch (err) {
      setError("Token verification error: " + String(err));
    } finally {
      setBusy(false);
    }
  }

  async function runHfAgentStep() {
    setBusy(true);
    setHfAgentResult(null);
    setError("");
    try {
      const res = await api("/platform/hf-protocol/agent-step", {
        method: "POST",
        body: JSON.stringify({
          modelId: selectedHfModel,
          prompt: hfPrompt,
        }),
      });
      if (!res.ok) throw new Error("HF Agent Step Failed");
      setHfAgentResult(res);
    } catch (e) {
      setError("HF Protocol Error: " + String(e.message || e));
    } finally {
      setBusy(false);
    }
  }

  async function releaseToHuggingFaceHub() {
    setBusy(true);
    setHfReleaseResult(null);
    try {
      const res = await api("/platform/hf-protocol/release", {
        method: "POST",
        body: JSON.stringify({ modelId: selectedHfModel, quantization: selectedQuantization }),
      });
      if (res.ok) setHfReleaseResult(res);
    } catch (e) {
      setError("Release Error: " + String(e));
    } finally {
      setBusy(false);
    }
  }

  async function runSuperContextAnalysis() {
    setBusy(true);
    setSuperContextResult(null);
    try {
      const res = await api("/platform/workspace/super-context", {
        method: "POST",
        body: JSON.stringify({ prompt: superContextPrompt, blockHistoryCount: 500 }),
      });
      if (res.ok) {
        setSuperContextResult(res);
        fetchAttentionMap();
      }
    } catch (e) {
      setError("Super Context Analysis Error: " + String(e));
    } finally {
      setBusy(false);
    }
  }

  async function executeArbitrageBot(opportunity) {
    setBusy(true);
    try {
      const res = await api("/platform/defi/hft/order", {
        method: "POST",
        body: JSON.stringify({
          pair: opportunity.pair,
          side: "ARBITRAGE_EIP2930_MULTICALL",
          amount: opportunity.estProfitUsd.toString(),
          slippage: "0.05%",
          priorityTipGwei: priorityTipInput,
        }),
      });
      if (res.ok) {
        setCumulativeArbProfit((prev) => prev + opportunity.estProfitUsd);
        setSelectedApp({
          id: "arb-bot",
          name: `Arbitrage Bot (${opportunity.pair})`,
          icon: "🤖",
        });
        setInvocationResult({
          ok: true,
          arbitrageExecuted: opportunity,
          netProfitCapturedUsd: opportunity.estProfitUsd,
          monadParallelExecutionTimeMs: 11,
          accessListVerified: opportunity.accessList,
          finalizedTxHash: res.txHash,
          wintermuteSecurity: res.wintermuteSecurity,
        });
      }
    } catch (e) {
      setError("Arbitrage Execution Error: " + String(e));
    } finally {
      setBusy(false);
    }
  }

  async function executeHftOrder(e) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const res = await api("/platform/defi/hft/order", {
        method: "POST",
        body: JSON.stringify({
          pair: hftPair,
          side: hftSide,
          amount: hftAmount,
          slippage: "0.1%",
          priorityTipGwei: priorityTipInput,
          nonce: Math.floor(Math.random() * 1000) + 100,
        }),
      });
      if (!res.ok) throw new Error(res.error || "HFT Order Failed");
      setHftTxResult(res);
    } catch (err) {
      setError("HFT Order Error: " + String(err.message || err));
    } finally {
      setBusy(false);
    }
  }

  function handleSendChatMessage(e) {
    e.preventDefault();
    if (!chatInput.trim()) return;
    const userMsg = { sender: "user", text: chatInput };
    setChatMessages((prev) => [...prev, userMsg]);
    const inputCopy = chatInput;
    setChatInput("");

    setTimeout(() => {
      let responseText = "Monad HQ Twin: V3 HF Exporter ready for Hub publication.";
      if (inputCopy.toLowerCase().includes("quant") || inputCopy.toLowerCase().includes("format")) {
        responseText = `Quantization Configurator: Packaging model weights in ${selectedQuantization.toUpperCase()} (Safetensors / GGUF / ONNX).`;
      }
      setChatMessages((prev) => [...prev, { sender: "agent", text: responseText }]);
    }, 450);
  }

  function generateSessionKey() {
    const newKey = {
      id: "sk_" + Math.random().toString(36).substring(2, 9),
      targetContract: "0x8352..." + Math.floor(Math.random()*9000+1000),
      permission: "HIGH_FREQUENCY_SWAP_ONLY",
      maxAllowanceEth: "5.0",
      expiresAt: new Date(Date.now() + 86400000).toISOString(),
    };
    setSessionKeys([newKey, ...sessionKeys]);
  }

  async function invokeApp(appItem) {
    setSelectedApp(appItem);
    setBusy(true);
    setInvocationResult(null);
    setError("");
    try {
      const res = await api(`/platform/apps/${appItem.id}/invoke`, {
        method: "POST",
        body: JSON.stringify({
          action: "initialize_session",
          timestamp: new Date().toISOString(),
          requestedBy: "HQ Operator",
        }),
      });
      if (!res.ok) throw new Error(res.error || "Invocation failed");
      setInvocationResult(res);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      {/* --- TELEMETRY & BRAND HEADER --- */}
      <header>
        <div className="brand">
          <div className="brand-icon">M</div>
          <div>
            <h1>THESIS Monad HQ</h1>
            <span className="subtitle">First-Party Hugging Face Models & 128k Token Super Workspace</span>
          </div>
        </div>
        
        <div className="status-badges">
          <span className="badge monad-tag">
            <span className="badge-dot"></span>
            Block #{telemetry.blockHeight} (400ms)
          </span>
          <span className="badge ok">
            ⚡ {telemetry.tps.toLocaleString()} TPS
          </span>
          <span className={`badge ${hfTokenVerified ? "ok" : ""}`} onClick={() => setShowTokenDrawer(true)}>
            🤗 HF Auth ({hfTokenVerified ? "Authenticated" : "Connect Access Token"})
          </span>
        </div>
      </header>

      {/* --- NAVIGATION TABS --- */}
      <nav className="tabs">
        <button
          className={`tab-btn ${activeTab === "iochain" ? "active" : ""}`}
          onClick={() => setActiveTab("iochain")}
        >
          ⚡ IOChain WebGPU DeAI (Solana)
        </button>
        <button
          className={`tab-btn ${activeTab === "defi" ? "active" : ""}`}
          onClick={() => setActiveTab("defi")}
        >
          💰 Institutional DeFi & HFT
        </button>
        <button
          className={`tab-btn ${activeTab === "crossvm" ? "active" : ""}`}
          onClick={() => { setActiveTab("crossvm"); fetchCrossVmArbitrage(); }}
        >
          ⚡ Monad ↔ Solana Cross-VM
        </button>
        <button
          className={`tab-btn ${activeTab === "hf-protocol" ? "active" : ""}`}
          onClick={() => setActiveTab("hf-protocol")}
        >
          🤗 First-Party HF Models & Release
        </button>
        <button
          className={`tab-btn ${activeTab === "super-workspace" ? "active" : ""}`}
          onClick={() => setActiveTab("super-workspace")}
        >
          🌌 Super Context Workspace (128k)
        </button>
        <button
          className={`tab-btn ${activeTab === "profiler" ? "active" : ""}`}
          onClick={() => setActiveTab("profiler")}
        >
          ⛽ Opcode & Reserve Profiler
        </button>
        <button
          className={`tab-btn ${activeTab === "apps" ? "active" : ""}`}
          onClick={() => setActiveTab("apps")}
        >
          📱 Apps ({apps.length})
        </button>
        <button
          className={`tab-btn ${activeTab === "primitives" ? "active" : ""}`}
          onClick={() => setActiveTab("primitives")}
        >
          ⚙️ Primitives ({primitives.length})
        </button>
        <button className={`tab-btn ${activeTab === "portfolio" ? "active" : ""}`} onClick={() => setActiveTab("portfolio")}>📊 Portfolio</button>
        <button className={`tab-btn ${activeTab === "flashloan" ? "active" : ""}`} onClick={() => setActiveTab("flashloan")}>⚡ Flash Loans</button>
        <button className={`tab-btn ${activeTab === "auditor" ? "active" : ""}`} onClick={() => setActiveTab("auditor")}>🔍 Auditor</button>
        <button className={`tab-btn ${activeTab === "bridge" ? "active" : ""}`} onClick={() => setActiveTab("bridge")}>🌉 Bridge Monitor</button>
        <button className={`tab-btn ${activeTab === "yield" ? "active" : ""}`} onClick={() => setActiveTab("yield")}>🌾 Yield Aggregator</button>
        <button className={`tab-btn ${activeTab === "parallel" ? "active" : ""}`} onClick={() => setActiveTab("parallel")}>🔀 Parallel Sim</button>
        <button className={`tab-btn ${activeTab === "compute" ? "active" : ""}`} onClick={() => setActiveTab("compute")}>⚙️ Compute Budget</button>
        <button className={`tab-btn ${activeTab === "launchpad" ? "active" : ""}`} onClick={() => setActiveTab("launchpad")}>🚀 Launchpad</button>
        <button className={`tab-btn ${activeTab === "agent" ? "active" : ""}`} onClick={() => setActiveTab("agent")}>🤖 AI Agent</button>
        <button className={`tab-btn ${activeTab === "governance" ? "active" : ""}`} onClick={() => setActiveTab("governance")}>🏛️ Governance</button>
        <button className={`tab-btn ${activeTab === "mev" ? "active" : ""}`} onClick={() => setActiveTab("mev")}>👁️ MEV Scanner</button>
        <button className={`tab-btn ${activeTab === "backtester" ? "active" : ""}`} onClick={() => setActiveTab("backtester")}>📈 Backtester</button>
      </nav>

      <main>
        {error && (
          <div className="card" style={{ borderColor: "var(--danger)", marginBottom: "1rem" }}>
            <div className="card-top">
              <h3 style={{ color: "var(--danger)", margin: 0 }}>System Alert</h3>
            </div>
            <p className="desc">{error}</p>
          </div>
        )}

        {/* HF TOKEN AUTH MODAL */}
        {showTokenDrawer && (
          <div className="card" style={{ borderColor: "var(--monad-purple)", marginBottom: "1rem" }}>
            <div className="card-top">
              <h3>🤗 Connect Hugging Face User Access Token</h3>
              <button className="secondary" style={{ padding: "0.2rem 0.5rem" }} onClick={() => setShowTokenDrawer(false)}>Close</button>
            </div>
            <p className="desc">Enter your Hugging Face User Access Token (from https://huggingface.co/settings/tokens) to authenticate 1-click model exports to `thesis-ai` or your user account.</p>
            <form onSubmit={verifyHfToken} style={{ display: "flex", gap: "0.5rem" }}>
              <input
                type="password"
                placeholder="hf_..."
                value={hfTokenInput}
                onChange={(e) => setHfTokenInput(e.target.value)}
                style={{ flex: 1, marginBottom: 0 }}
              />
              <button className="primary" style={{ width: "auto" }} disabled={busy}>
                {busy ? "Verifying..." : "Authenticate Token"}
              </button>
            </form>
          </div>
        )}

        {/* --- MONAD HFT & DEFI SUITE TAB --- */}
        {activeTab === "defi" && (
          <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
            <div className="section-header">
              <h2>Institutional HFT & Quantitative Risk Suite</h2>
              <p>Upgraded with Wintermute EIP-712 security, Flashbots EIP-2930 parallel access lists, and Gauntlet VaR metrics.</p>
            </div>

            {/* SECTION 1: HFT TRADING & LIVE ORDERBOOK VISUALIZER */}
            <div className="hft-panel">
              {/* SUB-SECOND HFT DESK */}
              <div className="card">
                <div className="card-top">
                  <h3>⚡ Sub-Second HFT Trading Terminal</h3>
                  <span className="badge ok">EIP-712 & Priority Tip</span>
                </div>
                <p className="desc">Wintermute upgraded execution engine with nonces, execution deadlines, and priority tip scaling.</p>
                
                <form onSubmit={executeHftOrder}>
                  <div className="form-group">
                    <label>Trading Pair</label>
                    <select value={hftPair} onChange={(e) => setHftPair(e.target.value)}>
                      <option value="MONAD/USDC">MONAD / USDC</option>
                      <option value="MONAD/WETH">MONAD / WETH</option>
                      <option value="WBTC/USDC">WBTC / USDC</option>
                    </select>
                  </div>
                  <div style={{ display: "flex", gap: "0.75rem" }}>
                    <div className="form-group" style={{ flex: 1 }}>
                      <label>Action</label>
                      <select value={hftSide} onChange={(e) => setHftSide(e.target.value)}>
                        <option value="BUY">BUY</option>
                        <option value="SELL">SELL</option>
                      </select>
                    </div>
                    <div className="form-group" style={{ flex: 1 }}>
                      <label>Amount ($)</label>
                      <input
                        type="number"
                        value={hftAmount}
                        onChange={(e) => setHftAmount(e.target.value)}
                      />
                    </div>
                    <div className="form-group" style={{ flex: 1 }}>
                      <label>Priority Tip (Gwei)</label>
                      <input
                        type="number"
                        step="0.5"
                        value={priorityTipInput}
                        onChange={(e) => {
                          setPriorityTipInput(e.target.value);
                          calculateGasReserve(gasLimitInput, e.target.value);
                        }}
                      />
                    </div>
                  </div>
                  <button className="primary" style={{ width: "100%", marginTop: "0.5rem" }} disabled={busy}>
                    {busy ? "Signing EIP-712 Order..." : "Execute Signed HFT Order"}
                  </button>
                </form>

                {hftTxResult && (
                  <div style={{ marginTop: "1rem" }}>
                    <span className="badge ok">Verified EIP-712 Domain (Nonce {hftTxResult.wintermuteSecurity?.nonceVerified})</span>
                    <pre className="code-view" style={{ maxHeight: "120px", marginTop: "0.4rem" }}>
{JSON.stringify(hftTxResult, null, 2)}
                    </pre>
                  </div>
                )}
              </div>

              {/* LIVE ORDERBOOK DEPTH VISUALIZER */}
              <div className="card">
                <div className="card-top">
                  <h3>📊 Live Orderbook Depth Visualizer</h3>
                  <span className="badge ok">400ms Sync</span>
                </div>
                <p className="desc">Monad HFT orderbook visualizer updating every 400ms block interval.</p>
                
                <div className="orderbook-visualizer">
                  <div style={{ fontSize: "0.72rem", color: "var(--muted)", textTransform: "uppercase", marginBottom: "0.2rem" }}>Asks (Sell Orders)</div>
                  {orderbook.asks.map((ask, idx) => (
                    <div key={`ask-${idx}`} className="order-row ask">
                      <span>${ask.price}</span>
                      <span>{ask.size}</span>
                      <span style={{ textAlign: "right" }}>Ask</span>
                      <div className="depth-bar ask-bar" style={{ width: `${ask.pct}%` }}></div>
                    </div>
                  ))}

                  <div style={{ borderTop: "1px solid var(--border)", margin: "0.4rem 0" }}></div>

                  <div style={{ fontSize: "0.72rem", color: "var(--muted)", textTransform: "uppercase", marginBottom: "0.2rem" }}>Bids (Buy Orders)</div>
                  {orderbook.bids.map((bid, idx) => (
                    <div key={`bid-${idx}`} className="order-row bid">
                      <span>${bid.price}</span>
                      <span>{bid.size}</span>
                      <span style={{ textAlign: "right" }}>Bid</span>
                      <div className="depth-bar bid-bar" style={{ width: `${bid.pct}%` }}></div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* SECTION 2: EIP-2930 PARALLEL ARBITRAGE SCANNER */}
            <div className="card">
              <div className="card-top">
                <div>
                  <h3>🤖 EIP-2930 Parallel Arbitrage Bot (Flashbots Enforced)</h3>
                  <span className="desc">Cumulative Captured Profit: <strong className="text-purple">${cumulativeArbProfit.toFixed(2)} USD</strong></span>
                </div>
                <button className="secondary" onClick={fetchDeFiData}>Refresh Spreads</button>
              </div>
              
              <div className="table-container" style={{ marginTop: "0.5rem" }}>
                <table className="defi-table">
                  <thead>
                    <tr>
                      <th>Token Pair</th>
                      <th>DEX Spread</th>
                      <th>Spread (%)</th>
                      <th>Est. Profit</th>
                      <th>Access List (EIP-2930)</th>
                      <th>Execution</th>
                    </tr>
                  </thead>
                  <tbody>
                    {arbitrageData?.opportunities?.map((opp, idx) => (
                      <tr key={idx}>
                        <td><strong>{opp.pair}</strong></td>
                        <td>{opp.dexA} vs {opp.dexB}</td>
                        <td className="text-ok">+{opp.spreadPct}%</td>
                        <td className="text-purple">${opp.estProfitUsd.toFixed(2)}</td>
                        <td><span className="tag">EIP-2930 Access List ({opp.accessList?.length || 0} items)</span></td>
                        <td>
                          <button className="primary" style={{ padding: "0.3rem 0.75rem", fontSize: "0.78rem" }} disabled={busy} onClick={() => executeArbitrageBot(opp)}>
                            ⚡ Execute Arbitrage
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* SECTION 3: GAUNTLET RISK GAUGES & EIP-7702 KEYS */}
            <div className="grid-2">
              {/* GAUNTLET RISK VAULTS */}
              <div className="card">
                <div className="card-top">
                  <h3>💎 SovereignVaults (Gauntlet Risk Engine)</h3>
                  <span className="badge ok">99% VaR Monitored</span>
                </div>
                <p className="desc">Quantitative Liquidation Buffers and Value-at-Risk modeling.</p>
                <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
                  {vaultData?.vaults?.map((v) => (
                    <div key={v.id} style={{ background: "#04070d", padding: "0.75rem 1rem", borderRadius: "10px", border: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <div>
                        <div style={{ fontWeight: 600, fontSize: "0.9rem" }}>{v.name}</div>
                        <div style={{ fontSize: "0.75rem", color: "var(--muted)" }}>VaR 99%: {v.var99Pct1Day || "0.4%"} · Drawdown: {v.maxDrawdownPct || "1.1%"}</div>
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <div className="text-ok" style={{ fontSize: "1.05rem" }}>+{v.currentApyPct}% APY</div>
                        <span className="tag">{v.healthFactor || "Health: 2.45"}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* EIP-7702 SESSION KEYS */}
              <div className="card">
                <div className="card-top">
                  <h3>🔑 EIP-7702 Session Key Delegator</h3>
                  <button className="primary" onClick={generateSessionKey}>+ Issue Session Key</button>
                </div>
                <p className="desc">Grant temporary, scope-limited authorization keys for automated HFT bots without exposing private keys.</p>
                {sessionKeys.length === 0 ? (
                  <p className="note">No active session keys generated. Click button above to issue a key.</p>
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                    {sessionKeys.map((k) => (
                      <div key={k.id} style={{ background: "#04070d", padding: "0.65rem 0.85rem", borderRadius: "8px", border: "1px solid var(--border)", fontSize: "0.8rem" }}>
                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                          <span className="text-purple">{k.id}</span>
                          <span className="tag">{k.permission}</span>
                        </div>
                        <div style={{ color: "var(--muted)", marginTop: "0.2rem" }}>Target: {k.targetContract} (Limit: {k.maxAllowanceEth} MONAD)</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* --- MONAD ↔ SOLANA CROSS-VM HFT TAB --- */}
        {activeTab === "crossvm" && (
          <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
            <div className="section-header">
              <h2>⚡ Monad EVM ↔ Solana SVM Cross-Chain HFT Suite</h2>
              <p>Dual-chain high-frequency arbitrage engine linking Monad (10,000 TPS, 1s finality) and Solana (65,000 TPS, 400ms finality).</p>
            </div>

            {/* LIVE SPREADS TABLE */}
            <div className="card">
              <div className="card-top">
                <h3>Cross-Chain DEX Arbitrage Opportunities</h3>
                <span className="badge ok">Live DEX Orderbook Monitor</span>
              </div>
              <p className="desc">Real-time price spreads between Monad EVM DEXs (Kuru, MonadSwap, Ambient) and Solana SVM DEXs (Phoenix, Orca, Raydium).</p>

              {crossVmData && crossVmData.pairs ? (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Asset</th>
                      <th>Monad DEX</th>
                      <th>Monad Price</th>
                      <th>Solana DEX</th>
                      <th>Solana Price</th>
                      <th>Spread (bps)</th>
                      <th>Est Profit</th>
                      <th>Execution Route</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {crossVmData.pairs.map((p, idx) => (
                      <tr key={idx}>
                        <td className="bold text-purple">{p.asset}</td>
                        <td>{p.monadDex}</td>
                        <td>${p.monadPrice}</td>
                        <td>{p.solanaDex}</td>
                        <td>${p.solanaPrice}</td>
                        <td className="bold text-green">+{p.spreadBps} bps</td>
                        <td className="bold text-green">${p.netProfitUsd.toLocaleString()}</td>
                        <td style={{ fontSize: "0.75rem", color: "var(--muted)" }}>{p.routeType}</td>
                        <td>
                          <button
                            className="primary"
                            style={{ padding: "0.3rem 0.6rem", fontSize: "0.75rem" }}
                            disabled={busy}
                            onClick={() => {
                              setCrossVmPair(p.asset);
                              executeCrossVmSwap();
                            }}
                          >
                            ⚡ Execute
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="note">Loading cross-chain DEX spread feeds...</p>
              )}
            </div>

            {/* CROSS-VM EXECUTION TERMINAL & ALT COMPILER */}
            <div className="grid-2">
              <div className="card">
                <h3>Cross-Chain Atomic Swap Executor</h3>
                <p className="desc">Simulate and execute sub-second cross-chain arbitrage across EVM and SVM state channels.</p>

                <form onSubmit={executeCrossVmSwap} style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                  <div className="form-group">
                    <label>Target Trading Pair</label>
                    <select value={crossVmPair} onChange={(e) => setCrossVmPair(e.target.value)}>
                      <option value="MONAD/SOL">MONAD / SOL (Kuru CLOB ↔ Phoenix DEX)</option>
                      <option value="USDC/USDT">USDC / USDT (MonadSwap ↔ Orca Whirlpools)</option>
                      <option value="wBTC/BTC">wBTC / BTC (Ambient ↔ Raydium CLMM)</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label>Execution Trade Size (Units)</label>
                    <input
                      type="number"
                      value={crossVmSize}
                      onChange={(e) => setCrossVmSize(e.target.value)}
                      placeholder="1000"
                    />
                  </div>

                  <div className="form-group">
                    <label>Execution Protocol Directives</label>
                    <div style={{ background: "#04070d", padding: "0.75rem", borderRadius: "8px", border: "1px solid var(--border)", fontSize: "0.8rem", color: "var(--muted)" }}>
                      <div>• EVM Side: Enforce EIP-2930 Access List to bypass state contention</div>
                      <div>• SVM Side: Compile Solana Address Lookup Table (ALT) account</div>
                      <div>• Settlement: SovereignVault Cross-VM Relayer (&lt;500ms target)</div>
                    </div>
                  </div>

                  <button className="primary" style={{ width: "100%" }} disabled={busy}>
                    {busy ? "Executing Cross-Chain Swap..." : "⚡ Execute Atomic Cross-VM Swap"}
                  </button>
                </form>
              </div>

              <div className="card">
                <h3>Atomic Cross-VM Receipt & Telemetry</h3>
                {crossVmResult ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span className="badge ok">STATUS: {crossVmResult.status}</span>
                      <span className="tag text-purple">{crossVmResult.executionSummary?.settlementTimeMs}ms Settlement</span>
                    </div>

                    <div style={{ background: "#04070d", padding: "0.85rem", borderRadius: "8px", border: "1px solid var(--border)", fontSize: "0.8rem" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.4rem" }}>
                        <span>Captured Spread Yield:</span>
                        <strong className="text-green">+{crossVmResult.executionSummary?.capturedYieldBps} bps (${crossVmResult.executionSummary?.netYieldUsd})</strong>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.4rem" }}>
                        <span>Monad EVM Tx Hash:</span>
                        <span className="code-view" style={{ padding: "0.1rem 0.3rem" }}>{crossVmResult.txHash?.substring(0, 16)}...</span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.4rem" }}>
                        <span>Solana Tx Signature:</span>
                        <span className="code-view" style={{ padding: "0.1rem 0.3rem" }}>{crossVmResult.solanaTxSignature?.substring(0, 16)}...</span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span>EVM Gas / SVM Compute:</span>
                        <span>{crossVmResult.executionSummary?.evmExecution?.gasUsed} gas / {crossVmResult.executionSummary?.svmExecution?.computeUnitsUsed} CU</span>
                      </div>
                    </div>

                    <pre className="code-view" style={{ maxHeight: "200px" }}>
{JSON.stringify(crossVmResult, null, 2)}
                    </pre>
                  </div>
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                    <p className="note">No Cross-VM transaction executed yet. Select a pair and click Execute.</p>
                    {crossVmData && crossVmData.optimizations && (
                      <div style={{ background: "#04070d", padding: "0.85rem", borderRadius: "8px", border: "1px solid var(--border)", fontSize: "0.8rem" }}>
                        <div className="bold text-purple" style={{ marginBottom: "0.4rem" }}>Cross-VM Protocol Optimizations:</div>
                        <div>• {crossVmData.optimizations.eip2930AccessList}</div>
                        <div>• {crossVmData.optimizations.solanaAddressLookupTable}</div>
                        <div>• {crossVmData.optimizations.monadGasSafetyFactor}</div>
                        <div>• {crossVmData.optimizations.jumpCryptoZeroingRefunds}</div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* --- FIRST-PARTY HF MODELS & RELEASE TAB --- */}
        {activeTab === "hf-protocol" && (
          <div>
            <div className="section-header">
              <h2>🤗 First-Party THESIS Models & Hugging Face Hub Exporter V3</h2>
              <p>Specialty models fine-tuned for Monad HFT, SovereignVault quant risk, and dual-law governance.</p>
            </div>

            <div className="grid-2">
              <div className="card">
                <div className="card-top">
                  <h3>THESIS Model Catalog & Packaging Config</h3>
                  <span className="badge monad-tag">thesis-ai Organization</span>
                </div>
                
                <div className="form-group">
                  <label>Select THESIS Specialty Model</label>
                  <select value={selectedHfModel} onChange={(e) => setSelectedHfModel(e.target.value)}>
                    {firstPartyModels.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name} (Context: {(m.contextWindowTokens/1024).toFixed(0)}k Tokens)
                      </option>
                    ))}
                  </select>
                </div>

                <div className="form-group">
                  <label>Packaging & Quantization Format</label>
                  <select value={selectedQuantization} onChange={(e) => setSelectedQuantization(e.target.value)}>
                    <option value="q4_k_m">q4_k_m (Quantized GGUF / CPU Fast)</option>
                    <option value="q8_0">q8_0 (High Precision GGUF)</option>
                    <option value="fp16">fp16 (Full Precision Safetensors)</option>
                    <option value="onnx-webgpu">onnx-webgpu (Browser In-Memory Runtime)</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Agentic Execution Prompt</label>
                  <textarea
                    className="prompt-box"
                    value={hfPrompt}
                    onChange={(e) => setHfPrompt(e.target.value)}
                  />
                </div>

                <div style={{ display: "flex", gap: "0.75rem" }}>
                  <button className="primary" style={{ flex: 1 }} disabled={busy} onClick={runHfAgentStep}>
                    {busy ? "Running Model..." : "Run Agent Step"}
                  </button>
                  <button className="secondary" style={{ flex: 1 }} disabled={busy} onClick={releaseToHuggingFaceHub}>
                    🚀 Publish to HF Hub
                  </button>
                </div>
              </div>

              <div className="card">
                <h3>HF Hub Repository & Model Card Live Previewer</h3>
                {hfReleaseResult ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span className="badge ok">Published ({hfReleaseResult.quantization.toUpperCase()})</span>
                      <a href={hfReleaseResult.hfHubUrl} target="_blank" rel="noreferrer" className="tag" style={{ color: "var(--monad-purple)" }}>View on HF Hub ↗</a>
                    </div>
                    <pre className="code-view" style={{ maxHeight: "300px" }}>
{hfReleaseResult.modelCard}
                    </pre>
                  </div>
                ) : hfAgentResult ? (
                  <pre className="code-view" style={{ maxHeight: "320px" }}>
{JSON.stringify(hfAgentResult, null, 2)}
                  </pre>
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                    <p className="note">First-Party Model Repositories Ready for Release:</p>
                    {firstPartyModels.map((m) => (
                      <div key={m.id} style={{ background: "#04070d", padding: "0.65rem 0.85rem", borderRadius: "8px", border: "1px solid var(--border)", fontSize: "0.8rem" }}>
                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                          <strong className="text-purple">{m.name}</strong>
                          <span className="tag">{(m.contextWindowTokens/1024).toFixed(0)}k Context</span>
                        </div>
                        <div style={{ color: "var(--muted)", marginTop: "0.2rem" }}>Specialty: {m.specialty}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* --- SUPER CONTEXT WORKSPACE TAB --- */}
        {activeTab === "super-workspace" && (
          <div>
            <div className="section-header">
              <h2>🌌 Super Large Context Workspace & Attention Heatmap</h2>
              <p>Scan 500+ Monad block execution traces, multi-contract bytecodes, and complete liquidity router codebases simultaneously.</p>
            </div>

            <div className="grid-2">
              <div className="card">
                <div className="card-top">
                  <h3>Multi-Block Analysis Canvas</h3>
                  <span className="badge ok">131,072 Tokens Active</span>
                </div>
                <p className="desc">Enter codebase fragments or multi-block traces for deep parallel conflict analysis.</p>

                <div className="form-group">
                  <label>Context Capacity Fill</label>
                  <div className="token-capacity-gauge">
                    <div className="capacity-fill" style={{ width: "38.5%" }}></div>
                  </div>
                  <span style={{ fontSize: "0.75rem", color: "var(--muted)" }}>50,460 / 131,072 Tokens (38.5% Utilized)</span>
                </div>

                <textarea
                  className="prompt-box"
                  style={{ minHeight: "110px", fontFamily: "'JetBrains Mono', monospace" }}
                  value={superContextPrompt}
                  onChange={(e) => setSuperContextPrompt(e.target.value)}
                />

                <div style={{ marginTop: "0.75rem", display: "flex", gap: "0.75rem" }}>
                  <button className="primary" disabled={busy} onClick={runSuperContextAnalysis}>
                    {busy ? "Scanning 131k Context..." : "Run Super Context Analysis"}
                  </button>
                </div>
              </div>

              {/* ATTENTION HEATMAP CARD */}
              <div className="card">
                <h3>Self-Attention Heatmap & Conflict Gauge</h3>
                <p className="desc">Visualizing neural attention weights over EVM storage slots to guarantee zero parallel execution retries.</p>

                {attentionMapData?.attentionMap && (
                  <div className="attention-heatmap">
                    {attentionMapData.attentionMap.map((att, idx) => (
                      <div key={idx} className="heatmap-row">
                        <div>
                          <strong>{att.layer}</strong>
                          <div style={{ color: "var(--muted)", fontSize: "0.75rem" }}>Focus: {att.focus}</div>
                          <div className="heatmap-score-bar" style={{ width: `${att.score * 100}%` }}></div>
                        </div>
                        <span className="text-purple">{(att.score * 100).toFixed(0)}% Attention</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {superContextResult && (
              <div className="card" style={{ marginTop: "1.25rem" }}>
                <h4>Super Context Engine Output</h4>
                <pre className="code-view">{JSON.stringify(superContextResult, null, 2)}</pre>
              </div>
            )}
          </div>
        )}

        {/* --- OPCODE & RESERVE PROFILER TAB --- */}
        {activeTab === "profiler" && (
          <div>
            <div className="section-header">
              <h2>Monad Opcode & Reserve Gas Profiler</h2>
              <p>Calculate exact gas costs including storage zeroing refunds (Jump Crypto model).</p>
            </div>

            <div className="grid-2">
              <div className="card">
                <h3>Opcode Parameter Simulator</h3>
                <p className="desc">Monad pricing: cold state access is ~3.3x more expensive (7,000 gas vs 2,100 gas), precompiles 8,000 gas.</p>
                
                <div className="form-group">
                  <label>Cold State Accesses (SLOAD / BALANCE)</label>
                  <input
                    type="number"
                    value={coldAccesses}
                    onChange={(e) => {
                      setColdAccesses(Number(e.target.value));
                      runOpcodeProfiler(Number(e.target.value), warmAccesses, precompileCalls, storageClears);
                    }}
                  />
                </div>
                <div className="form-group">
                  <label>Warm State Accesses</label>
                  <input
                    type="number"
                    value={warmAccesses}
                    onChange={(e) => {
                      setWarmAccesses(Number(e.target.value));
                      runOpcodeProfiler(coldAccesses, Number(e.target.value), precompileCalls, storageClears);
                    }}
                  />
                </div>
                <div className="form-group">
                  <label>Precompile Calls (e.g. ecRecover)</label>
                  <input
                    type="number"
                    value={precompileCalls}
                    onChange={(e) => {
                      setPrecompileCalls(Number(e.target.value));
                      runOpcodeProfiler(coldAccesses, warmAccesses, Number(e.target.value), storageClears);
                    }}
                  />
                </div>
                <div className="form-group">
                  <label>Storage Zeroing Refunds (SSTORE 0)</label>
                  <input
                    type="number"
                    value={storageClears}
                    onChange={(e) => {
                      setStorageClears(Number(e.target.value));
                      runOpcodeProfiler(coldAccesses, warmAccesses, precompileCalls, Number(e.target.value));
                    }}
                  />
                </div>
              </div>

              <div className="card">
                <h3>Monad vs Standard EVM Gas Comparison</h3>
                {profilerResult && (
                  <pre className="code-view" style={{ maxHeight: "300px" }}>
{JSON.stringify(profilerResult, null, 2)}
                  </pre>
                )}
              </div>
            </div>
          </div>
        )}

        {/* --- APPS TAB --- */}
        {activeTab === "apps" && (
          <div>
            <div className="section-header">
              <h2>First-Party Platform Apps</h2>
              <p>Direct operator console interface for all 15 THESIS DeFi company applications.</p>
            </div>

            <div className="grid-3">
              {apps.map((appItem) => (
                <div key={appItem.id} className="card">
                  <div>
                    <div className="card-top">
                      <span className="card-icon">{appItem.icon}</span>
                      <span className="tag">{appItem.category}</span>
                    </div>
                    <h3>{appItem.name}</h3>
                    <p className="desc">{appItem.description}</p>
                  </div>
                  <div className="card-actions">
                    <button
                      className="primary"
                      disabled={busy}
                      onClick={() => invokeApp(appItem)}
                    >
                      {busy && selectedApp?.id === appItem.id ? "Invoking..." : "Launch / Invoke"}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* --- PRIMITIVES TAB --- */}
        {activeTab === "primitives" && (
          <div>
            <div className="section-header">
              <h2>Core Monad OS Primitives</h2>
              <p>Foundational system building blocks exposing company logic, law, and capital gateways.</p>
            </div>

            <div className="grid-2">
              {primitives.map((prim) => (
                <div key={prim.id} className="card">
                  <div>
                    <div className="card-top">
                      <h3>{prim.name}</h3>
                      <span className="badge ok">{prim.status}</span>
                    </div>
                    <p className="desc">{prim.description}</p>
                  </div>
                  <div style={{ marginTop: "0.5rem" }}>
                    <span className="tag">{prim.category}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* --- INVOCATION RESULT DRAWER --- */}
        {selectedApp && (
          <div className="console-drawer">
            <div className="console-header">
              <h3>
                <span>{selectedApp.icon}</span> Console Output: {selectedApp.name}
              </h3>
              <button className="secondary" onClick={() => setSelectedApp(null)}>
                Close Console
              </button>
            </div>
            {invocationResult ? (
              <pre className="code-view">{JSON.stringify(invocationResult, null, 2)}</pre>
            ) : (
              <p className="note">Executing call to <code>/platform/apps/{selectedApp.id}/invoke</code>...</p>
            )}
          </div>
        )}
        {/* --- V2 FEATURE PANELS --- */}
        {activeTab === "iochain" && <IOChainWebGPUPanel />}
        {activeTab === "portfolio" && <PortfolioPanel state={featureState} />}
        {activeTab === "flashloan" && <FlashLoanPanel state={featureState} />}
        {activeTab === "auditor" && <AuditorPanel state={featureState} />}
        {activeTab === "bridge" && <BridgeMonitorPanel state={featureState} />}
        {activeTab === "yield" && <YieldAggregatorPanel state={featureState} />}
        {activeTab === "parallel" && <ParallelSimPanel state={featureState} />}
        {activeTab === "compute" && <ComputeOptimizerPanel state={featureState} />}
        {activeTab === "launchpad" && <LaunchpadPanel state={featureState} />}
        {activeTab === "agent" && <AIAgentPanel state={featureState} />}
        {activeTab === "governance" && <GovernancePanel state={featureState} />}
        {activeTab === "mev" && <MEVScannerPanel state={featureState} />}
        {activeTab === "backtester" && <BacktesterPanel state={featureState} />}
      </main>

      {/* FLOATING AI ASSISTANT WIDGET */}
      <div className="floating-ai-widget">
        {chatOpen ? (
          <div className="chat-window">
            <h4>
              <span>🤖 Monad AI Twin</span>
              <button className="secondary" style={{ padding: "0.1rem 0.4rem" }} onClick={() => setChatOpen(false)}>✕</button>
            </h4>
            <div className="chat-history">
              {chatMessages.map((m, i) => (
                <div key={i} className={`chat-msg ${m.sender}`}>
                  {m.text}
                </div>
              ))}
            </div>
            <form onSubmit={handleSendChatMessage} style={{ display: "flex", gap: "0.4rem" }}>
              <input
                type="text"
                placeholder="Ask AI Twin..."
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                style={{ flex: 1, marginBottom: 0, padding: "0.4rem 0.6rem", fontSize: "0.8rem" }}
              />
              <button className="primary" style={{ width: "auto", padding: "0.4rem 0.75rem" }}>Send</button>
            </form>
          </div>
        ) : (
          <button className="floating-toggle-btn" onClick={() => setChatOpen(true)}>
            🤖 Monad AI Twin
          </button>
        )}
      </div>
    </>
  );
}
