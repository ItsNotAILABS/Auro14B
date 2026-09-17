import React, { useState } from 'react';

// Shared API helper (matches App.jsx)
async function api(path, opts = {}) {
  try {
    const r = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
    return await r.json();
  } catch (e) {
    console.error("API Error:", e);
    return { error: e.message };
  }
}

export function useFeatureState() {
  // 1. Portfolio State
  const [portfolio, setPortfolio] = useState({
    loading: false,
    positions: [
      { asset: 'MON', chain: 'Monad', protocol: 'Kuru', amount: '10,000', entry: '$1.20', current: '$1.45', pnlUsd: '+$2,500', pnlPct: '+20.8%' },
      { asset: 'SOL', chain: 'Solana', protocol: 'Ambient', amount: '250', entry: '$140.00', current: '$155.00', pnlUsd: '+$3,750', pnlPct: '+10.7%' },
      { asset: 'USDC', chain: 'Monad', protocol: 'Wallet', amount: '50,000', entry: '$1.00', current: '$1.00', pnlUsd: '$0', pnlPct: '0.0%' },
    ],
    summary: { total: '$75,250.00', change24h: '+$1,250 (1.6%)', change7d: '+$6,250 (9.0%)', sharpe: '2.4', drawdown: '-4.2%' }
  });
  
  const refreshPortfolio = async () => {
    setPortfolio(p => ({ ...p, loading: true }));
    setTimeout(() => setPortfolio(p => ({ ...p, loading: false })), 800);
  };

  // 2. Flash Loan State
  const [flashLoan, setFlashLoan] = useState({
    steps: [{ id: 1, action: 'Borrow', protocol: 'Kuru', amount: '1000', chain: 'Monad' }],
    simulating: false,
    result: null
  });

  const addFlashLoanStep = () => {
    setFlashLoan(s => ({
      ...s,
      steps: [...s.steps, { id: Date.now(), action: 'Swap', protocol: 'Ambient', amount: '', chain: 'Monad' }]
    }));
  };

  const removeFlashLoanStep = (id) => {
    setFlashLoan(s => ({ ...s, steps: s.steps.filter(step => step.id !== id) }));
  };

  const updateFlashLoanStep = (id, field, value) => {
    setFlashLoan(s => ({
      ...s,
      steps: s.steps.map(step => step.id === id ? { ...step, [field]: value } : step)
    }));
  };

  const simulateFlashLoan = () => {
    setFlashLoan(s => ({ ...s, simulating: true, result: null }));
    setTimeout(() => setFlashLoan(s => ({
      ...s, simulating: false, result: { profit: '+$45.20', gas: '-$0.12', latency: '400ms', net: '+$45.08', score: '98/100 (Safe)' }
    })), 1200);
  };

  // 3. Auditor State
  const [auditor, setAuditor] = useState({
    code: '',
    language: 'Solidity',
    contractName: '',
    running: false,
    findings: null
  });

  const runAudit = () => {
    setAuditor(s => ({ ...s, running: true, findings: null }));
    setTimeout(() => setAuditor(s => ({
      ...s, running: false, findings: [
        { severity: 'Critical', title: 'Reentrancy in withdraw()', location: 'Line 42', rec: 'Use check-effects-interactions pattern or ReentrancyGuard' },
        { severity: 'Medium', title: 'Unchecked return value', location: 'Line 89', rec: 'Ensure token transfer return value is checked' },
        { severity: 'Info', title: 'Gas optimization', location: 'Line 12', rec: 'Pack storage variables tightly' }
      ]
    })), 2000);
  };

  // 4. Bridge Monitor State
  const [bridge, setBridge] = useState({
    refreshing: false
  });

  // 5. Yield Aggregator State
  const [yieldState, setYieldState] = useState({
    allocations: { kuru: 0, ambient: 0, raydium: 0 }
  });

  const updateYieldAllocation = (protocol, value) => {
    setYieldState(s => ({ ...s, allocations: { ...s.allocations, [protocol]: value } }));
  };

  // 6. Parallel Sim State
  const [parallelSim, setParallelSim] = useState({
    txs: '[\n  {"to": "0xABC...", "data": "0x123..."},\n  {"to": "0xDEF...", "data": "0x456..."}\n]',
    running: false,
    result: null
  });

  const runSim = () => {
    setParallelSim(s => ({ ...s, running: true, result: null }));
    setTimeout(() => setParallelSim(s => ({
      ...s, running: false, result: { groups: 4, speedup: '3.2x', eip2930: '[{"address": "0xABC...", "storageKeys": ["0x0"]}]' }
    })), 1500);
  };

  // 7. Compute Optimizer State
  const [compute, setCompute] = useState({
    instructions: '', accounts: '', running: false, result: null
  });

  const optimizeCompute = () => {
    setCompute(s => ({ ...s, running: true, result: null }));
    setTimeout(() => setCompute(s => ({
      ...s, running: false, result: { cuLimit: 145000, cuPrice: '12 micro-lamports', savings: '45 bytes' }
    })), 1000);
  };

  // 8. Launchpad State
  const [launchpad, setLaunchpad] = useState({
    name: '', symbol: '', supply: '', standard: 'ERC-20', decimals: '18', deploying: false, receipt: null
  });

  const deployToken = () => {
    setLaunchpad(s => ({ ...s, deploying: true, receipt: null }));
    setTimeout(() => setLaunchpad(s => ({
      ...s, deploying: false, receipt: '0x9a8fB... (Confirmed)'
    })), 2500);
  };

  // 9. AI Agent State
  const [agent, setAgent] = useState({
    status: 'Running',
    uptime: '4d 12h',
    txs: 1432,
    profit: '+$1,240.50',
    strategies: { mev: true, deltaNeutral: false }
  });

  const toggleStrategy = (strat) => {
    setAgent(s => ({ ...s, strategies: { ...s.strategies, [strat]: !s.strategies[strat] } }));
  };

  const stopAgent = () => {
    setAgent(s => ({ ...s, status: 'Stopped' }));
  };

  // 10. Governance State
  const [gov, setGov] = useState({});

  // 11. MEV Scanner State
  const [mev, setMev] = useState({ scanning: true });

  // 12. Backtester State
  const [backtest, setBacktest] = useState({
    running: false, result: null
  });

  const runBacktest = () => {
    setBacktest(s => ({ ...s, running: true, result: null }));
    setTimeout(() => setBacktest(s => ({
      ...s, running: false, result: { return: '+45.2%', sharpe: 1.8, drawdown: '-12.4%', winRate: '68%' }
    })), 2000);
  };

  return {
    portfolio, refreshPortfolio,
    flashLoan, addFlashLoanStep, removeFlashLoanStep, updateFlashLoanStep, simulateFlashLoan,
    auditor, setAuditor, runAudit,
    bridge, setBridge,
    yieldState, updateYieldAllocation,
    parallelSim, setParallelSim, runSim,
    compute, setCompute, optimizeCompute,
    launchpad, setLaunchpad, deployToken,
    agent, toggleStrategy, stopAgent,
    gov, setGov,
    mev, setMev,
    backtest, setBacktest, runBacktest
  };
}

export function PortfolioPanel({ state }) {
  const { portfolio, refreshPortfolio } = state;
  return (
    <div className="hft-panel" style={{ background: '#04070d', padding: '20px', borderRadius: '8px' }}>
      <div className="section-header" style={{ marginBottom: '20px' }}>
        <h2 style={{ margin: 0, color: 'var(--monad-purple)' }}>Portfolio Management</h2>
        <button className="secondary" onClick={refreshPortfolio} disabled={portfolio.loading}>
          {portfolio.loading ? 'Refreshing...' : 'Refresh Positions'}
        </button>
      </div>
      
      <div className="grid-2" style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '15px', marginBottom: '20px' }}>
        <div className="card" style={{ padding: '15px' }}>
          <div className="desc">Total Value</div>
          <div className="bold" style={{ fontSize: '1.4em' }}>{portfolio.summary.total}</div>
        </div>
        <div className="card" style={{ padding: '15px' }}>
          <div className="desc">24h Change</div>
          <div className="bold text-green">{portfolio.summary.change24h}</div>
        </div>
        <div className="card" style={{ padding: '15px' }}>
          <div className="desc">7d Change</div>
          <div className="bold text-green">{portfolio.summary.change7d}</div>
        </div>
        <div className="card" style={{ padding: '15px' }}>
          <div className="desc">Sharpe Ratio</div>
          <div className="bold">{portfolio.summary.sharpe}</div>
        </div>
        <div className="card" style={{ padding: '15px' }}>
          <div className="desc">Max Drawdown</div>
          <div className="bold" style={{ color: 'var(--danger)' }}>{portfolio.summary.drawdown}</div>
        </div>
      </div>

      <div className="card" style={{ padding: '0', overflow: 'hidden' }}>
        <div className="card-top" style={{ padding: '15px', borderBottom: '1px solid var(--border)' }}>
          <h3 style={{ margin: 0 }}>Unified Positions</h3>
          <span className="badge ok">Live Sync</span>
        </div>
        <table className="data-table" style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              <th style={{ padding: '10px 15px' }}>Asset</th>
              <th style={{ padding: '10px 15px' }}>Chain</th>
              <th style={{ padding: '10px 15px' }}>Protocol</th>
              <th style={{ padding: '10px 15px' }}>Amount</th>
              <th style={{ padding: '10px 15px' }}>Entry Price</th>
              <th style={{ padding: '10px 15px' }}>Current Price</th>
              <th style={{ padding: '10px 15px' }}>PnL ($)</th>
              <th style={{ padding: '10px 15px' }}>PnL (%)</th>
            </tr>
          </thead>
          <tbody>
            {portfolio.positions.map((pos, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #1a1a2e' }}>
                <td style={{ padding: '10px 15px' }} className="bold">{pos.asset}</td>
                <td style={{ padding: '10px 15px' }}><span className="tag">{pos.chain}</span></td>
                <td style={{ padding: '10px 15px' }}>{pos.protocol}</td>
                <td style={{ padding: '10px 15px' }}>{pos.amount}</td>
                <td style={{ padding: '10px 15px', color: 'var(--muted)' }}>{pos.entry}</td>
                <td style={{ padding: '10px 15px' }}>{pos.current}</td>
                <td style={{ padding: '10px 15px' }} className={pos.pnlUsd.startsWith('+') ? 'text-green' : pos.pnlUsd === '$0' ? '' : 'text-red'}>{pos.pnlUsd}</td>
                <td style={{ padding: '10px 15px' }} className={pos.pnlPct.startsWith('+') ? 'text-green' : pos.pnlPct === '0.0%' ? '' : 'text-red'}>{pos.pnlPct}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card" style={{ marginTop: '20px', padding: '15px' }}>
        <div className="card-top" style={{ marginBottom: '15px' }}>
          <h3 style={{ margin: 0 }}>Concentration Risk</h3>
        </div>
        <div style={{ display: 'flex', height: '24px', borderRadius: '4px', overflow: 'hidden' }}>
          <div style={{ width: '40%', background: '#6d28d9', title: 'USDC (40%)' }}></div>
          <div style={{ width: '35%', background: '#10b981', title: 'MON (35%)' }}></div>
          <div style={{ width: '25%', background: '#3b82f6', title: 'SOL (25%)' }}></div>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '8px', fontSize: '12px', color: 'var(--muted)' }}>
          <span>USDC: 40%</span>
          <span>MON: 35%</span>
          <span>SOL: 25%</span>
        </div>
      </div>
    </div>
  );
}

export function FlashLoanPanel({ state }) {
  const { flashLoan, addFlashLoanStep, removeFlashLoanStep, updateFlashLoanStep, simulateFlashLoan } = state;
  return (
    <div className="hft-panel" style={{ background: '#04070d', padding: '20px', borderRadius: '8px' }}>
      <h2 style={{ color: 'var(--monad-purple)' }}>Flash Loan Strategy Composer</h2>
      
      <div className="card" style={{ padding: '20px', marginBottom: '20px' }}>
        {flashLoan.steps.map((step, idx) => (
          <div key={step.id} style={{ display: 'flex', gap: '10px', alignItems: 'center', marginBottom: '10px' }}>
            <span className="badge" style={{ minWidth: '30px', textAlign: 'center' }}>{idx + 1}</span>
            <div className="form-group" style={{ margin: 0, flex: 1 }}>
              <select value={step.action} onChange={e => updateFlashLoanStep(step.id, 'action', e.target.value)} style={{ width: '100%' }}>
                <option>Borrow</option>
                <option>Swap</option>
                <option>Bridge</option>
                <option>Repay</option>
              </select>
            </div>
            <div className="form-group" style={{ margin: 0, flex: 1 }}>
              <select value={step.protocol} onChange={e => updateFlashLoanStep(step.id, 'protocol', e.target.value)} style={{ width: '100%' }}>
                <option>Kuru</option>
                <option>Ambient</option>
                <option>Raydium</option>
                <option>Wormhole</option>
              </select>
            </div>
            <div className="form-group" style={{ margin: 0, flex: 1 }}>
              <input type="text" placeholder="Amount" value={step.amount} onChange={e => updateFlashLoanStep(step.id, 'amount', e.target.value)} style={{ width: '100%' }} />
            </div>
            <div className="form-group" style={{ margin: 0, flex: 1 }}>
              <select value={step.chain} onChange={e => updateFlashLoanStep(step.id, 'chain', e.target.value)} style={{ width: '100%' }}>
                <option>Monad</option>
                <option>Solana</option>
              </select>
            </div>
            <button className="secondary" onClick={() => removeFlashLoanStep(step.id)} style={{ padding: '5px 10px' }}>X</button>
          </div>
        ))}
        <button className="secondary" onClick={addFlashLoanStep}>+ Add Step</button>
      </div>

      <div style={{ display: 'flex', gap: '15px' }}>
        <button className="primary" onClick={simulateFlashLoan} disabled={flashLoan.simulating} style={{ flex: 1 }}>
          {flashLoan.simulating ? 'Simulating...' : 'Simulate Strategy'}
        </button>
        <button className="primary" style={{ flex: 1, background: 'var(--monad-purple)' }} disabled={!flashLoan.result}>
          Execute Flash Loan
        </button>
      </div>

      {flashLoan.result && (
        <div className="card" style={{ marginTop: '20px', padding: '15px', border: '1px solid var(--monad-purple)' }}>
          <div className="card-top">
            <h3 style={{ margin: 0 }}>Simulation Results</h3>
            <span className="badge ok">Safety Score: {flashLoan.result.score}</span>
          </div>
          <div className="grid-2" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '15px', marginTop: '15px' }}>
            <div>
              <div className="desc">Est. Profit</div>
              <div className="bold text-green">{flashLoan.result.profit}</div>
            </div>
            <div>
              <div className="desc">Gas Costs</div>
              <div className="bold" style={{ color: 'var(--danger)' }}>{flashLoan.result.gas}</div>
            </div>
            <div>
              <div className="desc">Bridge Latency</div>
              <div className="bold">{flashLoan.result.latency}</div>
            </div>
            <div>
              <div className="desc">Net Profit</div>
              <div className="bold text-green" style={{ fontSize: '1.2em' }}>{flashLoan.result.net}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export function AuditorPanel({ state }) {
  const { auditor, setAuditor, runAudit } = state;
  return (
    <div className="hft-panel" style={{ background: '#04070d', padding: '20px', borderRadius: '8px' }}>
      <div className="section-header">
        <h2 style={{ color: 'var(--monad-purple)', margin: 0 }}>Smart Contract Auditor</h2>
      </div>
      
      <div className="grid-2" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', margin: '20px 0' }}>
        <div className="form-group">
          <label>Language</label>
          <select value={auditor.language} onChange={e => setAuditor({...auditor, language: e.target.value})}>
            <option>Solidity</option>
            <option>Rust (Anchor)</option>
          </select>
        </div>
        <div className="form-group">
          <label>Contract Name</label>
          <input type="text" placeholder="e.g. Vault.sol" value={auditor.contractName} onChange={e => setAuditor({...auditor, contractName: e.target.value})} />
        </div>
      </div>

      <div className="form-group">
        <label>Contract Code</label>
        <textarea 
          style={{ width: '100%', minHeight: '200px', fontFamily: 'monospace', background: '#0a0f1a', color: '#e2e8f0', border: '1px solid var(--border)', padding: '10px' }}
          value={auditor.code}
          onChange={e => setAuditor({...auditor, code: e.target.value})}
          placeholder="// Paste your smart contract code here..."
        />
      </div>

      <button className="primary" onClick={runAudit} disabled={auditor.running} style={{ width: '100%', padding: '12px' }}>
        {auditor.running ? 'Analyzing Code...' : 'Run Audit'}
      </button>

      {auditor.findings && (
        <div style={{ marginTop: '20px' }}>
          <div className="card" style={{ padding: '15px', marginBottom: '15px' }}>
            <h3 style={{ margin: '0 0 10px 0' }}>Cross-Chain Safety Evaluation</h3>
            <p className="note" style={{ margin: 0 }}>Heuristic score: 85/100. Contract appears safe against known cross-chain replay attacks. Standard Reentrancy flags identified.</p>
          </div>
          
          <h3 style={{ marginBottom: '10px' }}>Audit Findings</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {auditor.findings.map((f, i) => (
              <div key={i} className="card" style={{ padding: '15px', borderLeft: `4px solid ${f.severity === 'Critical' ? 'red' : f.severity === 'Medium' ? 'orange' : 'var(--muted)'}` }}>
                <div className="card-top" style={{ marginBottom: '10px' }}>
                  <div className="bold">{f.title} <span className="desc">({f.location})</span></div>
                  <span className="badge" style={{ background: f.severity === 'Critical' ? '#7f1d1d' : f.severity === 'Medium' ? '#7c2d12' : '#1e293b' }}>
                    {f.severity}
                  </span>
                </div>
                <div className="note">{f.rec}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export function BridgeMonitorPanel({ state }) {
  return (
    <div className="hft-panel" style={{ background: '#04070d', padding: '20px', borderRadius: '8px' }}>
      <h2 style={{ color: 'var(--monad-purple)' }}>Bridge Monitor & DVN Health</h2>
      
      <div className="grid-2" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '15px', marginBottom: '20px' }}>
        <div className="card" style={{ padding: '15px' }}>
          <div className="desc">In-Flight Messages</div>
          <div className="bold" style={{ fontSize: '1.4em' }}>12</div>
        </div>
        <div className="card" style={{ padding: '15px' }}>
          <div className="desc">Avg Relay Latency</div>
          <div className="bold text-green">1.2s</div>
        </div>
        <div className="card" style={{ padding: '15px' }}>
          <div className="desc">Stuck Messages</div>
          <div className="bold">0</div>
        </div>
        <div className="card" style={{ padding: '15px' }}>
          <div className="desc">Guardian Set Status</div>
          <div className="bold text-green">19/19 Active</div>
        </div>
      </div>

      <div className="card" style={{ padding: '0', overflow: 'hidden', marginBottom: '20px' }}>
        <div className="card-top" style={{ padding: '15px', borderBottom: '1px solid var(--border)' }}>
          <h3 style={{ margin: 0 }}>Cost Comparison (Monad → Solana)</h3>
        </div>
        <table className="data-table" style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              <th style={{ padding: '10px 15px' }}>Provider</th>
              <th style={{ padding: '10px 15px' }}>Fee (USDC)</th>
              <th style={{ padding: '10px 15px' }}>Latency</th>
              <th style={{ padding: '10px 15px' }}>Reliability</th>
              <th style={{ padding: '10px 15px' }}>Status</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid #1a1a2e' }}>
              <td style={{ padding: '10px 15px' }} className="bold">Wormhole</td>
              <td style={{ padding: '10px 15px' }}>$0.45</td>
              <td style={{ padding: '10px 15px' }}>~2.5s</td>
              <td style={{ padding: '10px 15px' }}>99.9%</td>
              <td style={{ padding: '10px 15px' }}><span className="badge ok">Optimal</span></td>
            </tr>
            <tr style={{ borderBottom: '1px solid #1a1a2e' }}>
              <td style={{ padding: '10px 15px' }} className="bold">LayerZero</td>
              <td style={{ padding: '10px 15px' }}>$0.60</td>
              <td style={{ padding: '10px 15px' }}>~1.8s</td>
              <td style={{ padding: '10px 15px' }}>99.9%</td>
              <td style={{ padding: '10px 15px' }}><span className="badge">Active</span></td>
            </tr>
            <tr>
              <td style={{ padding: '10px 15px' }} className="bold">deBridge</td>
              <td style={{ padding: '10px 15px' }}>$0.55</td>
              <td style={{ padding: '10px 15px' }}>~3.0s</td>
              <td style={{ padding: '10px 15px' }}>99.5%</td>
              <td style={{ padding: '10px 15px' }}><span className="badge">Active</span></td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="card" style={{ padding: '15px' }}>
        <h3 style={{ margin: '0 0 15px 0' }}>In-Flight Messages</h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
          <div style={{ border: '1px solid var(--border)', padding: '10px', borderRadius: '4px' }}>
            <div className="card-top" style={{ marginBottom: '8px' }}>
              <span className="desc">Tx: 0x9a8...f12</span>
              <span className="note">Monad → Solana (100 USDC)</span>
            </div>
            <div style={{ width: '100%', height: '8px', background: '#1e293b', borderRadius: '4px', overflow: 'hidden' }}>
              <div className="capacity-fill" style={{ width: '60%', height: '100%', background: 'var(--monad-purple)' }}></div>
            </div>
            <div className="note" style={{ marginTop: '8px', textAlign: 'right' }}>Awaiting Solana Finality...</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export function YieldAggregatorPanel({ state }) {
  const { yieldState, updateYieldAllocation } = state;
  return (
    <div className="hft-panel" style={{ background: '#04070d', padding: '20px', borderRadius: '8px' }}>
      <h2 style={{ color: 'var(--monad-purple)' }}>Cross-Chain Yield Aggregator</h2>
      
      <div className="card" style={{ padding: '20px', textAlign: 'center', marginBottom: '20px', background: 'linear-gradient(180deg, #1e1b4b 0%, #04070d 100%)', border: '1px solid var(--monad-purple)' }}>
        <div className="desc">Projected Aggregate APY</div>
        <div className="bold text-green" style={{ fontSize: '3em', textShadow: '0 0 10px rgba(16,185,129,0.3)' }}>24.8%</div>
      </div>

      <div className="card" style={{ padding: '0', overflow: 'hidden', marginBottom: '20px' }}>
        <table className="data-table" style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              <th style={{ padding: '10px 15px' }}>Protocol</th>
              <th style={{ padding: '10px 15px' }}>Chain</th>
              <th style={{ padding: '10px 15px' }}>Strategy</th>
              <th style={{ padding: '10px 15px' }}>APY</th>
              <th style={{ padding: '10px 15px' }}>TVL</th>
              <th style={{ padding: '10px 15px' }}>Risk Score</th>
              <th style={{ padding: '10px 15px' }}>Allocation (%)</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid #1a1a2e' }}>
              <td style={{ padding: '10px 15px' }} className="bold">Kuru</td>
              <td style={{ padding: '10px 15px' }}><span className="tag">Monad</span></td>
              <td style={{ padding: '10px 15px' }}>MON/USDC LP</td>
              <td style={{ padding: '10px 15px' }} className="bold text-green">18.5%</td>
              <td style={{ padding: '10px 15px' }}>$42M</td>
              <td style={{ padding: '10px 15px' }}><span className="badge ok">2/10</span></td>
              <td style={{ padding: '10px 15px' }}>
                <input type="number" min="0" max="100" value={yieldState.allocations.kuru || 0} onChange={e => updateYieldAllocation('kuru', e.target.value)} style={{ width: '60px', padding: '4px' }} />
              </td>
            </tr>
            <tr style={{ borderBottom: '1px solid #1a1a2e' }}>
              <td style={{ padding: '10px 15px' }} className="bold">Ambient</td>
              <td style={{ padding: '10px 15px' }}><span className="tag">Monad</span></td>
              <td style={{ padding: '10px 15px' }}>Stable Vol</td>
              <td style={{ padding: '10px 15px' }} className="bold text-green">12.1%</td>
              <td style={{ padding: '10px 15px' }}>$115M</td>
              <td style={{ padding: '10px 15px' }}><span className="badge ok">1/10</span></td>
              <td style={{ padding: '10px 15px' }}>
                <input type="number" min="0" max="100" value={yieldState.allocations.ambient || 0} onChange={e => updateYieldAllocation('ambient', e.target.value)} style={{ width: '60px', padding: '4px' }} />
              </td>
            </tr>
            <tr>
              <td style={{ padding: '10px 15px' }} className="bold">Raydium</td>
              <td style={{ padding: '10px 15px' }}><span className="tag">Solana</span></td>
              <td style={{ padding: '10px 15px' }}>SOL/USDC CLMM</td>
              <td style={{ padding: '10px 15px' }} className="bold text-green">34.2%</td>
              <td style={{ padding: '10px 15px' }}>$410M</td>
              <td style={{ padding: '10px 15px' }}><span className="badge" style={{ background: '#7c2d12' }}>6/10</span></td>
              <td style={{ padding: '10px 15px' }}>
                <input type="number" min="0" max="100" value={yieldState.allocations.raydium || 0} onChange={e => updateYieldAllocation('raydium', e.target.value)} style={{ width: '60px', padding: '4px' }} />
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <button className="primary" style={{ width: '100%', padding: '12px', background: 'var(--monad-purple)' }}>
        Rebalance Portfolio & Execute Routes
      </button>
    </div>
  );
}

export function ParallelSimPanel({ state }) {
  const { parallelSim, setParallelSim, runSim } = state;
  return (
    <div className="hft-panel" style={{ background: '#04070d', padding: '20px', borderRadius: '8px' }}>
      <h2 style={{ color: 'var(--monad-purple)' }}>Monad Parallel Execution Simulator</h2>
      
      <div className="form-group" style={{ marginBottom: '20px' }}>
        <label>Transaction Batch (JSON Array)</label>
        <textarea 
          style={{ width: '100%', minHeight: '120px', fontFamily: 'monospace', background: '#0a0f1a', color: '#e2e8f0', border: '1px solid var(--border)', padding: '10px' }}
          value={parallelSim.txs}
          onChange={e => setParallelSim({...parallelSim, txs: e.target.value})}
        />
      </div>

      <button className="primary" onClick={runSim} disabled={parallelSim.running} style={{ marginBottom: '20px' }}>
        {parallelSim.running ? 'Simulating...' : 'Simulate Parallel Execution'}
      </button>

      {parallelSim.result && (
        <div className="grid-2" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          <div className="card" style={{ padding: '15px' }}>
            <h3 style={{ margin: '0 0 15px 0' }}>Performance Profile</h3>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
              <span className="desc">Parallel Groups</span>
              <span className="bold">{parallelSim.result.groups}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
              <span className="desc">Estimated Speedup</span>
              <span className="bold text-green" style={{ fontSize: '1.2em' }}>{parallelSim.result.speedup}</span>
            </div>
            
            <h4 style={{ margin: '15px 0 10px 0', fontSize: '0.9em', color: 'var(--muted)' }}>Conflict Heatmap</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(10, 1fr)', gap: '4px' }}>
              {Array.from({ length: 30 }).map((_, i) => (
                <div key={i} style={{ aspectRatio: '1', background: Math.random() > 0.8 ? '#7f1d1d' : '#064e3b', borderRadius: '2px' }} title={Math.random() > 0.8 ? 'State Conflict' : 'Independent'}></div>
              ))}
            </div>
          </div>
          
          <div className="card" style={{ padding: '15px' }}>
            <h3 style={{ margin: '0 0 10px 0' }}>EIP-2930 Access List (Optimal)</h3>
            <pre className="code-view" style={{ background: '#000', padding: '10px', borderRadius: '4px', overflowX: 'auto', fontSize: '12px', color: 'var(--muted)', margin: 0 }}>
              {parallelSim.result.eip2930}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

export function ComputeOptimizerPanel({ state }) {
  const { compute, setCompute, optimizeCompute } = state;
  return (
    <div className="hft-panel" style={{ background: '#04070d', padding: '20px', borderRadius: '8px' }}>
      <h2 style={{ color: 'var(--monad-purple)' }}>Solana Compute Budget Optimizer</h2>
      
      <div className="grid-2" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
        <div className="form-group" style={{ margin: 0 }}>
          <label>Instructions (Base64 or JSON)</label>
          <textarea style={{ width: '100%', minHeight: '100px', fontFamily: 'monospace', background: '#0a0f1a', color: '#e2e8f0', border: '1px solid var(--border)' }} />
        </div>
        <div className="form-group" style={{ margin: 0 }}>
          <label>Account Keys</label>
          <textarea style={{ width: '100%', minHeight: '100px', fontFamily: 'monospace', background: '#0a0f1a', color: '#e2e8f0', border: '1px solid var(--border)' }} />
        </div>
      </div>

      <button className="primary" onClick={optimizeCompute} disabled={compute.running} style={{ marginBottom: '20px' }}>
        {compute.running ? 'Profiling...' : 'Optimize Transaction'}
      </button>

      {compute.result && (
        <div className="card" style={{ padding: '15px' }}>
          <div className="grid-2" style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '15px', marginBottom: '20px' }}>
            <div>
              <div className="desc">Optimal CU Limit</div>
              <div className="bold">{compute.result.cuLimit}</div>
            </div>
            <div>
              <div className="desc">Recommended CU Price</div>
              <div className="bold">{compute.result.cuPrice}</div>
            </div>
            <div>
              <div className="desc">Tx Size Savings (with ALT)</div>
              <div className="bold text-green">{compute.result.savings}</div>
            </div>
          </div>
          
          <h4 style={{ margin: '0 0 10px 0', color: 'var(--muted)' }}>Priority Fee Market Depth</h4>
          <div style={{ display: 'flex', height: '30px', borderRadius: '4px', overflow: 'hidden' }}>
            <div style={{ width: '20%', background: '#064e3b', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '10px' }}>Low (1-5)</div>
            <div style={{ width: '50%', background: '#b45309', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '10px' }}>Med (5-15)</div>
            <div style={{ width: '30%', background: '#7f1d1d', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '10px' }}>High (15+)</div>
          </div>
        </div>
      )}
    </div>
  );
}

export function LaunchpadPanel({ state }) {
  const { launchpad, setLaunchpad, deployToken } = state;
  return (
    <div className="hft-panel" style={{ background: '#04070d', padding: '20px', borderRadius: '8px' }}>
      <h2 style={{ color: 'var(--monad-purple)' }}>Cross-Chain Token Launchpad</h2>
      
      <div className="grid-2" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        <div className="card" style={{ padding: '20px' }}>
          <h3 style={{ margin: '0 0 15px 0' }}>Token Configuration</h3>
          <div className="form-group">
            <label>Name</label>
            <input type="text" value={launchpad.name} onChange={e => setLaunchpad({...launchpad, name: e.target.value})} placeholder="e.g. Monad Chad" style={{ width: '100%' }} />
          </div>
          <div className="form-group">
            <label>Symbol</label>
            <input type="text" value={launchpad.symbol} onChange={e => setLaunchpad({...launchpad, symbol: e.target.value})} placeholder="CHAD" style={{ width: '100%' }} />
          </div>
          <div className="form-group">
            <label>Total Supply</label>
            <input type="number" value={launchpad.supply} onChange={e => setLaunchpad({...launchpad, supply: e.target.value})} placeholder="1000000000" style={{ width: '100%' }} />
          </div>
          <div className="form-group">
            <label>Standard</label>
            <select value={launchpad.standard} onChange={e => setLaunchpad({...launchpad, standard: e.target.value})} style={{ width: '100%' }}>
              <option>ERC-20</option>
              <option>ERC-404</option>
            </select>
          </div>
        </div>

        <div className="card" style={{ padding: '20px' }}>
          <h3 style={{ margin: '0 0 15px 0' }}>Launch Settings</h3>
          <div className="form-group">
            <label>Kuru Initial Price (USDC)</label>
            <input type="text" placeholder="0.001" style={{ width: '100%' }} />
          </div>
          <div className="form-group" style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '20px' }}>
            <input type="checkbox" id="mirror" />
            <label htmlFor="mirror" style={{ margin: 0 }}>Deploy Mirror to Solana SPL via Wormhole</label>
          </div>
          
          <div style={{ marginTop: '30px', padding: '15px', background: '#0a0f1a', borderRadius: '4px', border: '1px solid var(--border)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
              <span className="desc">Estimated Deployment Gas</span>
              <span className="bold">0.005 MON</span>
            </div>
            <button className="primary" onClick={deployToken} disabled={launchpad.deploying} style={{ width: '100%', background: 'var(--monad-purple)' }}>
              {launchpad.deploying ? 'Deploying...' : 'Deploy Token & Liquidity'}
            </button>
            {launchpad.receipt && (
              <div className="note text-green" style={{ marginTop: '10px', textAlign: 'center' }}>
                Success! Receipt: {launchpad.receipt}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export function AIAgentPanel({ state }) {
  const { agent, toggleStrategy, stopAgent } = state;
  return (
    <div className="hft-panel" style={{ background: '#04070d', padding: '20px', borderRadius: '8px' }}>
      <div className="section-header" style={{ marginBottom: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: 0, color: 'var(--monad-purple)' }}>AI Trading Agent</h2>
        <button onClick={stopAgent} style={{ background: '#7f1d1d', color: 'white', border: 'none', padding: '8px 16px', borderRadius: '4px', fontWeight: 'bold', cursor: 'pointer' }}>
          EMERGENCY STOP
        </button>
      </div>

      <div className="grid-2" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '15px', marginBottom: '20px' }}>
        <div className="card" style={{ padding: '15px', border: agent.status === 'Running' ? '1px solid #10b981' : '1px solid #7f1d1d' }}>
          <div className="desc">Status</div>
          <div className="bold" style={{ color: agent.status === 'Running' ? '#10b981' : '#ef4444' }}>{agent.status}</div>
        </div>
        <div className="card" style={{ padding: '15px' }}>
          <div className="desc">Uptime</div>
          <div className="bold">{agent.uptime}</div>
        </div>
        <div className="card" style={{ padding: '15px' }}>
          <div className="desc">Total Txs Executed</div>
          <div className="bold">{agent.txs}</div>
        </div>
        <div className="card" style={{ padding: '15px' }}>
          <div className="desc">Agent PnL</div>
          <div className="bold text-green">{agent.profit}</div>
        </div>
      </div>

      <div className="grid-2" style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '20px' }}>
        <div className="card" style={{ padding: '15px' }}>
          <h3 style={{ margin: '0 0 15px 0' }}>Active Strategies</h3>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
            <span>Cross-Chain MEV</span>
            <button className={agent.strategies.mev ? "primary" : "secondary"} style={{ padding: '4px 12px' }} onClick={() => toggleStrategy('mev')}>
              {agent.strategies.mev ? 'ON' : 'OFF'}
            </button>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>Delta Neutral Yield</span>
            <button className={agent.strategies.deltaNeutral ? "primary" : "secondary"} style={{ padding: '4px 12px' }} onClick={() => toggleStrategy('deltaNeutral')}>
              {agent.strategies.deltaNeutral ? 'ON' : 'OFF'}
            </button>
          </div>
        </div>

        <div className="card" style={{ padding: '15px' }}>
          <h3 style={{ margin: '0 0 15px 0' }}>Agent Reasoning Trace</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <div className="note" style={{ borderLeft: '2px solid var(--monad-purple)', paddingLeft: '10px' }}>
              <div className="bold" style={{ color: '#e2e8f0' }}>Executed Arbitrage (MON/USDC)</div>
              Price discrepancy detected between Kuru ($1.45) and Ambient ($1.48). Executed atomic flash swap for 45 USDC profit.
            </div>
            <div className="note" style={{ borderLeft: '2px solid var(--muted)', paddingLeft: '10px' }}>
              <div className="bold" style={{ color: '#e2e8f0' }}>Skipped Yield Rebalance</div>
              Gas fees on Solana spiked to 5000 micro-lamports. Rebalance EV is negative. Waiting for base fee decay.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export function GovernancePanel({ state }) {
  return (
    <div className="hft-panel" style={{ background: '#04070d', padding: '20px', borderRadius: '8px' }}>
      <h2 style={{ color: 'var(--monad-purple)' }}>Protocol Governance & Voting</h2>
      
      <div className="card" style={{ padding: '0', overflow: 'hidden' }}>
        <table className="data-table" style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              <th style={{ padding: '10px 15px' }}>Proposal</th>
              <th style={{ padding: '10px 15px' }}>Protocol (Chain)</th>
              <th style={{ padding: '10px 15px' }}>Status</th>
              <th style={{ padding: '10px 15px' }}>AI Rec.</th>
              <th style={{ padding: '10px 15px' }}>Action</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid #1a1a2e' }}>
              <td style={{ padding: '10px 15px' }}>
                <div className="bold">MIP-24: Increase Monad RPC Rate Limits</div>
                <div style={{ display: 'flex', height: '4px', background: '#7f1d1d', marginTop: '5px' }}>
                  <div style={{ width: '85%', background: '#10b981' }}></div>
                </div>
              </td>
              <td style={{ padding: '10px 15px' }}>Monad DAO <span className="tag">Monad</span></td>
              <td style={{ padding: '10px 15px' }}>Active (2d left)</td>
              <td style={{ padding: '10px 15px' }}><span className="badge ok">YES</span></td>
              <td style={{ padding: '10px 15px', display: 'flex', gap: '5px' }}>
                <button style={{ background: '#10b981', color: 'white', border: 'none', padding: '4px 8px', borderRadius: '4px', cursor: 'pointer' }}>Yes</button>
                <button style={{ background: '#ef4444', color: 'white', border: 'none', padding: '4px 8px', borderRadius: '4px', cursor: 'pointer' }}>No</button>
              </td>
            </tr>
            <tr>
              <td style={{ padding: '10px 15px' }}>
                <div className="bold">JUP-12: Fee Tier Adjustment</div>
                <div style={{ display: 'flex', height: '4px', background: '#7f1d1d', marginTop: '5px' }}>
                  <div style={{ width: '45%', background: '#10b981' }}></div>
                </div>
              </td>
              <td style={{ padding: '10px 15px' }}>Jupiter <span className="tag">Solana</span></td>
              <td style={{ padding: '10px 15px' }}>Active (12h left)</td>
              <td style={{ padding: '10px 15px' }}><span className="badge" style={{ background: '#7f1d1d' }}>NO</span></td>
              <td style={{ padding: '10px 15px', display: 'flex', gap: '5px' }}>
                <button style={{ background: '#10b981', color: 'white', border: 'none', padding: '4px 8px', borderRadius: '4px', cursor: 'pointer' }}>Yes</button>
                <button style={{ background: '#ef4444', color: 'white', border: 'none', padding: '4px 8px', borderRadius: '4px', cursor: 'pointer' }}>No</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function MEVScannerPanel({ state }) {
  return (
    <div className="hft-panel" style={{ background: '#04070d', padding: '20px', borderRadius: '8px' }}>
      <h2 style={{ color: 'var(--monad-purple)' }}>MEV Scanner & Mempool Depth</h2>
      
      <div className="grid-2" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '15px', marginBottom: '20px' }}>
        <div className="card" style={{ padding: '15px' }}>
          <div className="desc">Total Extractable Value</div>
          <div className="bold" style={{ fontSize: '1.4em', color: '#10b981' }}>$4,250.00</div>
        </div>
        <div className="card" style={{ padding: '15px' }}>
          <div className="desc">Monad Parallel Windows</div>
          <div className="bold">14 Open</div>
        </div>
        <div className="card" style={{ padding: '15px' }}>
          <div className="desc">Solana Jito Bundles</div>
          <div className="bold">3 Pending</div>
        </div>
        <div className="card" style={{ padding: '15px' }}>
          <div className="desc">Cross-Chain Arbs</div>
          <div className="bold">2 Detected</div>
        </div>
      </div>

      <div className="card" style={{ padding: '0', overflow: 'hidden' }}>
        <div className="card-top" style={{ padding: '15px', borderBottom: '1px solid var(--border)' }}>
          <h3 style={{ margin: 0 }}>Live Opportunities</h3>
          <span className="badge ok">Scanning...</span>
        </div>
        <table className="data-table" style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              <th style={{ padding: '10px 15px' }}>Type</th>
              <th style={{ padding: '10px 15px' }}>Chain</th>
              <th style={{ padding: '10px 15px' }}>Route</th>
              <th style={{ padding: '10px 15px' }}>Req. Capital</th>
              <th style={{ padding: '10px 15px' }}>Est. Value</th>
              <th style={{ padding: '10px 15px' }}>Action</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid #1a1a2e' }}>
              <td style={{ padding: '10px 15px' }} className="bold">Atomic Arb</td>
              <td style={{ padding: '10px 15px' }}><span className="tag">Monad</span></td>
              <td style={{ padding: '10px 15px', fontSize: '12px' }}>Kuru → Ambient (MON/USDC)</td>
              <td style={{ padding: '10px 15px' }}>15,000 USDC</td>
              <td style={{ padding: '10px 15px' }} className="bold text-green">+$45.20</td>
              <td style={{ padding: '10px 15px' }}><button className="primary" style={{ padding: '4px 8px' }}>Execute</button></td>
            </tr>
            <tr>
              <td style={{ padding: '10px 15px' }} className="bold">Liquidation</td>
              <td style={{ padding: '10px 15px' }}><span className="tag">Solana</span></td>
              <td style={{ padding: '10px 15px', fontSize: '12px' }}>MarginFi (SOL Debt)</td>
              <td style={{ padding: '10px 15px' }}>450 SOL</td>
              <td style={{ padding: '10px 15px' }} className="bold text-green">+$120.50</td>
              <td style={{ padding: '10px 15px' }}><button className="primary" style={{ padding: '4px 8px' }}>Execute</button></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function BacktesterPanel({ state }) {
  const { backtest, runBacktest } = state;
  return (
    <div className="hft-panel" style={{ background: '#04070d', padding: '20px', borderRadius: '8px' }}>
      <h2 style={{ color: 'var(--monad-purple)' }}>Strategy Backtester</h2>
      
      <div className="grid-2" style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '20px' }}>
        <div className="card" style={{ padding: '20px' }}>
          <h3 style={{ margin: '0 0 15px 0' }}>Configuration</h3>
          <div className="form-group">
            <label>Strategy Type</label>
            <select style={{ width: '100%' }}>
              <option>Cross-VM Arbitrage</option>
              <option>Mean Reversion</option>
              <option>Momentum</option>
            </select>
          </div>
          <div className="form-group">
            <label>Date Range</label>
            <div style={{ display: 'flex', gap: '10px' }}>
              <input type="date" style={{ width: '100%' }} />
              <input type="date" style={{ width: '100%' }} />
            </div>
          </div>
          <div className="form-group">
            <label>Initial Capital (USDC)</label>
            <input type="number" defaultValue="10000" style={{ width: '100%' }} />
          </div>
          <button className="primary" onClick={runBacktest} disabled={backtest.running} style={{ width: '100%', marginTop: '10px' }}>
            {backtest.running ? 'Running Backtest...' : 'Run Backtest'}
          </button>
        </div>

        <div className="card" style={{ padding: '20px' }}>
          <h3 style={{ margin: '0 0 15px 0' }}>Results Dashboard</h3>
          {backtest.result ? (
            <>
              <div className="grid-2" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px', marginBottom: '20px' }}>
                <div>
                  <div className="desc">Total Return</div>
                  <div className="bold text-green">{backtest.result.return}</div>
                </div>
                <div>
                  <div className="desc">Sharpe Ratio</div>
                  <div className="bold">{backtest.result.sharpe}</div>
                </div>
                <div>
                  <div className="desc">Max Drawdown</div>
                  <div className="bold text-red" style={{ color: 'var(--danger)' }}>{backtest.result.drawdown}</div>
                </div>
                <div>
                  <div className="desc">Win Rate</div>
                  <div className="bold">{backtest.result.winRate}</div>
                </div>
              </div>
              <div className="form-group">
                <label>Equity Curve (Mock Sparkline)</label>
                <div style={{ padding: '20px 10px', background: '#0a0f1a', borderRadius: '4px', border: '1px solid var(--border)', fontFamily: 'monospace', color: 'var(--monad-purple)', letterSpacing: '2px' }}>
                  _.-~^~-._   _.-~^~-._   _.-~^~-._.-~^~-._.-~^~-._
                </div>
              </div>
            </>
          ) : (
            <div className="note" style={{ textAlign: 'center', marginTop: '40px' }}>Configure and run a backtest to view results.</div>
          )}
        </div>
      </div>
    </div>
  );
}
