import React, { FormEvent, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { useAgent } from 'agents/react';
import { useAgentChat } from '@cloudflare/ai-chat/react';
import './style.css';

function App() {
  const [input, setInput] = useState('Inspect my Cloudflare account and propose the highest-value improvement. Do not change anything.');
  const [approved, setApproved] = useState(false);
  const agent = useAgent({ agent: 'AuroPlatform', name: 'operator' });
  const { messages, sendMessage, status } = useAgentChat({ agent });
  const busy = status === 'submitted' || status === 'streaming';
  function submit(event: FormEvent) {
    event.preventDefault();
    const text = `${approved ? 'OPERATOR_APPROVED\n' : ''}${input}`.trim();
    if (!text) return;
    sendMessage({ text }); setInput(''); setApproved(false);
  }
  return <main>
    <header><div><p className="eyebrow">AURO / MESIE / SOVEREIGN</p><h1>Cloudflare operator</h1><p className="lede">A durable agent that can inspect, plan, build, browse, and, only with explicit approval, manage Cloudflare.</p></div><span className="live">EDGE ACTIVE</span></header>
    <section className="rail"><div><b>Workers AI</b><span>native inference</span></div><div><b>Think</b><span>durable memory</span></div><div><b>API MCP</b><span>search + execute</span></div><div><b>Code Mode</b><span>isolated workers</span></div><div><b>Browser Run</b><span>human handoff</span></div></section>
    <section className="workspace">
      <div className="conversation">
        {messages.length === 0 && <div className="empty"><p>Start with inspection.</p><small>Ask Auro to search the Cloudflare API, inventory resources, or design a deployment. Changes remain locked until approval is enabled both in Worker configuration and for this turn.</small></div>}
        {messages.map(message => <article className={message.role} key={message.id}><strong>{message.role === 'user' ? 'OPERATOR' : 'AURO'}</strong><div>{message.parts.map((part, index) => part.type === 'text' ? <p key={index}>{part.text}</p> : <pre key={index}>{JSON.stringify(part, null, 2)}</pre>)}</div></article>)}
      </div>
      <aside><p className="eyebrow">CONTROL PLANE</p><h2>One platform, bounded authority.</h2><ul><li>Cloudflare API discovery</li><li>Worker and resource plans</li><li>Durable files and receipts</li><li>Dynamic extension runtime</li><li>Headless browser actions</li><li>Logs and sampled traces</li></ul><p className="warning">Approval in this UI does not override the server policy. Set <code>ALLOW_CLOUDFLARE_MUTATIONS=true</code> deliberately.</p></aside>
    </section>
    <form onSubmit={submit}><textarea value={input} onChange={event => setInput(event.target.value)} placeholder="Tell Auro what outcome you want..."/><label><input type="checkbox" checked={approved} onChange={event => setApproved(event.target.checked)}/> Mark this exact turn OPERATOR_APPROVED</label><button disabled={busy}>{busy ? 'Auro is working...' : 'Send to durable agent ->'}</button></form>
    <footer>Cloudflare-native / secrets remain in Worker bindings / destructive work requires human approval</footer>
  </main>;
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
