export type RuntimeKind="browser"|"edge"|"python"|"julia"|"wasm"|"evm";
export type RuntimeCapability={id:string;runtime:RuntimeKind;owner:string;execution:"local"|"service-binding"|"edge"|"wallet";mutating:boolean;available:boolean;description:string};
export type EnvLike={PYTHON_ENGINE?:Fetcher;JULIA_ENGINE?:Fetcher};
export function capabilityRegistry(env:EnvLike):RuntimeCapability[]{return [
 {id:"thesis.plan",runtime:"edge",owner:"THESIS",execution:"edge",mutating:false,available:true,description:"Route missions across the polyglot envelope."},
 {id:"him.local.infer",runtime:"browser",owner:"HIM",execution:"local",mutating:false,available:true,description:"Transformers.js/WebGPU inference; private data stays in the browser."},
 {id:"him.local.documents",runtime:"browser",owner:"HIM",execution:"local",mutating:true,available:true,description:"Generate Markdown/CSV and teach local memory from user files."},
 {id:"nio.orchestrate",runtime:"python",owner:"NIO",execution:"service-binding",mutating:true,available:!!env.PYTHON_ENGINE,description:"Python/FastAPI engines, company OS, research, security and task workers."},
 {id:"mathesis.compute",runtime:"julia",owner:"MATHESIS",execution:"service-binding",mutating:false,available:!!env.JULIA_ENGINE,description:"Spectral, Monte Carlo, VaR/CVaR, optimization and gas modeling."},
 {id:"portable.wasm",runtime:"wasm",owner:"THESIS",execution:"local",mutating:false,available:true,description:"Portable sandboxed compute in browser and Worker contexts."},
 {id:"cloudflare.manage",runtime:"edge",owner:"HERMES",execution:"edge",mutating:true,available:true,description:"Governed Cloudflare control-plane operations with receipts."},
 {id:"monad.read",runtime:"evm",owner:"THESIS",execution:"edge",mutating:false,available:true,description:"Allowlisted Monad JSON-RPC reads and simulations."},
 {id:"monad.prepare",runtime:"evm",owner:"LAWBOOK",execution:"wallet",mutating:true,available:true,description:"Prepare unsigned transactions for explicit owner wallet signing."},
 {id:"solidity.enforce",runtime:"evm",owner:"CPL",execution:"wallet",mutating:true,available:true,description:"PolicyKernel, ReceiptChain, LawBook and SovereignVault enforcement."}
]};
export const capability=(env:EnvLike,id:string)=>capabilityRegistry(env).find(x=>x.id===id);
