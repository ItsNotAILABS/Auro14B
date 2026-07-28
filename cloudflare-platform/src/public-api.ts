type PublicEnv={AI:Ai;WORKERS_AI_MODEL:string;MONAD_RPC_URL:string;OPERATOR_TOKEN:string;PUBLIC_CHAT_TOKEN?:string;PUBLIC_CHAT_REQUESTS_PER_MINUTE?:string};
type ReceiptStub={append:(kind:string,value:unknown)=>Promise<unknown>;consume:(limit:number,windowMs:number)=>Promise<{allowed:boolean;remaining:number;reset_at:number}>};

const json=(body:unknown,status=200,extra:Record<string,string>={})=>new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json","cache-control":"no-store","x-content-type-options":"nosniff",...extra}});
const safeEqual=(a:string,b:string)=>{if(a.length!==b.length)return false;let out=0;for(let i=0;i<a.length;i++)out|=a.charCodeAt(i)^b.charCodeAt(i);return out===0};

async function rpc(env:PublicEnv,method:string,params:unknown[]=[]){
 const response=await fetch(env.MONAD_RPC_URL,{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({jsonrpc:"2.0",id:crypto.randomUUID(),method,params})});
 const payload=await response.json() as {result?:unknown;error?:unknown};
 if(!response.ok||payload.error)throw new Error(JSON.stringify(payload.error||{status:response.status}));
 return payload.result;
}

function hexToDecimal(value:unknown){if(typeof value!=="string"||!value.startsWith("0x"))return value;return BigInt(value).toString(10)}
function chatAuthorized(request:Request,env:PublicEnv){const expected=env.PUBLIC_CHAT_TOKEN||env.OPERATOR_TOKEN;const supplied=request.headers.get("authorization")||"";return Boolean(expected&&safeEqual(supplied,`Bearer ${expected}`))}

export async function handlePublicApi(request:Request,env:PublicEnv,stub:ReceiptStub):Promise<Response|null>{
 const url=new URL(request.url);
 if(request.method==="GET"&&(url.pathname==="/health"||url.pathname==="/api/health"))return json({ok:true,status:"ok",service:"him-cloudflare-platform",apps:["web3","foundry","operator"],rpc_configured:Boolean(env.MONAD_RPC_URL),model:{model_id:env.WORKERS_AI_MODEL,provider:"cloudflare-workers-ai",claim:"hosted compatibility lane; not an AURO-trained checkpoint"}});
 if(request.method==="GET"&&url.pathname==="/api/him")return json({ok:true,name:"HIM",architecture:"shared-cloudflare-runtime",apps:{web3:"/web3/",foundry:"/foundry/",operator:"/operator/"},security:"secrets remain in Worker bindings"});
 if(request.method==="GET"&&url.pathname==="/api/chain/block-number"){const raw=await rpc(env,"eth_blockNumber");const receipt=await stub.append("chain_block_number",{raw});return json({ok:true,blockNumber:hexToDecimal(raw),raw,via:"cloudflare-json-rpc",receipt})}
 if(request.method==="GET"&&url.pathname.startsWith("/api/chain/balance/")){const address=decodeURIComponent(url.pathname.slice("/api/chain/balance/".length));if(!/^0x[a-fA-F0-9]{40}$/.test(address))return json({ok:false,error:"invalid address"},400);const raw=await rpc(env,"eth_getBalance",[address,"latest"]);const receipt=await stub.append("chain_balance",{address,raw});return json({ok:true,address,balanceWei:hexToDecimal(raw),raw,via:"cloudflare-json-rpc",receipt})}
 if(request.method==="GET"&&url.pathname==="/api/chain/block"){const tag=url.searchParams.get("tag")||"latest";const normalized=/^\d+$/.test(tag)?`0x${BigInt(tag).toString(16)}`:tag;const block=await rpc(env,"eth_getBlockByNumber",[normalized,false]);const receipt=await stub.append("chain_block",{tag:normalized});return json({ok:true,block,via:"cloudflare-json-rpc",receipt})}
 if(request.method==="GET"&&url.pathname==="/v1/models")return json({object:"list",data:[{id:"auro-cloudflare",object:"model",owned_by:"ItsNotAILABS",provider_model:env.WORKERS_AI_MODEL,claim:"hosted Workers AI compatibility lane; not Medina/AURO-trained weights"}]});
 if(request.method==="POST"&&url.pathname==="/v1/chat/completions"){
  if(!chatAuthorized(request,env))return json({error:"authenticated chat token required"},401,{"www-authenticate":"Bearer"});
  const sessionId=(request.headers.get("x-session-id")||"").trim();
  if(!/^[A-Za-z0-9_-]{16,128}$/.test(sessionId))return json({error:"x-session-id must be a stable 16..128 character identity-bound identifier"},400);
  const limit=Math.max(1,Math.min(600,Number(env.PUBLIC_CHAT_REQUESTS_PER_MINUTE)||20));
  const quota=await stub.consume(limit,60_000);
  if(!quota.allowed)return json({error:"rate limit exceeded",reset_at:quota.reset_at},429,{"retry-after":String(Math.max(1,Math.ceil((quota.reset_at-Date.now())/1000)))});
  const body=await request.json() as {messages?:Array<{role?:string;content?:string}>;max_tokens?:number;temperature?:number};
  if(!Array.isArray(body.messages)||body.messages.length===0)return json({error:"messages are required"},400);
  const messages=body.messages.map(item=>({role:item.role||"user",content:String(item.content||"")}));
  const rawTemperature=Number(body.temperature);const temperature=Number.isFinite(rawTemperature)?Math.min(2,Math.max(0,rawTemperature)):0.7;
  const output=await env.AI.run(env.WORKERS_AI_MODEL as keyof AiModels,{messages,max_tokens:Math.min(2048,Math.max(1,Number(body.max_tokens)||256)),temperature} as never) as {response?:string};
  const content=output.response||"";
  const receipt=await stub.append("foundry_chat",{session_id:sessionId,message_count:messages.length,response_length:content.length,model:env.WORKERS_AI_MODEL,provider:"cloudflare-workers-ai"});
  return json({id:crypto.randomUUID(),object:"chat.completion",created:Math.floor(Date.now()/1000),model:"auro-cloudflare",provider:"cloudflare-workers-ai",claim:"hosted compatibility lane; not an AURO-trained checkpoint",choices:[{index:0,message:{role:"assistant",content},finish_reason:"stop"}],quota,receipt});
 }
 return null;
}
