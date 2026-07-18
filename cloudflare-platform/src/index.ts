import {DurableObject} from "cloudflare:workers";
import {UI} from "./ui";
import {capabilityRegistry} from "./capabilities";
import {monadRead,prepareUnsignedTransaction} from "./monad";
import {isReadOnly,normalizeProposal,safeEqual,sha256,signGrant,type Proposal} from "./policy";

type Env={AI:Ai;SESSIONS:DurableObjectNamespace<HIMSession>;CLOUDFLARE_API_TOKEN?:string;CLOUDFLARE_ACCOUNT_ID?:string;OPERATOR_TOKEN:string;EXECUTION_SECRET:string;WORKERS_AI_MODEL:string;EXECUTION_TTL_SECONDS:string;MONAD_RPC_URL:string;PYTHON_ENGINE?:Fetcher;JULIA_ENGINE?:Fetcher};
const json=(body:unknown,status=200)=>new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json","cache-control":"no-store","x-content-type-options":"nosniff"}});
const session=(r:Request,e:Env)=>e.SESSIONS.get(e.SESSIONS.idFromName(r.headers.get("x-session-id")||"anonymous"));

export class HIMSession extends DurableObject<Env>{
 async append(kind:string,value:unknown){const ts=Date.now(),prev=(await this.ctx.storage.get<string>("head"))||"0".repeat(64),hash=await sha256(JSON.stringify({ts,kind,value,prev}));await this.ctx.storage.put(`receipt:${ts}:${crypto.randomUUID()}`,{ts,kind,value,prev,hash});await this.ctx.storage.put("head",hash);return {ts,hash,previous_hash:prev}}
 async history(){const rows=await this.ctx.storage.list({prefix:"receipt:",reverse:true,limit:50});return {head:await this.ctx.storage.get("head"),receipts:[...rows.values()]}}
}

async function cloudflare(env:Env,p:Proposal,sessionToken?:string){
 const token=sessionToken||env.CLOUDFLARE_API_TOKEN;if(!token)throw new Error("Connect a session API token in Settings or configure the managed Worker secret");
 const url=new URL(`https://api.cloudflare.com/client/v4${p.path}`);for(const [k,v] of Object.entries(p.query||{}))url.searchParams.set(k,String(v));
 return fetch(url,{method:p.method,headers:{authorization:`Bearer ${token}`,"content-type":"application/json"},body:p.body===undefined?undefined:JSON.stringify(p.body)}).then(async r=>({status:r.status,ok:r.ok,data:await r.json()}));
}
async function plan(env:Env,message:string){
 const prompt=`You are HIM's governed Cloudflare operator. Return strict JSON with answer and optional proposal. A proposal is {method,path,query?,body?} using a relative https://api.cloudflare.com/client/v4 path. Prefer read-only discovery. Never claim execution. For mutation, explain impact and propose exactly one bounded operation. User: ${message}`;
 const out=await env.AI.run(env.WORKERS_AI_MODEL as keyof AiModels,{messages:[{role:"user",content:prompt}],response_format:{type:"json_object"}} as never) as {response?:string};
 try{return JSON.parse(out.response||"{}")}catch{return {answer:out.response||"Unable to produce a plan."}}
}

export default {async fetch(request:Request,env:Env):Promise<Response>{
 try{const url=new URL(request.url);if(request.method==="GET"&&url.pathname==="/")return new Response(UI,{headers:{"content-type":"text/html;charset=utf-8","content-security-policy":"default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'","referrer-policy":"no-referrer"}});
  if(request.method==="GET"&&url.pathname==="/api/health")return json({ok:true,service:"him-cloudflare-platform",authorization:"read/write-separated"});
  const bearer=request.headers.get("authorization")||"";if(!env.OPERATOR_TOKEN||!safeEqual(bearer,`Bearer ${env.OPERATOR_TOKEN}`))return json({error:"operator authentication required"},401);
  const stub=session(request,env);
  if(request.method==="GET"&&url.pathname==="/api/capabilities")return json({schema:"thesis.polyglot.capabilities.v1",capabilities:capabilityRegistry(env),setup:{cloudflare:["session-token","managed-worker-secret"],monad:{chain_id:143,signing:"browser-wallet-only"}}});
  if(request.method==="POST"&&url.pathname==="/api/monad/read"){const body=await request.json();const result=await monadRead(env.MONAD_RPC_URL,body);const receipt=await stub.append("monad_read",{request:body,status:result.status});return json({...result,receipt},result.ok?200:502)}
  if(request.method==="POST"&&url.pathname==="/api/monad/prepare"){const transaction=prepareUnsignedTransaction((await request.json() as {transaction?:unknown}).transaction);const receipt=await stub.append("monad_unsigned_preparation",{chainId:143,to:transaction.to});return json({transaction,receipt})}
  if(request.method==="POST"&&url.pathname==="/api/chat"){const body=await request.json() as {message?:string};if(!body.message||body.message.length>12000)return json({error:"message must be 1..12000 characters"},400);const result=await plan(env,body.message);if(result.proposal)result.proposal=normalizeProposal(result.proposal);await stub.append("plan",{message:body.message,result});return json(result)}
  if(request.method==="POST"&&url.pathname==="/api/grant"){const p=normalizeProposal((await request.json() as {proposal:unknown}).proposal);if(isReadOnly(p))return json({error:"read operations do not need an execution grant"},400);const expires=Date.now()+Math.min(900,Number(env.EXECUTION_TTL_SECONDS)||300)*1000;const digest=await sha256(JSON.stringify(p));const payload=`${digest}.${expires}`;return json({grant:`${payload}.${await signGrant(env.EXECUTION_SECRET,payload)}`,expires})}
  if(request.method==="POST"&&url.pathname==="/api/execute"){const p=normalizeProposal((await request.json() as {proposal:unknown}).proposal);if(!isReadOnly(p)){const grant=request.headers.get("x-execution-grant")||"",parts=grant.split(".");if(parts.length!==3||Number(parts[1])<Date.now()||parts[0]!==await sha256(JSON.stringify(p))||!safeEqual(parts[2],await signGrant(env.EXECUTION_SECRET,`${parts[0]}.${parts[1]}`)))return json({error:"valid unexpired execution grant required"},403)}const result=await cloudflare(env,p,request.headers.get("x-cloudflare-token")||undefined);const receipt=await stub.append("cloudflare_api",{proposal:p,status:result.status,ok:result.ok});return json({...result,receipt},result.ok?200:502)}
  if(request.method==="GET"&&url.pathname==="/api/receipts")return json(await stub.history());return json({error:"not found"},404)
 }catch(e){return json({error:e instanceof Error?e.message:"request failed"},400)}}};
