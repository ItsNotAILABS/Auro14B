export type Proposal={method:string;path:string;query?:Record<string,string>;body?:unknown};
const READ=new Set(["GET","HEAD","OPTIONS"]);
const BLOCKED=[/\/user\/tokens/i,/\/accounts\/[^/]+\/members/i,/\/zones\/[^/]+\/purge_cache/i];
export function normalizeProposal(value:unknown):Proposal{
 if(!value||typeof value!=="object")throw new Error("proposal must be an object");
 const v=value as Record<string,unknown>, method=String(v.method||"").toUpperCase(), path=String(v.path||"");
 if(!/^(GET|HEAD|OPTIONS|POST|PUT|PATCH|DELETE)$/.test(method))throw new Error("unsupported method");
 if(!path.startsWith("/")||path.includes("://")||path.includes(".."))throw new Error("path must be a relative Cloudflare API path");
 if(BLOCKED.some(x=>x.test(path)))throw new Error("operation is blocked by platform policy");
 return {method,path,query:(v.query&&typeof v.query==="object"?v.query:undefined) as Record<string,string>|undefined,body:v.body};
}
export const isReadOnly=(p:Proposal)=>READ.has(p.method);
export async function sha256(value:string){return [...new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(value)))].map(x=>x.toString(16).padStart(2,"0")).join("");}
export async function signGrant(secret:string,payload:string){const key=await crypto.subtle.importKey("raw",new TextEncoder().encode(secret),{name:"HMAC",hash:"SHA-256"},false,["sign"]);return btoa(String.fromCharCode(...new Uint8Array(await crypto.subtle.sign("HMAC",key,new TextEncoder().encode(payload))))).replaceAll("+","-").replaceAll("/","_").replaceAll("=","");}
export const safeEqual=(a:string,b:string)=>{if(a.length!==b.length)return false;let n=0;for(let i=0;i<a.length;i++)n|=a.charCodeAt(i)^b.charCodeAt(i);return n===0};
