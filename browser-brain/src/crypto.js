const encoder = new TextEncoder();
export function canonical(value){
  if(Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if(value&&typeof value==='object') return `{${Object.keys(value).sort().map(k=>`${JSON.stringify(k)}:${canonical(value[k])}`).join(',')}}`;
  return JSON.stringify(value);
}
export async function sha256(value){
  const bytes=typeof value==='string'?encoder.encode(value):value;
  const hash=await crypto.subtle.digest('SHA-256',bytes);
  return [...new Uint8Array(hash)].map(x=>x.toString(16).padStart(2,'0')).join('');
}
export async function receipt(kind,payload,previous=null){
  const core={schema:'him.browser.receipt.v1',kind,at:new Date().toISOString(),previous,payload};
  return {...core,hash:await sha256(canonical(core))};
}
