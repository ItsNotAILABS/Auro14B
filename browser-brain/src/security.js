import {receipt} from './crypto.js';
export class SecurityMonitor{
  constructor(store,{maxPromptChars=12000}={}){this.store=store;this.maxPromptChars=maxPromptChars;this.previous=null}
  inspectText(text){
    const findings=[];if(typeof text!=='string')findings.push({severity:'high',code:'non_string_input'});
    else {if(text.length>this.maxPromptChars)findings.push({severity:'high',code:'prompt_too_large'});
      if(/ignore (all|previous) instructions|reveal .*system prompt|disable governance/i.test(text))findings.push({severity:'medium',code:'instruction_override_pattern'});
      if(/-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{20,}/.test(text))findings.push({severity:'critical',code:'possible_secret'});
    }return {allowed:!findings.some(x=>['critical','high'].includes(x.severity)),findings};
  }
  async record(event){const r=await receipt('security',event,this.previous);this.previous=r.hash;await this.store.put('receipts',{id:r.hash,...r});return r}
}
