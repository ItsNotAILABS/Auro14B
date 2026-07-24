import {receipt,sha256} from './crypto.js';
export class ResearchCouncil{
  constructor({store,graph,inference,security}){Object.assign(this,{store,graph,inference,security});this.previous=null}
  async run(objective){
    const check=this.security.inspectText(objective);await this.security.record({operation:'research',...check});if(!check.allowed)throw new Error('research objective denied');
    const roles=['sensus','corpus','mathesis','red_team','consolidator'];const findings=[];
    for(const role of roles){const result=await this.inference.generate([{role:'system',content:`You are internal research organ ${role}. Return concise evidence, uncertainty, and training-worthy conclusions.`},{role:'user',content:objective}]);findings.push({role,text:result.text})}
    const artifact={schema:'him.research.v1',id:crypto.randomUUID(),objective,findings,status:'candidate',createdAt:new Date().toISOString()};
    artifact.contentHash=await sha256(JSON.stringify(artifact));await this.store.put('research',artifact);
    const r=await receipt('research',artifact,this.previous);this.previous=r.hash;await this.store.put('receipts',{id:r.hash,...r});return {...artifact,receipt:r};
  }
  async nominateForTraining(research,{approved=false}={}){
    const row={id:`train:${research.contentHash}`,sourceHash:research.contentHash,text:research.findings.map(x=>`<${x.role}> ${x.text}`).join('\n'),approved,status:approved?'approved':'awaiting_human_approval'};
    await this.store.put('training',row);return row;
  }
  async exportApprovedTraining(){const rows=await this.store.getAll('training');return rows.filter(x=>x.approved).map(x=>({text:x.text,source_hash:x.sourceHash}))}
}
