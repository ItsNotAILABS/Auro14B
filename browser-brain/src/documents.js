import {sha256} from './crypto.js';
export class DocumentEngine{
  async markdown({title,sections,provenance=[]}){const body=[`# ${title}`,...sections.map(x=>`\n## ${x.heading}\n\n${x.body}`),`\n## Provenance\n\n${provenance.map(x=>`- ${x}`).join('\n')||'- Local HIM browser brain'}`].join('\n');return this.bundle(`${slug(title)}.md`,body,'text/markdown')}
  async csv(name,rows){if(!rows.length)throw new Error('rows required');const keys=Object.keys(rows[0]);const esc=v=>`"${String(v??'').replaceAll('"','""')}"`;return this.bundle(`${slug(name)}.csv`,[keys,...rows.map(r=>keys.map(k=>r[k]))].map(r=>r.map(esc).join(',')).join('\n'),'text/csv')}
  async bundle(name,text,mime){return {schema:'him.document.v1',name,mime,text,bytes:new TextEncoder().encode(text).length,sha256:await sha256(text),createdAt:new Date().toISOString()}}
  download(artifact){const url=URL.createObjectURL(new Blob([artifact.text],{type:artifact.mime}));const a=Object.assign(globalThis.document.createElement('a'),{href:url,download:artifact.name});a.click();URL.revokeObjectURL(url)}
}
const slug=s=>s.toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'')||'document';
