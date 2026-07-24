const words=text=>new Set(String(text||'').toLowerCase().match(/[a-z0-9_]{2,}/g)||[]);
export class MemoryEngine{
  constructor(store){this.store=store}
  async remember({text,kind='knowledge',source='local',importance=.5,meta={}}){if(!text?.trim())throw new Error('memory text required');return this.store.put('memory',{text:text.trim(),kind,source,importance:Math.max(0,Math.min(1,importance)),meta,createdAt:new Date().toISOString()})}
  async recall(query,{limit=8}={}){const q=words(query),now=Date.now(),rows=await this.store.getAll('memory');return rows.map(row=>{const tokens=words(`${row.text} ${JSON.stringify(row.meta||{})}`);let overlap=0;for(const token of q)if(tokens.has(token))overlap++;const ageDays=Math.max(0,(now-Date.parse(row.updatedAt||row.createdAt||0))/86400000),recency=1/(1+ageDays/30);return {...row,score:(q.size?overlap/q.size:0)*.65+Number(row.importance||.5)*.25+recency*.1}}).sort((a,b)=>b.score-a.score).slice(0,limit)}
  async consolidate(){const rows=await this.store.getAll('memory'),seen=new Set(),unique=[];for(const row of rows.sort((a,b)=>Date.parse(b.updatedAt)-Date.parse(a.updatedAt))){const key=String(row.text).trim().toLowerCase();if(!seen.has(key)){seen.add(key);unique.push(row)}}return {before:rows.length,after:unique.length,duplicates:rows.length-unique.length}}
}
