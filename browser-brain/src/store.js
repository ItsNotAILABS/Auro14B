export class BrainStore{
  constructor(name='him-browser-brain'){this.name=name;this.memory=new Map();this.db=null}
  async open(){
    if(typeof indexedDB==='undefined') return this;
    this.db=await new Promise((resolve,reject)=>{const r=indexedDB.open(this.name,1);r.onupgradeneeded=()=>{
      const d=r.result;for(const n of ['memory','graph','receipts','research','training'])if(!d.objectStoreNames.contains(n))d.createObjectStore(n,{keyPath:'id'});
    };r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error)});return this;
  }
  async put(bucket,value){
    const row={...value,id:value.id||crypto.randomUUID(),updatedAt:new Date().toISOString()};
    if(!this.db){this.memory.set(`${bucket}:${row.id}`,row);return row}
    await new Promise((resolve,reject)=>{const r=this.db.transaction(bucket,'readwrite').objectStore(bucket).put(row);r.onsuccess=resolve;r.onerror=()=>reject(r.error)});return row;
  }
  async getAll(bucket){
    if(!this.db)return [...this.memory.entries()].filter(([k])=>k.startsWith(bucket+':')).map(([,v])=>v);
    return new Promise((resolve,reject)=>{const r=this.db.transaction(bucket).objectStore(bucket).getAll();r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error)});
  }
  async clear(bucket){
    if(!this.db){for(const k of this.memory.keys())if(k.startsWith(bucket+':'))this.memory.delete(k);return}
    await new Promise((resolve,reject)=>{const r=this.db.transaction(bucket,'readwrite').objectStore(bucket).clear();r.onsuccess=resolve;r.onerror=()=>reject(r.error)});
  }
}
