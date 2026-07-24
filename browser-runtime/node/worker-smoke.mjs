import { OutsideWorkerPool } from './worker-pool.mjs';
const pool=new OutsideWorkerPool(2);
const results=await Promise.all([pool.run('process',{text:'Auro'}),pool.run('process',{text:'MESIE Sovereign'})]);
if(results.some(item=>!item.ok||item.result.sha256.length!==64))process.exitCode=1;
else console.log(JSON.stringify({ok:true,jobs:results.length,plane:'node-worker_threads'}));
await pool.close();
