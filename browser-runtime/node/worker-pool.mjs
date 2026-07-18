import { Worker } from 'node:worker_threads';
import { availableParallelism } from 'node:os';

export class OutsideWorkerPool {
  constructor(size = Math.min(4, Math.max(1, availableParallelism() - 1))) {
    this.queue=[]; this.pending=new Map();
    this.workers=Array.from({length:size},()=>this.#create());
  }
  #create(){ const worker=new Worker(new URL('./task-worker.mjs',import.meta.url)); worker.busy=false; worker.on('message',(data)=>{const job=this.pending.get(data.id); if(job){clearTimeout(job.timer);this.pending.delete(data.id);data.ok?job.resolve(data):job.reject(new Error(data.error));}worker.busy=false;this.#drain();}); return worker; }
  run(task,payload,timeoutMs=30000){return new Promise((resolve,reject)=>{this.queue.push({id:crypto.randomUUID(),task,payload,timeoutMs,resolve,reject});this.#drain();});}
  #drain(){const worker=this.workers.find(x=>!x.busy),job=this.queue.shift();if(!worker||!job){if(job)this.queue.unshift(job);return;}worker.busy=true;job.timer=setTimeout(()=>{this.pending.delete(job.id);job.reject(new Error(`Outside worker timed out: ${job.task}`));},job.timeoutMs);this.pending.set(job.id,job);worker.postMessage({id:job.id,task:job.task,payload:job.payload});this.#drain();}
  async close(){await Promise.all(this.workers.map(worker=>worker.terminate()));}
}
