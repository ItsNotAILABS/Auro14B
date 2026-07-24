export class BrowserWorkerPool {
  constructor(size = Math.min(4, Math.max(1, navigator.hardwareConcurrency || 2))) {
    this.workers = Array.from({ length: size }, () => new Worker(new URL('./task-worker.js', import.meta.url), { type: 'module' }));
    this.queue = [];
    this.pending = new Map();
    this.workers.forEach((worker) => {
      worker.busy = false;
      worker.onmessage = ({ data }) => {
        const job = this.pending.get(data.id);
        if (job) {
          clearTimeout(job.timer);
          this.pending.delete(data.id);
          data.ok ? job.resolve(data) : job.reject(new Error(data.error));
        }
        worker.busy = false;
        this.#drain();
      };
    });
  }
  run(task, payload, timeoutMs = 30000) {
    return new Promise((resolve, reject) => {
      this.queue.push({ id: crypto.randomUUID(), task, payload, timeoutMs, resolve, reject });
      this.#drain();
    });
  }
  #drain() {
    const worker = this.workers.find((item) => !item.busy);
    const job = this.queue.shift();
    if (!worker || !job) { if (job) this.queue.unshift(job); return; }
    worker.busy = true;
    job.timer = setTimeout(() => { this.pending.delete(job.id); worker.terminate(); job.reject(new Error(`Worker task timed out: ${job.task}`)); }, job.timeoutMs);
    this.pending.set(job.id, job);
    worker.postMessage({ id: job.id, task: job.task, payload: job.payload });
    this.#drain();
  }
  close() { this.workers.forEach((worker) => worker.terminate()); }
}
