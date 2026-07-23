import { randomUUID } from 'node:crypto';

export function createScheduler({ receipts, limit = 1000 } = {}) {
  const jobs = new Map();
  const idempotency = new Map();

  return {
    submit({ kind, payload, idempotencyKey } = {}) {
      if (idempotencyKey && idempotency.has(idempotencyKey)) {
        return jobs.get(idempotency.get(idempotencyKey));
      }
      if (typeof kind !== 'string' || kind.length === 0) {
        throw new Error('job kind is required');
      }
      const job = {
        id: randomUUID(),
        kind,
        payload: payload ?? null,
        status: 'accepted',
        created_at: new Date().toISOString(),
      };
      jobs.set(job.id, job);
      if (idempotencyKey) idempotency.set(idempotencyKey, job.id);
      while (jobs.size > limit) {
        const first = jobs.keys().next().value;
        jobs.delete(first);
      }
      receipts?.append('job.accepted', { job_id: job.id, kind: job.kind });
      return job;
    },
    list() {
      return [...jobs.values()].reverse();
    },
  };
}
