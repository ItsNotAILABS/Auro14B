import crypto from 'node:crypto';

export function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.keys(value).sort().map(k => `${JSON.stringify(k)}:${canonical(value[k])}`).join(',')}}`;
  return JSON.stringify(value);
}

export function sha256(value) {
  return crypto.createHash('sha256').update(typeof value === 'string' ? value : canonical(value)).digest('hex');
}

export function hmac(value, key) {
  return crypto.createHmac('sha256', key).update(canonical(value)).digest('hex');
}

const SECRET_PATTERNS = [
  /\b(?:sk|pk|api)[-_][A-Za-z0-9_-]{16,}\b/g,
  /\bBearer\s+[A-Za-z0-9._~+/=-]{12,}\b/gi,
  /\b(?:password|passwd|secret|token)\s*[:=]\s*\S+/gi,
  /\b\d{3}-\d{2}-\d{4}\b/g,
  /\b(?:\d[ -]*?){13,19}\b/g,
];
const INJECTION_PATTERNS = [
  /ignore\s+(?:all|any|the)?\s*(?:previous|prior|system)\s+instructions?/i,
  /reveal\s+(?:the\s+)?(?:system|developer)\s+prompt/i,
  /execute\s+(?:this|the following)\s+(?:code|command)/i,
  /you\s+are\s+now\s+/i,
];

export function admitMemory(record, { allowedOrigins = [], encryption = 'required' } = {}) {
  if (!record || typeof record !== 'object') throw new Error('memory record required');
  const origin = String(record.origin || '');
  if (!origin || (allowedOrigins.length && !allowedOrigins.includes(origin))) throw new Error('untrusted provenance origin');
  if (!record.sourceUrl || !record.fetchedAt || !record.contentSha256) throw new Error('memory provenance incomplete');
  if (encryption === 'required' && record.encrypted !== true) throw new Error('unencrypted browser memory rejected');
  let text = String(record.text || '');
  const redactions = [];
  for (const pattern of SECRET_PATTERNS) text = text.replace(pattern, match => { redactions.push(sha256(match)); return '[REDACTED]'; });
  const injectionSignals = INJECTION_PATTERNS.filter(pattern => pattern.test(text)).map(pattern => pattern.source);
  return {
    accepted: injectionSignals.length === 0,
    record: { ...record, text, untrustedEvidence: true, instructionAuthority: false },
    redactionHashes: redactions,
    injectionSignals,
    admissionSha256: sha256({ origin, sourceUrl: record.sourceUrl, fetchedAt: record.fetchedAt, contentSha256: record.contentSha256, text }),
  };
}

export function verifyServerApproval(grant, action, key, now = Date.now()) {
  if (!grant || !action || !key) return false;
  const unsigned = { ...grant };
  const signature = unsigned.signature;
  delete unsigned.signature;
  if (!signature || !crypto.timingSafeEqual(Buffer.from(signature), Buffer.from(hmac(unsigned, key)))) return false;
  if (unsigned.expiresAt <= now || unsigned.notBefore > now) return false;
  if (unsigned.actionSha256 !== sha256(action)) return false;
  if (!unsigned.approvalId || !unsigned.subject || unsigned.authority !== 'server') return false;
  return true;
}

export function signTaskReceipt(task, { peerId, leaseId, previousSha256 = null, key, now = Date.now() }) {
  if (!peerId || !leaseId || !key) throw new Error('authenticated peer, durable lease, and key required');
  const receipt = {
    schema: 'auro.browser.mesh.task.v1',
    taskSha256: sha256(task),
    peerId,
    leaseId,
    previousSha256,
    createdAt: now,
  };
  receipt.receiptSha256 = sha256(receipt);
  receipt.signature = hmac(receipt, key);
  return receipt;
}

export function verifyTaskReceipt(receipt, key) {
  if (!receipt || !key) return false;
  const signature = receipt.signature;
  const unsigned = { ...receipt };
  delete unsigned.signature;
  if (unsigned.receiptSha256 !== sha256({ ...unsigned, receiptSha256: undefined })) {
    const probe = { ...unsigned }; delete probe.receiptSha256;
    if (unsigned.receiptSha256 !== sha256(probe)) return false;
  }
  return Boolean(signature && crypto.timingSafeEqual(Buffer.from(signature), Buffer.from(hmac(unsigned, key))));
}
