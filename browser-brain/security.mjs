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

function safeEqualHex(a, b) {
  if (typeof a !== 'string' || typeof b !== 'string' || !/^[0-9a-f]{64}$/i.test(a) || !/^[0-9a-f]{64}$/i.test(b)) return false;
  return crypto.timingSafeEqual(Buffer.from(a, 'hex'), Buffer.from(b, 'hex'));
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
  const originalText = String(record.text || '');
  if (!safeEqualHex(String(record.contentSha256), sha256(originalText))) throw new Error('memory content digest mismatch');
  let text = originalText;
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
  if (!safeEqualHex(signature, hmac(unsigned, key))) return false;
  if (Number(unsigned.expiresAt) <= now || Number(unsigned.notBefore) > now) return false;
  if (!safeEqualHex(String(unsigned.actionSha256 || ''), sha256(action))) return false;
  return Boolean(unsigned.approvalId && unsigned.subject && unsigned.authority === 'server');
}

export function signTaskReceipt(task, { peerId, leaseId, previousSha256 = null, key, now = Date.now() }) {
  if (!peerId || !leaseId || !key) throw new Error('authenticated peer, durable lease, and key required');
  const receipt = { schema: 'auro.browser.mesh.task.v1', taskSha256: sha256(task), peerId, leaseId, previousSha256, createdAt: now };
  receipt.receiptSha256 = sha256(receipt);
  receipt.signature = hmac(receipt, key);
  return receipt;
}

export function verifyTaskReceipt(receipt, key) {
  if (!receipt || !key) return false;
  const signature = receipt.signature;
  const unsigned = { ...receipt };
  delete unsigned.signature;
  const receiptSha256 = unsigned.receiptSha256;
  delete unsigned.receiptSha256;
  if (!safeEqualHex(receiptSha256, sha256(unsigned))) return false;
  const signedBody = { ...unsigned, receiptSha256 };
  return safeEqualHex(signature, hmac(signedBody, key));
}
