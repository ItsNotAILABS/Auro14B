import http from "node:http";
import https from "node:https";
import dns from "node:dns/promises";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const PORT = Number(process.env.PORT || 8788);
const SECRET = process.env.RELAY_EGRESS_SECRET || "";
const MAX_BYTES = Math.min(Number(process.env.MAX_RESPONSE_BYTES || 5_000_000), 10_000_000);
const MAX_REDIRECTS = Math.min(Number(process.env.MAX_REDIRECTS || 5), 10);
const NONCE_TTL_MS = Math.min(Math.max(Number(process.env.RELAY_EGRESS_NONCE_TTL_MS || 60_000), 10_000), 300_000);
const NONCE_STORE_PATH = process.env.RELAY_EGRESS_NONCE_STORE || "/tmp/nexus-relay-egress/nonces.jsonl";

function blocked(ip) {
  if (!ip) return true;
  const s = ip.toLowerCase();
  if (s === "::" || s === "::1" || s.startsWith("fc") || s.startsWith("fd") || /^fe[89ab]/.test(s) || s.startsWith("ff") || s.startsWith("2001:db8:")) return true;
  if (s.startsWith("::ffff:")) return blocked(s.slice(7));
  const p = s.split(".").map(Number);
  if (p.length !== 4 || p.some((x) => !Number.isInteger(x) || x < 0 || x > 255)) return false;
  const [a,b] = p;
  return a === 0 || a === 10 || a === 127 || (a === 100 && b >= 64 && b <= 127) || (a === 169 && b === 254) || (a === 172 && b >= 16 && b <= 31) || (a === 192 && (b === 0 || b === 168)) || (a === 198 && (b === 18 || b === 19 || b === 51)) || (a === 203 && b === 0) || a >= 224;
}

function validUrl(raw) {
  const url = new URL(raw);
  if (!["http:", "https:"].includes(url.protocol)) throw new Error("unsupported protocol");
  if (!url.hostname || url.hostname === "localhost" || url.hostname.endsWith(".local") || blocked(url.hostname)) throw new Error("blocked target");
  url.username = ""; url.password = ""; url.hash = "";
  return url;
}

function safeEqual(a, b) {
  const x = Buffer.from(a || "", "hex"); const y = Buffer.from(b || "", "hex");
  return x.length === y.length && crypto.timingSafeEqual(x, y);
}

export class PersistentNonceStore {
  constructor(filePath = NONCE_STORE_PATH, ttlMs = NONCE_TTL_MS) {
    this.filePath = filePath;
    this.ttlMs = ttlMs;
    this.active = new Map();
    this._load();
  }

  _load(now = Date.now()) {
    this.active.clear();
    if (!fs.existsSync(this.filePath)) return;
    for (const line of fs.readFileSync(this.filePath, "utf8").split(/\r?\n/)) {
      if (!line.trim()) continue;
      try {
        const row = JSON.parse(line);
        const timestamp = Number(row.timestamp);
        if (row.nonce && now - timestamp <= this.ttlMs) this.active.set(String(row.nonce), timestamp);
      } catch {}
    }
    this._compact(now);
  }

  _compact(now = Date.now()) {
    for (const [nonce, timestamp] of this.active.entries()) {
      if (now - timestamp > this.ttlMs) this.active.delete(nonce);
    }
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
    const body = [...this.active.entries()]
      .sort((a, b) => a[1] - b[1])
      .map(([nonce, timestamp]) => JSON.stringify({ nonce, timestamp }))
      .join("\n");
    const temporary = `${this.filePath}.${process.pid}.tmp`;
    fs.writeFileSync(temporary, body ? `${body}\n` : "", { mode: 0o600 });
    fs.renameSync(temporary, this.filePath);
  }

  claim(nonce, timestamp, now = Date.now()) {
    if (!/^[A-Za-z0-9_-]{24,128}$/.test(String(nonce || ""))) throw new Error("invalid egress nonce");
    this._compact(now);
    if (this.active.has(nonce)) throw new Error("replayed egress request");
    this.active.set(nonce, Number(timestamp));
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
    fs.appendFileSync(this.filePath, `${JSON.stringify({ nonce, timestamp: Number(timestamp) })}\n`, { mode: 0o600 });
    return true;
  }
}

export function authenticateRequest(headers, target, options = {}) {
  const secret = options.secret ?? SECRET;
  const now = Number(options.now ?? Date.now());
  const nonceStore = options.nonceStore;
  if (!secret) throw new Error("egress secret not configured");
  const timestamp = String(headers["x-relay-timestamp"] || "");
  const nonce = String(headers["x-relay-nonce"] || "");
  const signature = String(headers["x-relay-signature"] || "");
  if (!/^\d{13}$/.test(timestamp) || Math.abs(now - Number(timestamp)) > NONCE_TTL_MS) throw new Error("stale egress request");
  const expected = crypto.createHmac("sha256", secret).update(`${timestamp}.${nonce}.${target}`).digest("hex");
  if (!safeEqual(signature, expected)) throw new Error("invalid egress signature");
  if (!nonceStore) throw new Error("egress nonce store not configured");
  nonceStore.claim(nonce, Number(timestamp), now);
  return { timestamp: Number(timestamp), nonce };
}

async function pinnedRequest(url) {
  const answers = await dns.lookup(url.hostname, { all: true, verbatim: true });
  if (!answers.length || answers.some((x) => blocked(x.address))) throw new Error("blocked DNS resolution");
  const selected = answers[0];
  const transport = url.protocol === "https:" ? https : http;
  return new Promise((resolve, reject) => {
    const request = transport.request({
      protocol: url.protocol,
      hostname: url.hostname,
      port: url.port || undefined,
      path: `${url.pathname}${url.search}`,
      method: "GET",
      servername: url.hostname,
      headers: { host: url.host, accept: "*/*", "user-agent": "NEXUS-Relay-Egress/1.1" },
      lookup: (_hostname, _options, callback) => callback(null, selected.address, selected.family),
      timeout: 20_000,
      rejectUnauthorized: true,
    }, (response) => resolve(response));
    request.on("timeout", () => request.destroy(new Error("upstream timeout")));
    request.on("error", reject);
    request.end();
  });
}

async function retrieve(raw) {
  let current = validUrl(raw);
  const chain = [];
  for (let hop = 0; hop <= MAX_REDIRECTS; hop += 1) {
    chain.push(current.href);
    const response = await pinnedRequest(current);
    if ([301,302,303,307,308].includes(response.statusCode || 0)) {
      response.resume();
      const location = response.headers.location;
      if (!location) throw new Error("redirect omitted location");
      if (hop === MAX_REDIRECTS) throw new Error("redirect limit exceeded");
      current = validUrl(new URL(location, current).href);
      continue;
    }
    const chunks = []; let bytes = 0;
    for await (const chunk of response) {
      bytes += chunk.length;
      if (bytes > MAX_BYTES) throw new Error("response too large");
      chunks.push(chunk);
    }
    return { status: response.statusCode || 502, headers: response.headers, body: Buffer.concat(chunks), finalUrl: current.href, chain };
  }
  throw new Error("redirect limit exceeded");
}

export function createEgressServer(options = {}) {
  const nonceStore = options.nonceStore || new PersistentNonceStore(options.nonceStorePath || NONCE_STORE_PATH, options.nonceTtlMs || NONCE_TTL_MS);
  return http.createServer(async (req, res) => {
    try {
      if (req.method !== "POST" || req.url !== "/fetch") { res.writeHead(404).end(); return; }
      const buffers = []; for await (const chunk of req) buffers.push(chunk);
      const body = JSON.parse(Buffer.concat(buffers).toString("utf8"));
      authenticateRequest(req.headers, body.url, { nonceStore, secret: options.secret ?? SECRET });
      const result = await retrieve(body.url);
      const headers = {
        "content-type": result.headers["content-type"] || "application/octet-stream",
        "x-relay-final-url": result.finalUrl,
        "x-relay-redirect-chain": Buffer.from(JSON.stringify(result.chain)).toString("base64url"),
        "x-relay-upstream-status": String(result.status),
      };
      res.writeHead(result.status, headers); res.end(result.body);
    } catch (error) {
      res.writeHead(400, { "content-type": "application/json" });
      res.end(JSON.stringify({ ok: false, error: String(error?.message || error) }));
    }
  });
}

const isMain = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) createEgressServer().listen(PORT, "0.0.0.0", () => console.log(`relay egress listening on ${PORT}`));
