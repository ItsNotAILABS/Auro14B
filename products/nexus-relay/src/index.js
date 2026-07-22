import { normalizeFeed, normalizeHtml, normalizeJson } from "./extract.js";
import { handleMcp } from "./mcp.js";

const JSON_HEADERS = { "content-type": "application/json; charset=utf-8", "access-control-allow-origin": "*" };

function json(value, status = 200, extra = {}) {
  return new Response(JSON.stringify(value, null, 2), { status, headers: { ...JSON_HEADERS, ...extra } });
}

function isPrivateHost(hostname) {
  const host = hostname.toLowerCase();
  if (host === "localhost" || host.endsWith(".local") || host === "0.0.0.0" || host === "::1") return true;
  if (/^127\./.test(host) || /^10\./.test(host) || /^192\.168\./.test(host)) return true;
  const m = host.match(/^172\.(\d+)\./);
  return Boolean(m && Number(m[1]) >= 16 && Number(m[1]) <= 31);
}

function validateUrl(raw) {
  let url;
  try { url = new URL(raw); } catch { throw new Error("url must be a valid absolute URL"); }
  if (!["http:", "https:"].includes(url.protocol)) throw new Error("only http and https URLs are supported");
  if (isPrivateHost(url.hostname)) throw new Error("private and loopback network targets are blocked");
  url.username = "";
  url.password = "";
  url.hash = "";
  return url;
}

function detectMode(contentType, text, requested) {
  if (requested && requested !== "auto") return requested;
  const type = (contentType || "").toLowerCase();
  if (type.includes("json")) return "json";
  if (type.includes("rss") || type.includes("atom") || type.includes("xml") || /^\s*<(rss|feed)\b/i.test(text)) return "feed";
  return "html";
}

async function sha256(value) {
  const bytes = new TextEncoder().encode(value);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function readPublicUrl(args, env) {
  const target = validateUrl(args.url);
  const maxBytes = Math.min(Number(args.max_bytes || env.MAX_RESPONSE_BYTES || 1500000), 5000000);
  const cacheKey = `read:${await sha256(`${target.href}|${args.mode || "auto"}|${maxBytes}`)}`;
  if (env.CACHE) {
    const hit = await env.CACHE.get(cacheKey, "json");
    if (hit) return { ...hit, cache: { hit: true, key: cacheKey } };
  }
  const started = Date.now();
  const response = await fetch(target.href, {
    redirect: "follow",
    headers: {
      "user-agent": "NEXUS-Relay/0.1 (+public-web-context; contact=operator)",
      accept: "text/html,application/xhtml+xml,application/json,application/rss+xml,application/atom+xml,application/xml;q=0.9,*/*;q=0.5"
    }
  });
  const length = Number(response.headers.get("content-length") || 0);
  if (length > maxBytes) throw new Error(`response exceeds max_bytes (${length} > ${maxBytes})`);
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength > maxBytes) throw new Error(`response exceeds max_bytes (${buffer.byteLength} > ${maxBytes})`);
  const text = new TextDecoder().decode(buffer);
  const fetchedAt = new Date().toISOString();
  const mode = detectMode(response.headers.get("content-type"), text, args.mode);
  let normalized;
  if (mode === "json") normalized = normalizeJson({ value: JSON.parse(text), url: response.url, fetchedAt, status: response.status, headers: response.headers });
  else if (mode === "feed") normalized = normalizeFeed({ xml: text, url: response.url, fetchedAt, status: response.status, headers: response.headers });
  else normalized = normalizeHtml({ html: text, url: response.url, fetchedAt, status: response.status, headers: response.headers });
  const contentHash = await sha256(JSON.stringify(normalized.content));
  const result = {
    ok: response.ok,
    ...normalized,
    receipt: {
      schema: "nexus.relay.receipt.v1",
      request_url: target.href,
      final_url: response.url,
      mode,
      bytes: buffer.byteLength,
      latency_ms: Date.now() - started,
      content_sha256: contentHash,
      fetched_at: fetchedAt
    },
    cache: { hit: false, key: cacheKey }
  };
  if (env.CACHE && response.ok) {
    await env.CACHE.put(cacheKey, JSON.stringify(result), { expirationTtl: Number(env.DEFAULT_CACHE_TTL || 900) });
  }
  return result;
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: { ...JSON_HEADERS, "access-control-allow-methods": "GET,POST,OPTIONS", "access-control-allow-headers": "content-type,authorization" } });
    }
    const url = new URL(request.url);
    if (url.pathname === "/" || url.pathname === "/health") {
      return json({ ok: true, product: env.NEXUS_RELAY_NAME || "NEXUS Relay", version: "0.1.0", surfaces: ["REST", "MCP"], doctrine: "public data only; no login bypass; provenance on every response" });
    }
    if (url.pathname === "/v1/read" && request.method === "GET") {
      try { return json(await readPublicUrl({ url: url.searchParams.get("url"), mode: url.searchParams.get("mode") || "auto" }, env)); }
      catch (error) { return json({ ok: false, error: String(error?.message || error) }, 400); }
    }
    if (url.pathname === "/v1/read" && request.method === "POST") {
      try { return json(await readPublicUrl(await request.json(), env)); }
      catch (error) { return json({ ok: false, error: String(error?.message || error) }, 400); }
    }
    if (url.pathname === "/mcp" && request.method === "POST") {
      const payload = await handleMcp(request, (args) => readPublicUrl(args, env));
      return payload === null ? new Response(null, { status: 202 }) : json(payload);
    }
    return json({ ok: false, error: "not found" }, 404);
  }
};

export { detectMode, isPrivateHost, readPublicUrl, validateUrl };
