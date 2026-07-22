import { normalizeFeed, normalizeHtml, normalizeJson } from "./extract.js";
import { handleMcp } from "./mcp.js";
import { authenticate, checkAndRecordUsage, sha256, usageSummary } from "./auth.js";
import { NEXUS_RELAY_SKILL } from "./skill.js";

const JSON_HEADERS = { "content-type": "application/json; charset=utf-8", "access-control-allow-origin": "*" };

function json(value, status = 200, extra = {}) {
  return new Response(JSON.stringify(value, null, 2), { status, headers: { ...JSON_HEADERS, ...extra } });
}

function errorResponse(error) {
  const status = Number(error?.status || 400);
  const headers = error?.retryAfter ? { "retry-after": String(error.retryAfter) } : {};
  return json({ ok: false, error: String(error?.message || error) }, status, headers);
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
      "user-agent": "NEXUS-Relay/0.2 (+hosted-public-web-context; contact=operator)",
      accept: "text/html,application/xhtml+xml,application/json,application/rss+xml,application/atom+xml,application/xml;q=0.9,*/*;q=0.5"
    }
  });
  const length = Number(response.headers.get("content-length") || 0);
  if (length > maxBytes) throw Object.assign(new Error(`response exceeds max_bytes (${length} > ${maxBytes})`), { status: 413 });
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength > maxBytes) throw Object.assign(new Error(`response exceeds max_bytes (${buffer.byteLength} > ${maxBytes})`), { status: 413 });
  const text = new TextDecoder().decode(buffer);
  const fetchedAt = new Date().toISOString();
  const mode = detectMode(response.headers.get("content-type"), text, args.mode);
  let normalized;
  if (mode === "json") normalized = normalizeJson({ value: JSON.parse(text), url: response.url, fetchedAt, status: response.status, headers: response.headers });
  else if (mode === "feed") normalized = normalizeFeed({ xml: text, url: response.url, fetchedAt, status: response.status, headers: response.headers });
  else normalized = normalizeHtml({ html: text, url: response.url, fetchedAt, status: response.status, headers: response.headers });
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
      content_sha256: await sha256(JSON.stringify(normalized.content)),
      fetched_at: fetchedAt
    },
    cache: { hit: false, key: cacheKey }
  };
  if (env.CACHE && response.ok) {
    await env.CACHE.put(cacheKey, JSON.stringify(result), { expirationTtl: Number(args.cache_ttl || env.DEFAULT_CACHE_TTL || 900) });
  }
  return result;
}

async function meteredRead(args, env, principal, surface) {
  const usage = await checkAndRecordUsage(env, principal, "relay_read", { surface, url_host: validateUrl(args.url).hostname });
  const result = await readPublicUrl(args, env);
  return { ...result, billing: { event_id: usage.event_id, customer_id: principal.customer_id, plan: principal.plan } };
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: { ...JSON_HEADERS, "access-control-allow-methods": "GET,POST,OPTIONS", "access-control-allow-headers": "content-type,authorization" } });
    }
    const url = new URL(request.url);
    if (url.pathname === "/" || url.pathname === "/health") {
      return json({ ok: true, product: env.NEXUS_RELAY_NAME || "NEXUS Relay", version: "0.2.0", model: "hosted metered API", surfaces: ["REST", "MCP", "SKILL.md"], skill_url: `${url.origin}/SKILL.md`, doctrine: "public data only; no login bypass; provenance on every response" });
    }
    if (url.pathname === "/SKILL.md" && request.method === "GET") {
      return new Response(NEXUS_RELAY_SKILL, { headers: { "content-type": "text/markdown; charset=utf-8", "content-disposition": "attachment; filename=SKILL.md", "access-control-allow-origin": "*" } });
    }
    try {
      if (url.pathname === "/v1/usage" && request.method === "GET") {
        const principal = await authenticate(request, env);
        return json({ ok: true, usage: await usageSummary(env, principal) });
      }
      if (url.pathname === "/v1/read" && request.method === "GET") {
        const principal = await authenticate(request, env);
        return json(await meteredRead({ url: url.searchParams.get("url"), mode: url.searchParams.get("mode") || "auto" }, env, principal, "rest_get"));
      }
      if (url.pathname === "/v1/read" && request.method === "POST") {
        const principal = await authenticate(request, env);
        return json(await meteredRead(await request.json(), env, principal, "rest_post"));
      }
      if (url.pathname === "/mcp" && request.method === "POST") {
        const principal = await authenticate(request, env);
        const payload = await handleMcp(request, (args) => meteredRead(args, env, principal, "mcp"));
        return payload === null ? new Response(null, { status: 202 }) : json(payload);
      }
    } catch (error) {
      return errorResponse(error);
    }
    return json({ ok: false, error: "not found" }, 404);
  }
};

export { detectMode, isPrivateHost, meteredRead, readPublicUrl, validateUrl };
