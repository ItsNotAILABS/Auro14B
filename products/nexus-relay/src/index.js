import { normalizeCsv, normalizeFeed, normalizeHtml, normalizeJson, normalizeSitemap, normalizeText } from "./extract.js";
import { handleMcp } from "./mcp.js";
import { authenticate, completeUsage, releaseUsage, reserveUsage, sha256, usageSummary } from "./auth.js";
import { NEXUS_RELAY_SKILL } from "./skill.js";
import { isBlockedIp, secureFetch, validateUrl } from "./security.js";

const MODES = new Set(["auto", "html", "feed", "json", "markdown", "text", "csv", "sitemap"]);

function allowedOrigin(request, env) {
  const origin = request.headers.get("origin");
  if (!origin) return null;
  const allowed = String(env.ALLOWED_ORIGINS || "").split(",").map((x) => x.trim()).filter(Boolean);
  return allowed.includes(origin) ? origin : null;
}

function responseHeaders(request, env, contentType = "application/json; charset=utf-8") {
  const headers = { "content-type": contentType, vary: "Origin" };
  const origin = allowedOrigin(request, env);
  if (origin) headers["access-control-allow-origin"] = origin;
  return headers;
}

function json(request, env, value, status = 200, extra = {}) {
  return new Response(JSON.stringify(value, null, 2), { status, headers: { ...responseHeaders(request, env), ...extra } });
}

function errorResponse(request, env, error) {
  const status = Number(error?.status || 400);
  const headers = error?.retryAfter ? { "retry-after": String(error.retryAfter) } : {};
  return json(request, env, { ok: false, error: String(error?.message || error) }, status, headers);
}

function detectMode(contentType, text, requested, pathname = "") {
  if (requested && !MODES.has(requested)) throw new Error(`unsupported mode: ${requested}`);
  if (requested && requested !== "auto") return requested;
  const type = (contentType || "").toLowerCase();
  const path = pathname.toLowerCase();
  if (type.includes("json") || path.endsWith(".json")) return "json";
  if (type.includes("csv") || path.endsWith(".csv")) return "csv";
  if (type.includes("markdown") || path.endsWith(".md") || path.endsWith(".markdown")) return "markdown";
  if (type.startsWith("text/plain") || path.endsWith(".txt")) return "text";
  if (/^\s*<(urlset|sitemapindex)\b/i.test(text) || path.includes("sitemap")) return "sitemap";
  if (type.includes("rss") || type.includes("atom") || /^\s*<(rss|feed)\b/i.test(text)) return "feed";
  if (type.includes("xml")) return "feed";
  return "html";
}

async function readPublicUrl(args, env, dependencies = {}) {
  const target = validateUrl(args.url);
  const maxBytes = Math.min(Number(args.max_bytes || env.MAX_RESPONSE_BYTES || 1500000), 5000000);
  const cacheKey = `read:${await sha256(`${target.href}|${args.mode || "auto"}|${maxBytes}`)}`;
  if (env.CACHE) {
    const hit = await env.CACHE.get(cacheKey, "json");
    if (hit) return { ...hit, cache: { hit: true, key: cacheKey } };
  }
  const started = Date.now();
  const { response, finalUrl, redirectChain } = await secureFetch(target.href, {
    fetchFn: dependencies.fetchFn,
    resolver: dependencies.resolver,
    maxRedirects: Number(env.MAX_REDIRECTS || 5),
    request: {
      headers: {
        "user-agent": "NEXUS-Relay/0.3 (+hosted-public-web-context; contact=operator)",
        accept: "text/html,application/xhtml+xml,application/json,text/markdown,text/plain,text/csv,application/rss+xml,application/atom+xml,application/xml;q=0.9,*/*;q=0.5"
      }
    }
  });
  const length = Number(response.headers.get("content-length") || 0);
  if (length > maxBytes) throw Object.assign(new Error(`response exceeds max_bytes (${length} > ${maxBytes})`), { status: 413 });
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength > maxBytes) throw Object.assign(new Error(`response exceeds max_bytes (${buffer.byteLength} > ${maxBytes})`), { status: 413 });
  const text = new TextDecoder().decode(buffer);
  const fetchedAt = new Date().toISOString();
  const mode = detectMode(response.headers.get("content-type"), text, args.mode, finalUrl.pathname);
  const common = { url: finalUrl.href, fetchedAt, status: response.status, headers: response.headers };
  let normalized;
  if (mode === "json") normalized = normalizeJson({ value: JSON.parse(text), ...common });
  else if (mode === "feed") normalized = normalizeFeed({ xml: text, ...common });
  else if (mode === "sitemap") normalized = normalizeSitemap({ xml: text, ...common });
  else if (mode === "csv") normalized = normalizeCsv({ text, ...common });
  else if (mode === "markdown") normalized = normalizeText({ text, kind: "markdown", ...common });
  else if (mode === "text") normalized = normalizeText({ text, kind: "text", ...common });
  else normalized = normalizeHtml({ html: text, ...common });
  const result = {
    ok: response.ok,
    ...normalized,
    receipt: {
      schema: "nexus.relay.receipt.v2",
      request_url: target.href,
      final_url: finalUrl.href,
      redirect_chain: redirectChain,
      mode,
      bytes: buffer.byteLength,
      latency_ms: Date.now() - started,
      content_sha256: await sha256(JSON.stringify(normalized.content)),
      fetched_at: fetchedAt
    },
    cache: { hit: false, key: cacheKey }
  };
  if (env.CACHE && response.ok) await env.CACHE.put(cacheKey, JSON.stringify(result), { expirationTtl: Number(args.cache_ttl || env.DEFAULT_CACHE_TTL || 900) });
  return result;
}

async function meteredRead(args, env, principal, surface, dependencies = {}) {
  const target = validateUrl(args.url);
  const reservation = await reserveUsage(env, principal, "relay_read", { surface, url_host: target.hostname, requested_mode: args.mode || "auto" });
  try {
    const result = await readPublicUrl(args, env, dependencies);
    if (!result.ok) throw Object.assign(new Error(`upstream returned HTTP ${result.source?.status || "error"}`), { status: 502 });
    await completeUsage(env, reservation.event_id, { surface, url_host: target.hostname, receipt_sha256: result.receipt.content_sha256 });
    return { ...result, metering: { event_id: reservation.event_id, customer_id: principal.customer_id, plan: principal.plan, status: "completed" } };
  } catch (error) {
    await releaseUsage(env, reservation.event_id);
    throw error;
  }
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      const origin = allowedOrigin(request, env);
      if (!origin) return new Response(null, { status: 403 });
      return new Response(null, { status: 204, headers: { ...responseHeaders(request, env), "access-control-allow-methods": "GET,POST,OPTIONS", "access-control-allow-headers": "content-type,authorization" } });
    }
    const url = new URL(request.url);
    if (url.pathname === "/" || url.pathname === "/health") return json(request, env, { ok: true, product: env.NEXUS_RELAY_NAME || "NEXUS Relay", version: "0.3.0", model: "hosted metered API", supported_modes: [...MODES], surfaces: ["REST", "MCP", "SKILL.md"], skill_url: `${url.origin}/SKILL.md`, billing_status: "metering only; subscriptions and payment reconciliation not implemented" });
    if (url.pathname === "/SKILL.md" && request.method === "GET") return new Response(NEXUS_RELAY_SKILL, { headers: { ...responseHeaders(request, env, "text/markdown; charset=utf-8"), "content-disposition": "attachment; filename=SKILL.md" } });
    try {
      if (url.pathname === "/v1/usage" && request.method === "GET") { const principal = await authenticate(request, env); return json(request, env, { ok: true, usage: await usageSummary(env, principal) }); }
      if (url.pathname === "/v1/read" && request.method === "GET") { const principal = await authenticate(request, env); return json(request, env, await meteredRead({ url: url.searchParams.get("url"), mode: url.searchParams.get("mode") || "auto" }, env, principal, "rest_get")); }
      if (url.pathname === "/v1/read" && request.method === "POST") { const principal = await authenticate(request, env); return json(request, env, await meteredRead(await request.json(), env, principal, "rest_post")); }
      if (url.pathname === "/mcp" && request.method === "POST") { const principal = await authenticate(request, env); const payload = await handleMcp(request, (args) => meteredRead(args, env, principal, "mcp")); return payload === null ? new Response(null, { status: 202 }) : json(request, env, payload); }
    } catch (error) { return errorResponse(request, env, error); }
    return json(request, env, { ok: false, error: "not found" }, 404);
  }
};

const isPrivateHost = (hostname) => hostname === "localhost" || hostname.endsWith(".local") || isBlockedIp(hostname);
export { detectMode, isPrivateHost, meteredRead, readPublicUrl, validateUrl };
