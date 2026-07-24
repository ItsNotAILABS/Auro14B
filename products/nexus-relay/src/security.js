const BLOCKED_V4 = [
  ["0.0.0.0", 8], ["10.0.0.0", 8], ["100.64.0.0", 10], ["127.0.0.0", 8],
  ["169.254.0.0", 16], ["172.16.0.0", 12], ["192.0.0.0", 24], ["192.0.2.0", 24],
  ["192.168.0.0", 16], ["198.18.0.0", 15], ["198.51.100.0", 24], ["203.0.113.0", 24],
  ["224.0.0.0", 4], ["240.0.0.0", 4]
];

function ipv4Number(value) {
  const parts = value.split(".");
  if (parts.length !== 4 || parts.some((x) => !/^\d+$/.test(x) || Number(x) > 255)) return null;
  return parts.reduce((n, x) => (n * 256 + Number(x)) >>> 0, 0) >>> 0;
}

function ipv4InCidr(ip, base, prefix) {
  const a = ipv4Number(ip); const b = ipv4Number(base);
  if (a === null || b === null) return false;
  const mask = prefix === 0 ? 0 : (0xffffffff << (32 - prefix)) >>> 0;
  return (a & mask) === (b & mask);
}

export function isBlockedIp(value) {
  const ip = String(value || "").trim().toLowerCase().replace(/^\[|\]$/g, "");
  if (ipv4Number(ip) !== null) return BLOCKED_V4.some(([base, prefix]) => ipv4InCidr(ip, base, prefix));
  if (!ip.includes(":")) return false;
  if (ip === "::" || ip === "::1") return true;
  if (ip.startsWith("fc") || ip.startsWith("fd")) return true;
  if (/^fe[89ab]/.test(ip)) return true;
  if (ip.startsWith("ff")) return true;
  if (ip.startsWith("2001:db8:") || ip.startsWith("2001:10:")) return true;
  if (ip.startsWith("::ffff:")) return isBlockedIp(ip.slice(7));
  return false;
}

export function validateUrl(raw) {
  let url;
  try { url = new URL(raw); } catch { throw Object.assign(new Error("url must be a valid absolute URL"), { status: 400 }); }
  if (!["http:", "https:"].includes(url.protocol)) throw Object.assign(new Error("only http and https URLs are supported"), { status: 400 });
  const host = url.hostname.toLowerCase();
  if (!host || host === "localhost" || host.endsWith(".localhost") || host.endsWith(".local") || isBlockedIp(host)) throw Object.assign(new Error("private, local, or reserved network targets are blocked"), { status: 400 });
  url.username = ""; url.password = ""; url.hash = "";
  return url;
}

export async function resolvePublicHost(hostname, resolver = fetch) {
  if (isBlockedIp(hostname)) throw Object.assign(new Error("blocked literal IP target"), { status: 400 });
  const answers = [];
  for (const type of ["A", "AAAA"]) {
    const endpoint = `https://cloudflare-dns.com/dns-query?name=${encodeURIComponent(hostname)}&type=${type}`;
    const response = await resolver(endpoint, { headers: { accept: "application/dns-json" }, redirect: "error" });
    if (!response.ok) throw Object.assign(new Error("DNS preflight failed"), { status: 502 });
    const payload = await response.json();
    for (const item of payload.Answer || []) if (item.type === 1 || item.type === 28) answers.push(item.data);
  }
  if (!answers.length) throw Object.assign(new Error("hostname did not resolve"), { status: 502 });
  if (answers.some(isBlockedIp)) throw Object.assign(new Error("hostname resolves to a blocked network"), { status: 400 });
  return answers;
}

async function hmacHex(secret, value) {
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey("raw", encoder.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const signature = await crypto.subtle.sign("HMAC", key, encoder.encode(value));
  return [...new Uint8Array(signature)].map((x) => x.toString(16).padStart(2, "0")).join("");
}

export async function secureFetchViaEgress(raw, options = {}) {
  const target = validateUrl(raw);
  const egressUrl = options.egressUrl;
  const secret = options.egressSecret;
  if (!egressUrl || !secret) throw Object.assign(new Error("pinned egress is not configured"), { status: 503 });
  const timestamp = String(Date.now());
  const signature = await hmacHex(secret, `${timestamp}.${target.href}`);
  const response = await (options.fetchFn || fetch)(egressUrl, {
    method: "POST",
    redirect: "error",
    headers: { "content-type": "application/json", "x-relay-timestamp": timestamp, "x-relay-signature": signature },
    body: JSON.stringify({ url: target.href })
  });
  const finalUrl = validateUrl(response.headers.get("x-relay-final-url") || target.href);
  let redirectChain = [target.href];
  try { redirectChain = JSON.parse(atob((response.headers.get("x-relay-redirect-chain") || "").replace(/-/g, "+").replace(/_/g, "/"))); } catch {}
  return { response, finalUrl, redirectChain, egress: "pinned-dns" };
}

export async function secureFetch(raw, options = {}) {
  if (options.egressUrl) return secureFetchViaEgress(raw, options);
  const fetchFn = options.fetchFn || fetch;
  const resolver = options.resolver || fetch;
  const maxRedirects = Math.min(Number(options.maxRedirects ?? 5), 10);
  let current = validateUrl(raw);
  const chain = [];
  for (let hop = 0; hop <= maxRedirects; hop += 1) {
    await resolvePublicHost(current.hostname, resolver);
    chain.push(current.href);
    const response = await fetchFn(current.href, { ...(options.request || {}), redirect: "manual" });
    if (![301, 302, 303, 307, 308].includes(response.status)) return { response, finalUrl: current, redirectChain: chain, egress: "worker-preflight" };
    const location = response.headers.get("location");
    if (!location) throw Object.assign(new Error("redirect response omitted Location"), { status: 502 });
    if (hop === maxRedirects) throw Object.assign(new Error("redirect limit exceeded"), { status: 508 });
    current = validateUrl(new URL(location, current).href);
  }
  throw Object.assign(new Error("redirect limit exceeded"), { status: 508 });
}
