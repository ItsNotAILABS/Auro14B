import http from "node:http";
import https from "node:https";
import dns from "node:dns/promises";
import crypto from "node:crypto";

const PORT = Number(process.env.PORT || 8788);
const SECRET = process.env.RELAY_EGRESS_SECRET || "";
const MAX_BYTES = Math.min(Number(process.env.MAX_RESPONSE_BYTES || 5_000_000), 10_000_000);
const MAX_REDIRECTS = Math.min(Number(process.env.MAX_REDIRECTS || 5), 10);

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

function authenticate(req, target) {
  if (!SECRET) throw new Error("egress secret not configured");
  const timestamp = req.headers["x-relay-timestamp"] || "";
  const signature = req.headers["x-relay-signature"] || "";
  if (Math.abs(Date.now() - Number(timestamp)) > 60_000) throw new Error("stale egress request");
  const expected = crypto.createHmac("sha256", SECRET).update(`${timestamp}.${target}`).digest("hex");
  if (!safeEqual(signature, expected)) throw new Error("invalid egress signature");
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
      headers: { host: url.host, accept: "*/*", "user-agent": "NEXUS-Relay-Egress/1.0" },
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

http.createServer(async (req, res) => {
  try {
    if (req.method !== "POST" || req.url !== "/fetch") { res.writeHead(404).end(); return; }
    const buffers = []; for await (const chunk of req) buffers.push(chunk);
    const body = JSON.parse(Buffer.concat(buffers).toString("utf8"));
    authenticate(req, body.url);
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
}).listen(PORT, "0.0.0.0", () => console.log(`relay egress listening on ${PORT}`));
