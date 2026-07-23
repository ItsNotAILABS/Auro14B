import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { Miniflare } from "miniflare";
import { authenticate, reserveUsage, sha256 } from "../src/auth.js";
import { isBlockedIp, secureFetch, secureFetchViaEgress, validateUrl } from "../src/security.js";

async function database() {
  const mf = new Miniflare({
    modules: true,
    script: "export default { fetch(){ return new Response('ok') } }",
    d1Databases: { DB: "00000000-0000-0000-0000-000000000001" }
  });
  const db = await mf.getD1Database("DB");
  const schema = await readFile(new URL("../schema.sql", import.meta.url), "utf8");
  for (const statement of schema.split(";").map((item) => item.trim()).filter(Boolean)) await db.prepare(statement).run();
  return { mf, db };
}

async function seed(db, { customerStatus = "active", monthly = 3, rpm = 3 } = {}) {
  const token = "nr_test_abcdefghijklmnopqrstuvwxyz";
  const hash = await sha256(token);
  await db.prepare("INSERT INTO customers (id,email,status,created_at) VALUES (?,?,?,?)")
    .bind("cus_test", "test@example.com", customerStatus, new Date().toISOString()).run();
  await db.prepare("INSERT INTO api_keys (id,customer_id,key_hash,name,plan,active,monthly_limit,rate_limit_per_minute,created_at) VALUES (?,?,?,?,?,?,?,?,?)")
    .bind("key_test", "cus_test", hash, "test", "developer", 1, monthly, rpm, new Date().toISOString()).run();
  return token;
}

test("authentication enforces parent customer suspension", async () => {
  const { mf, db } = await database();
  try {
    const token = await seed(db, { customerStatus: "suspended" });
    const request = new Request("https://relay.test/v1/usage", { headers: { authorization: `Bearer ${token}` } });
    await assert.rejects(() => authenticate(request, { DB: db }), /customer account is not active/);
  } finally { await mf.dispose(); }
});

test("atomic conditional reservation prevents concurrent quota overflow", async () => {
  const { mf, db } = await database();
  try {
    const token = await seed(db, { monthly: 3, rpm: 3 });
    const principal = await authenticate(new Request("https://relay.test", { headers: { authorization: `Bearer ${token}` } }), { DB: db });
    const attempts = await Promise.allSettled(Array.from({ length: 12 }, (_, i) => reserveUsage({ DB: db }, principal, "relay_read", { i })));
    assert.equal(attempts.filter((x) => x.status === "fulfilled").length, 3);
    assert.equal(attempts.filter((x) => x.status === "rejected").length, 9);
    const row = await db.prepare("SELECT COUNT(*) AS n FROM usage_events").first();
    assert.equal(Number(row.n), 3);
  } finally { await mf.dispose(); }
});

test("redirect destinations are revalidated before the next network hop", async () => {
  let outboundCalls = 0;
  const resolver = async () => new Response(JSON.stringify({ Answer: [{ type: 1, data: "93.184.216.34" }] }), { status: 200, headers: { "content-type": "application/json" } });
  const fetchFn = async () => {
    outboundCalls += 1;
    return new Response(null, { status: 302, headers: { location: "http://169.254.169.254/latest/meta-data" } });
  };
  await assert.rejects(() => secureFetch("https://example.com/start", { resolver, fetchFn }), /blocked/);
  assert.equal(outboundCalls, 1);
});

test("pinned egress calls are HMAC authenticated and preserve receipts", async () => {
  let captured;
  const chain = Buffer.from(JSON.stringify(["https://example.com/start", "https://example.com/final"])).toString("base64url");
  const fetchFn = async (url, options) => {
    captured = { url, options };
    return new Response("ok", { status: 200, headers: { "x-relay-final-url": "https://example.com/final", "x-relay-redirect-chain": chain } });
  };
  const result = await secureFetchViaEgress("https://example.com/start", { egressUrl: "https://egress.test/fetch", egressSecret: "secret", fetchFn });
  assert.equal(captured.url, "https://egress.test/fetch");
  assert.match(captured.options.headers["x-relay-signature"], /^[a-f0-9]{64}$/);
  assert.equal(result.egress, "pinned-dns");
  assert.equal(result.finalUrl.href, "https://example.com/final");
  assert.equal(result.redirectChain.length, 2);
});

test("blocked address coverage includes link-local reserved and IPv6 private space", () => {
  for (const ip of ["169.254.1.1", "100.64.0.1", "192.0.2.2", "198.51.100.2", "203.0.113.2", "::1", "fc00::1", "fd12::1", "fe80::1", "ff02::1", "2001:db8::1"]) assert.equal(isBlockedIp(ip), true, ip);
  assert.throws(() => validateUrl("http://[fd12::1]/"), /blocked/);
});
