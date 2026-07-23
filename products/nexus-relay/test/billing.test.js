import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createHmac } from "node:crypto";
import { Miniflare } from "miniflare";
import { processStripeWebhook } from "../src/billing.js";
import { sha256 } from "../src/auth.js";

async function database() {
  const mf = new Miniflare({ modules: true, script: "export default { fetch(){ return new Response('ok') } }", d1Databases: { DB: "00000000-0000-0000-0000-000000000002" } });
  const db = await mf.getD1Database("DB");
  const schema = await readFile(new URL("../schema.sql", import.meta.url), "utf8");
  for (const statement of schema.split(";").map((x) => x.trim()).filter(Boolean)) await db.prepare(statement).run();
  return { mf, db };
}

function signedRequest(event, secret) {
  const raw = JSON.stringify(event);
  const timestamp = Math.floor(Date.now() / 1000);
  const signature = createHmac("sha256", secret).update(`${timestamp}.${raw}`).digest("hex");
  return new Request("https://relay.test/v1/billing/webhook", { method: "POST", headers: { "stripe-signature": `t=${timestamp},v1=${signature}`, "content-type": "application/json" }, body: raw });
}

async function seed(db) {
  const key = "nr_test_billing_customer_key_123456789";
  await db.prepare("INSERT INTO customers (id,email,status,created_at) VALUES (?,?,?,?)").bind("cus_relay", "buyer@example.com", "active", new Date().toISOString()).run();
  await db.prepare("INSERT INTO api_keys (id,customer_id,key_hash,name,plan,active,monthly_limit,rate_limit_per_minute,created_at) VALUES (?,?,?,?,?,?,?,?,?)")
    .bind("key_relay", "cus_relay", await sha256(key), "default", "developer", 1, 1000, 30, new Date().toISOString()).run();
}

test("subscription webhook synchronizes entitlements and is idempotent", async () => {
  const { mf, db } = await database();
  try {
    await seed(db);
    const env = { DB: db, STRIPE_WEBHOOK_SECRET: "whsec_test", RELAY_PRICE_CATALOG: JSON.stringify({ price_team: { plan: "team", monthly_limit: 25000, rate_limit_per_minute: 120 } }) };
    const event = { id: "evt_subscription_1", type: "customer.subscription.updated", data: { object: { id: "sub_1", customer: "cus_stripe_1", status: "active", current_period_end: 1900000000, cancel_at_period_end: false, metadata: { relay_customer_id: "cus_relay" }, items: { data: [{ price: { id: "price_team" } }] } } } };
    const first = await processStripeWebhook(signedRequest(event, env.STRIPE_WEBHOOK_SECRET), env);
    const second = await processStripeWebhook(signedRequest(event, env.STRIPE_WEBHOOK_SECRET), env);
    assert.equal(first.duplicate, false);
    assert.equal(second.duplicate, true);
    const key = await db.prepare("SELECT plan,active,monthly_limit,rate_limit_per_minute FROM api_keys WHERE id='key_relay'").first();
    assert.deepEqual({ plan: key.plan, active: key.active, monthly: key.monthly_limit, rpm: key.rate_limit_per_minute }, { plan: "team", active: 1, monthly: 25000, rpm: 120 });
    const events = await db.prepare("SELECT COUNT(*) AS n FROM billing_events").first();
    assert.equal(Number(events.n), 1);
  } finally { await mf.dispose(); }
});

test("failed invoice suspends customer and keys", async () => {
  const { mf, db } = await database();
  try {
    await seed(db);
    await db.prepare("UPDATE customers SET stripe_customer_id='cus_stripe_2' WHERE id='cus_relay'").run();
    const env = { DB: db, STRIPE_WEBHOOK_SECRET: "whsec_test" };
    const event = { id: "evt_invoice_failed", type: "invoice.payment_failed", data: { object: { id: "in_1", customer: "cus_stripe_2", status: "open", amount_due: 2900, amount_paid: 0, currency: "usd", created: 1800000000, metadata: { relay_customer_id: "cus_relay" } } } };
    await processStripeWebhook(signedRequest(event, env.STRIPE_WEBHOOK_SECRET), env);
    const customer = await db.prepare("SELECT status FROM customers WHERE id='cus_relay'").first();
    const key = await db.prepare("SELECT active FROM api_keys WHERE id='key_relay'").first();
    assert.equal(customer.status, "suspended");
    assert.equal(key.active, 0);
  } finally { await mf.dispose(); }
});

test("invalid webhook signature is rejected", async () => {
  const { mf, db } = await database();
  try {
    const request = new Request("https://relay.test/v1/billing/webhook", { method: "POST", headers: { "stripe-signature": `t=${Math.floor(Date.now()/1000)},v1=bad` }, body: "{}" });
    await assert.rejects(() => processStripeWebhook(request, { DB: db, STRIPE_WEBHOOK_SECRET: "whsec_test" }), /invalid Stripe signature/);
  } finally { await mf.dispose(); }
});
