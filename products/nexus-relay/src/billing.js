import { sha256 } from "./auth.js";

const enc = new TextEncoder();

function httpError(message, status = 400) {
  return Object.assign(new Error(message), { status });
}

function form(data) {
  const out = new URLSearchParams();
  for (const [key, value] of Object.entries(data)) if (value !== undefined && value !== null) out.set(key, String(value));
  return out;
}

async function stripeRequest(env, path, data) {
  if (!env.STRIPE_SECRET_KEY) throw httpError("Stripe is not configured", 503);
  const response = await (env.STRIPE_FETCH || fetch)(`https://api.stripe.com/v1${path}`, {
    method: "POST",
    headers: { authorization: `Bearer ${env.STRIPE_SECRET_KEY}`, "content-type": "application/x-www-form-urlencoded" },
    body: form(data)
  });
  const payload = await response.json();
  if (!response.ok) throw httpError(payload?.error?.message || "Stripe request failed", 502);
  return payload;
}

function timingSafeEqual(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i += 1) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

async function hmacHex(secret, value) {
  const key = await crypto.subtle.importKey("raw", enc.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(value));
  return [...new Uint8Array(sig)].map((x) => x.toString(16).padStart(2, "0")).join("");
}

export async function verifyStripeWebhook(request, env) {
  const secret = env.STRIPE_WEBHOOK_SECRET;
  if (!secret) throw httpError("Stripe webhook is not configured", 503);
  const raw = await request.text();
  const header = request.headers.get("stripe-signature") || "";
  const parts = Object.fromEntries(header.split(",").map((item) => item.split("=", 2)));
  const timestamp = Number(parts.t || 0);
  if (!timestamp || Math.abs(Date.now() / 1000 - timestamp) > Number(env.STRIPE_WEBHOOK_TOLERANCE || 300)) throw httpError("stale Stripe signature", 400);
  const expected = await hmacHex(secret, `${timestamp}.${raw}`);
  const signatures = header.split(",").filter((x) => x.startsWith("v1=")).map((x) => x.slice(3));
  if (!signatures.some((value) => timingSafeEqual(value, expected))) throw httpError("invalid Stripe signature", 400);
  return { event: JSON.parse(raw), raw_sha256: await sha256(raw) };
}

export function priceCatalog(env) {
  try { return JSON.parse(env.RELAY_PRICE_CATALOG || "{}"); }
  catch { throw httpError("invalid RELAY_PRICE_CATALOG", 503); }
}

export function entitlementFor(env, priceId) {
  return priceCatalog(env)[priceId] || null;
}

export async function createCheckoutSession(env, principal, body = {}) {
  const priceId = body.price_id || env.STRIPE_DEFAULT_PRICE_ID;
  if (!priceId) throw httpError("price_id is required", 400);
  if (!entitlementFor(env, priceId) && String(env.ALLOW_UNLISTED_PRICE_IDS || "").toLowerCase() !== "true") throw httpError("price_id is not in the Relay catalog", 400);
  const success = body.success_url || env.BILLING_SUCCESS_URL;
  const cancel = body.cancel_url || env.BILLING_CANCEL_URL;
  if (!success || !cancel) throw httpError("billing return URLs are not configured", 503);
  const customer = await env.DB.prepare("SELECT stripe_customer_id,email FROM customers WHERE id=?").bind(principal.customer_id).first();
  const payload = await stripeRequest(env, "/checkout/sessions", {
    mode: "subscription",
    success_url: success,
    cancel_url: cancel,
    client_reference_id: principal.customer_id,
    customer: customer?.stripe_customer_id || undefined,
    customer_email: customer?.stripe_customer_id ? undefined : customer?.email,
    "line_items[0][price]": priceId,
    "line_items[0][quantity]": 1,
    "subscription_data[metadata][relay_customer_id]": principal.customer_id,
    "metadata[relay_customer_id]": principal.customer_id
  });
  return { id: payload.id, url: payload.url, expires_at: payload.expires_at };
}

export async function createPortalSession(env, principal, body = {}) {
  const customer = await env.DB.prepare("SELECT stripe_customer_id FROM customers WHERE id=?").bind(principal.customer_id).first();
  if (!customer?.stripe_customer_id) throw httpError("customer has no Stripe account", 409);
  const returnUrl = body.return_url || env.BILLING_PORTAL_RETURN_URL;
  if (!returnUrl) throw httpError("billing portal return URL is not configured", 503);
  const payload = await stripeRequest(env, "/billing_portal/sessions", { customer: customer.stripe_customer_id, return_url: returnUrl });
  return { id: payload.id, url: payload.url };
}

async function synchronizeSubscription(env, object, eventId) {
  const customerId = object.metadata?.relay_customer_id || object.client_reference_id;
  if (!customerId) return { ignored: true, reason: "missing relay_customer_id" };
  const item = object.items?.data?.[0];
  const priceId = item?.price?.id || object.metadata?.price_id || null;
  const entitlement = entitlementFor(env, priceId);
  const status = object.status || "active";
  const subscriptionHealthy = ["active", "trialing"].includes(status);
  const enabled = Boolean(subscriptionHealthy && entitlement);
  await env.DB.prepare(`INSERT INTO subscriptions
    (id,customer_id,stripe_customer_id,stripe_price_id,status,current_period_end,cancel_at_period_end,updated_at,last_event_id)
    VALUES (?,?,?,?,?,?,?,?,?)
    ON CONFLICT(id) DO UPDATE SET stripe_customer_id=excluded.stripe_customer_id,stripe_price_id=excluded.stripe_price_id,status=excluded.status,current_period_end=excluded.current_period_end,cancel_at_period_end=excluded.cancel_at_period_end,updated_at=excluded.updated_at,last_event_id=excluded.last_event_id`)
    .bind(object.id, customerId, String(object.customer || ""), priceId, status, object.current_period_end || null, object.cancel_at_period_end ? 1 : 0, new Date().toISOString(), eventId).run();
  await env.DB.prepare("UPDATE customers SET stripe_customer_id=?, status=? WHERE id=?")
    .bind(String(object.customer || ""), enabled ? "active" : "suspended", customerId).run();
  await env.DB.prepare("UPDATE api_keys SET active=?, plan=?, monthly_limit=?, rate_limit_per_minute=? WHERE customer_id=?")
    .bind(enabled ? 1 : 0, entitlement?.plan || "unentitled", Number(entitlement?.monthly_limit || 0), Number(entitlement?.rate_limit_per_minute || 0), customerId).run();
  return { customer_id: customerId, enabled, price_id: priceId, entitlement_found: Boolean(entitlement) };
}

async function recordInvoice(env, object, eventId) {
  await env.DB.prepare(`INSERT INTO invoices (id,customer_id,stripe_customer_id,status,amount_due,amount_paid,currency,hosted_invoice_url,created_at,last_event_id)
    VALUES (?,?,?,?,?,?,?,?,?,?)
    ON CONFLICT(id) DO UPDATE SET customer_id=excluded.customer_id,status=excluded.status,amount_due=excluded.amount_due,amount_paid=excluded.amount_paid,hosted_invoice_url=excluded.hosted_invoice_url,last_event_id=excluded.last_event_id`)
    .bind(object.id, object.metadata?.relay_customer_id || null, String(object.customer || ""), object.status || "unknown", Number(object.amount_due || 0), Number(object.amount_paid || 0), object.currency || "usd", object.hosted_invoice_url || null, new Date((object.created || Date.now()/1000) * 1000).toISOString(), eventId).run();
}

export async function processStripeWebhook(request, env) {
  const { event, raw_sha256 } = await verifyStripeWebhook(request, env);
  const reservation = await env.DB.prepare("INSERT OR IGNORE INTO billing_events (id,type,created_at,raw_sha256,result_json) VALUES (?,?,?,?,?)")
    .bind(event.id, event.type, new Date().toISOString(), raw_sha256, JSON.stringify({ status: "processing" })).run();
  if (Number(reservation.meta?.changes || 0) !== 1) return { duplicate: true, event_id: event.id };
  try {
    const object = event.data?.object || {};
    let result = { ignored: true };
    if (event.type.startsWith("customer.subscription.")) result = await synchronizeSubscription(env, object, event.id);
    else if (event.type.startsWith("invoice.")) {
      await recordInvoice(env, object, event.id);
      if (event.type === "invoice.payment_failed" && object.customer) {
        await env.DB.prepare("UPDATE customers SET status='suspended' WHERE stripe_customer_id=?").bind(String(object.customer)).run();
        await env.DB.prepare("UPDATE api_keys SET active=0 WHERE customer_id IN (SELECT id FROM customers WHERE stripe_customer_id=?)").bind(String(object.customer)).run();
      }
      result = { invoice_id: object.id, status: object.status || null };
    } else if (event.type === "checkout.session.completed") {
      if (object.client_reference_id && object.customer) await env.DB.prepare("UPDATE customers SET stripe_customer_id=? WHERE id=?").bind(String(object.customer), object.client_reference_id).run();
      result = { customer_id: object.client_reference_id || null };
    }
    await env.DB.prepare("UPDATE billing_events SET result_json=? WHERE id=?").bind(JSON.stringify({ status: "completed", ...result }), event.id).run();
    return { duplicate: false, event_id: event.id, type: event.type, result };
  } catch (error) {
    await env.DB.prepare("DELETE FROM billing_events WHERE id=?").bind(event.id).run();
    throw error;
  }
}

export async function billingSummary(env, principal) {
  const subscription = await env.DB.prepare("SELECT * FROM subscriptions WHERE customer_id=? ORDER BY updated_at DESC LIMIT 1").bind(principal.customer_id).first();
  const invoices = await env.DB.prepare("SELECT id,status,amount_due,amount_paid,currency,hosted_invoice_url,created_at FROM invoices WHERE customer_id=? ORDER BY created_at DESC LIMIT 12").bind(principal.customer_id).all();
  return { subscription: subscription || null, invoices: invoices.results || [], catalog: priceCatalog(env) };
}
