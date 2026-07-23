function errorMessage(error) { return String(error?.message || error).slice(0, 1000); }

function meterEventName(env, plan) {
  let catalog = {};
  try { catalog = JSON.parse(env.RELAY_PRICE_CATALOG || "{}"); } catch {}
  const entry = Object.values(catalog).find((item) => item?.plan === plan);
  return entry?.meter_event_name || env.STRIPE_METER_EVENT_NAME || "nexus_relay_requests";
}

async function sendMeterEvent(env, row) {
  if (!env.STRIPE_SECRET_KEY) throw new Error("Stripe is not configured");
  const body = new URLSearchParams();
  body.set("event_name", row.event_name);
  body.set("identifier", row.id);
  body.set("timestamp", String(Math.floor(new Date(row.created_at).getTime() / 1000)));
  body.set("payload[stripe_customer_id]", row.stripe_customer_id);
  body.set("payload[value]", String(row.value));
  const response = await (env.STRIPE_FETCH || fetch)("https://api.stripe.com/v1/billing/meter_events", {
    method: "POST",
    headers: { authorization: `Bearer ${env.STRIPE_SECRET_KEY}`, "content-type": "application/x-www-form-urlencoded" },
    body
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload?.error?.message || "Stripe meter event failed");
  return payload;
}

export async function enqueueUsageForBilling(env, principal, usageEventId) {
  if (!principal.stripe_customer_id) return { status: "unlinked", event_id: usageEventId };
  await env.DB.prepare(`INSERT OR IGNORE INTO billing_outbox
    (id,customer_id,stripe_customer_id,event_name,value,status,attempts,created_at)
    VALUES (?,?,?,?,1,'pending',0,?)`)
    .bind(usageEventId, principal.customer_id, principal.stripe_customer_id, meterEventName(env, principal.plan), new Date().toISOString()).run();
  try {
    await deliverBillingOutboxItem(env, usageEventId);
    return { status: "sent", event_id: usageEventId };
  } catch (error) {
    return { status: "pending", event_id: usageEventId, error: errorMessage(error) };
  }
}

export async function deliverBillingOutboxItem(env, id) {
  const row = await env.DB.prepare("SELECT * FROM billing_outbox WHERE id=? AND status='pending'").bind(id).first();
  if (!row) return { skipped: true, id };
  try {
    await sendMeterEvent(env, row);
    await env.DB.prepare("UPDATE billing_outbox SET status='sent', attempts=attempts+1, last_error=NULL, sent_at=? WHERE id=? AND status='pending'")
      .bind(new Date().toISOString(), id).run();
    return { sent: true, id };
  } catch (error) {
    await env.DB.prepare("UPDATE billing_outbox SET attempts=attempts+1, last_error=? WHERE id=? AND status='pending'")
      .bind(errorMessage(error), id).run();
    throw error;
  }
}

export async function flushBillingOutbox(env, limit = 100) {
  const rows = await env.DB.prepare("SELECT id FROM billing_outbox WHERE status='pending' ORDER BY created_at LIMIT ?").bind(Math.min(Number(limit || 100), 500)).all();
  const results = [];
  for (const row of rows.results || []) {
    try { results.push(await deliverBillingOutboxItem(env, row.id)); }
    catch (error) { results.push({ sent: false, id: row.id, error: errorMessage(error) }); }
  }
  return { attempted: results.length, sent: results.filter((x) => x.sent).length, failed: results.filter((x) => x.sent === false).length, results };
}
