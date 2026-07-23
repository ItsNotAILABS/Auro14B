const encoder = new TextEncoder();

export async function sha256(value) {
  const digest = await crypto.subtle.digest("SHA-256", encoder.encode(value));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export function bearerToken(request) {
  const value = request.headers.get("authorization") || "";
  const match = value.match(/^Bearer\s+(.+)$/i);
  return match?.[1]?.trim() || "";
}

function periodKey(date = new Date()) {
  return date.toISOString().slice(0, 7);
}

function minuteKey(date = new Date()) {
  return date.toISOString().slice(0, 16);
}

export async function authenticate(request, env) {
  const token = bearerToken(request);
  if (!token) throw Object.assign(new Error("missing bearer API key"), { status: 401 });
  if (!token.startsWith("nr_live_") && !token.startsWith("nr_test_")) {
    throw Object.assign(new Error("invalid API key format"), { status: 401 });
  }
  if (!env.DB) throw Object.assign(new Error("metering database is not configured"), { status: 503 });
  const keyHash = await sha256(token);
  const principal = await env.DB.prepare(
    `SELECT k.id, k.customer_id, k.name, k.plan, k.active,
            k.monthly_limit, k.rate_limit_per_minute,
            c.status AS customer_status
       FROM api_keys k
       JOIN customers c ON c.id = k.customer_id
      WHERE k.key_hash = ?
      LIMIT 1`
  ).bind(keyHash).first();
  if (!principal) throw Object.assign(new Error("invalid API key"), { status: 401 });
  if (!principal.active) throw Object.assign(new Error("API key is suspended"), { status: 403 });
  if (principal.customer_status !== "active") {
    throw Object.assign(new Error("customer account is not active"), { status: 403 });
  }
  return { ...principal, key_hash: keyHash };
}

export async function reserveUsage(env, principal, operation, metadata = {}, now = new Date()) {
  const eventId = crypto.randomUUID();
  const month = periodKey(now);
  const minute = minuteKey(now);
  const monthlyLimit = Number(principal.monthly_limit || 0);
  const rpm = Number(principal.rate_limit_per_minute || 0);
  const statement = env.DB.prepare(
    `INSERT INTO usage_events
       (id, api_key_id, customer_id, operation, status, period, minute_bucket, created_at, metadata_json)
     SELECT ?, ?, ?, ?, 'reserved', ?, ?, ?, ?
      WHERE (? <= 0 OR (SELECT COUNT(*) FROM usage_events WHERE api_key_id = ? AND period = ?) < ?)
        AND (? <= 0 OR (SELECT COUNT(*) FROM usage_events WHERE api_key_id = ? AND minute_bucket = ?) < ?)`
  ).bind(
    eventId,
    principal.id,
    principal.customer_id,
    operation,
    month,
    minute,
    now.toISOString(),
    JSON.stringify(metadata),
    monthlyLimit,
    principal.id,
    month,
    monthlyLimit,
    rpm,
    principal.id,
    minute,
    rpm
  );
  const result = await statement.run();
  if (Number(result?.meta?.changes || 0) !== 1) {
    throw Object.assign(new Error("request quota or rate limit exceeded"), { status: 429, retryAfter: 60 });
  }
  return { event_id: eventId, period: month, limit: monthlyLimit };
}

export async function completeUsage(env, eventId, metadata = {}) {
  await env.DB.prepare(
    "UPDATE usage_events SET status = 'completed', completed_at = ?, metadata_json = ? WHERE id = ? AND status = 'reserved'"
  ).bind(new Date().toISOString(), JSON.stringify(metadata), eventId).run();
}

export async function releaseUsage(env, eventId) {
  await env.DB.prepare("DELETE FROM usage_events WHERE id = ? AND status = 'reserved'").bind(eventId).run();
}

export async function usageSummary(env, principal) {
  const month = periodKey();
  const row = await env.DB.prepare(
    "SELECT COUNT(*) AS requests FROM usage_events WHERE api_key_id = ? AND period = ? AND status = 'completed'"
  ).bind(principal.id, month).first();
  const used = Number(row?.requests || 0);
  const limit = Number(principal.monthly_limit || 0);
  return {
    customer_id: principal.customer_id,
    customer_status: principal.customer_status,
    key_name: principal.name,
    plan: principal.plan,
    period: month,
    requests_used: used,
    request_limit: limit,
    requests_remaining: limit > 0 ? Math.max(0, limit - used) : null,
    rate_limit_per_minute: Number(principal.rate_limit_per_minute || 0)
  };
}
