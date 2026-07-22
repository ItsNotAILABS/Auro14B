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
  const keyHash = await sha256(token);
  if (!env.DB) throw Object.assign(new Error("billing database is not configured"), { status: 503 });
  const key = await env.DB.prepare(
    "SELECT id, customer_id, name, plan, active, monthly_limit, rate_limit_per_minute FROM api_keys WHERE key_hash = ? LIMIT 1"
  ).bind(keyHash).first();
  if (!key) throw Object.assign(new Error("invalid API key"), { status: 401 });
  if (!key.active) throw Object.assign(new Error("API key is suspended"), { status: 403 });
  return { ...key, key_hash: keyHash };
}

export async function checkAndRecordUsage(env, principal, operation, metadata = {}) {
  const now = new Date();
  const month = periodKey(now);
  const minute = minuteKey(now);
  const monthlyLimit = Number(principal.monthly_limit || 0);
  const rpm = Number(principal.rate_limit_per_minute || 0);

  const monthly = await env.DB.prepare(
    "SELECT COUNT(*) AS n FROM usage_events WHERE api_key_id = ? AND period = ?"
  ).bind(principal.id, month).first();
  const used = Number(monthly?.n || 0);
  if (monthlyLimit > 0 && used >= monthlyLimit) {
    throw Object.assign(new Error("monthly request quota exceeded"), { status: 429, retryAfter: 3600 });
  }

  const recent = await env.DB.prepare(
    "SELECT COUNT(*) AS n FROM usage_events WHERE api_key_id = ? AND minute_bucket = ?"
  ).bind(principal.id, minute).first();
  if (rpm > 0 && Number(recent?.n || 0) >= rpm) {
    throw Object.assign(new Error("per-minute rate limit exceeded"), { status: 429, retryAfter: 60 });
  }

  const eventId = crypto.randomUUID();
  await env.DB.prepare(
    "INSERT INTO usage_events (id, api_key_id, customer_id, operation, period, minute_bucket, created_at, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
  ).bind(
    eventId,
    principal.id,
    principal.customer_id,
    operation,
    month,
    minute,
    now.toISOString(),
    JSON.stringify(metadata)
  ).run();
  return { event_id: eventId, period: month, used_before: used, limit: monthlyLimit };
}

export async function usageSummary(env, principal) {
  const month = periodKey();
  const row = await env.DB.prepare(
    "SELECT COUNT(*) AS requests FROM usage_events WHERE api_key_id = ? AND period = ?"
  ).bind(principal.id, month).first();
  const used = Number(row?.requests || 0);
  const limit = Number(principal.monthly_limit || 0);
  return {
    customer_id: principal.customer_id,
    key_name: principal.name,
    plan: principal.plan,
    period: month,
    requests_used: used,
    request_limit: limit,
    requests_remaining: limit > 0 ? Math.max(0, limit - used) : null,
    rate_limit_per_minute: Number(principal.rate_limit_per_minute || 0)
  };
}
