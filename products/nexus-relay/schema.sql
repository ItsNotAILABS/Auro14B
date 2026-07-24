PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS customers (
  id TEXT PRIMARY KEY, email TEXT, name TEXT,
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','suspended','closed')),
  stripe_customer_id TEXT UNIQUE, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS api_keys (
  id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, key_hash TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL, plan TEXT NOT NULL DEFAULT 'developer', active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0,1)),
  monthly_limit INTEGER NOT NULL DEFAULT 1000, rate_limit_per_minute INTEGER NOT NULL DEFAULT 30, created_at TEXT NOT NULL,
  FOREIGN KEY (customer_id) REFERENCES customers(id)
);
CREATE TABLE IF NOT EXISTS usage_events (
  id TEXT PRIMARY KEY, api_key_id TEXT NOT NULL, customer_id TEXT NOT NULL, operation TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'reserved' CHECK (status IN ('reserved','completed')), period TEXT NOT NULL,
  minute_bucket TEXT NOT NULL, created_at TEXT NOT NULL, completed_at TEXT, metadata_json TEXT,
  FOREIGN KEY (api_key_id) REFERENCES api_keys(id), FOREIGN KEY (customer_id) REFERENCES customers(id)
);
CREATE TABLE IF NOT EXISTS subscriptions (
  id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, stripe_customer_id TEXT, stripe_price_id TEXT, status TEXT NOT NULL,
  current_period_end INTEGER, cancel_at_period_end INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL, last_event_id TEXT,
  FOREIGN KEY (customer_id) REFERENCES customers(id)
);
CREATE TABLE IF NOT EXISTS invoices (
  id TEXT PRIMARY KEY, customer_id TEXT, stripe_customer_id TEXT, status TEXT NOT NULL,
  amount_due INTEGER NOT NULL DEFAULT 0, amount_paid INTEGER NOT NULL DEFAULT 0, currency TEXT NOT NULL DEFAULT 'usd',
  hosted_invoice_url TEXT, created_at TEXT NOT NULL, last_event_id TEXT, FOREIGN KEY (customer_id) REFERENCES customers(id)
);
CREATE TABLE IF NOT EXISTS billing_events (
  id TEXT PRIMARY KEY, type TEXT NOT NULL, created_at TEXT NOT NULL, raw_sha256 TEXT NOT NULL, result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS billing_outbox (
  id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, stripe_customer_id TEXT NOT NULL, event_name TEXT NOT NULL,
  value INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','sent')),
  attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT, created_at TEXT NOT NULL, sent_at TEXT,
  FOREIGN KEY (customer_id) REFERENCES customers(id)
);
CREATE INDEX IF NOT EXISTS idx_usage_key_period ON usage_events(api_key_id, period, status);
CREATE INDEX IF NOT EXISTS idx_usage_key_minute ON usage_events(api_key_id, minute_bucket, status);
CREATE INDEX IF NOT EXISTS idx_usage_customer_created ON usage_events(customer_id, created_at);
CREATE INDEX IF NOT EXISTS idx_subscriptions_customer ON subscriptions(customer_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_invoices_customer ON invoices(customer_id, created_at);
CREATE INDEX IF NOT EXISTS idx_billing_outbox_status ON billing_outbox(status, created_at);
