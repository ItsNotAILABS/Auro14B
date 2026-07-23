PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS customers (
  id TEXT PRIMARY KEY,
  email TEXT,
  name TEXT,
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','suspended','closed')),
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
  id TEXT PRIMARY KEY,
  customer_id TEXT NOT NULL,
  key_hash TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  plan TEXT NOT NULL DEFAULT 'developer',
  active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0,1)),
  monthly_limit INTEGER NOT NULL DEFAULT 1000,
  rate_limit_per_minute INTEGER NOT NULL DEFAULT 30,
  created_at TEXT NOT NULL,
  FOREIGN KEY (customer_id) REFERENCES customers(id)
);

CREATE TABLE IF NOT EXISTS usage_events (
  id TEXT PRIMARY KEY,
  api_key_id TEXT NOT NULL,
  customer_id TEXT NOT NULL,
  operation TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'reserved' CHECK (status IN ('reserved','completed')),
  period TEXT NOT NULL,
  minute_bucket TEXT NOT NULL,
  created_at TEXT NOT NULL,
  completed_at TEXT,
  metadata_json TEXT,
  FOREIGN KEY (api_key_id) REFERENCES api_keys(id),
  FOREIGN KEY (customer_id) REFERENCES customers(id)
);

CREATE INDEX IF NOT EXISTS idx_usage_key_period ON usage_events(api_key_id, period, status);
CREATE INDEX IF NOT EXISTS idx_usage_key_minute ON usage_events(api_key_id, minute_bucket, status);
CREATE INDEX IF NOT EXISTS idx_usage_customer_created ON usage_events(customer_id, created_at);
