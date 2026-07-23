ALTER TABLE customers ADD COLUMN stripe_customer_id TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_customers_stripe ON customers(stripe_customer_id);

CREATE TABLE IF NOT EXISTS subscriptions (
  id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, stripe_customer_id TEXT, stripe_price_id TEXT, status TEXT NOT NULL,
  current_period_end INTEGER, cancel_at_period_end INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL, last_event_id TEXT,
  FOREIGN KEY (customer_id) REFERENCES customers(id)
);
CREATE INDEX IF NOT EXISTS idx_subscriptions_customer ON subscriptions(customer_id, updated_at);
CREATE TABLE IF NOT EXISTS invoices (
  id TEXT PRIMARY KEY, customer_id TEXT, stripe_customer_id TEXT, status TEXT NOT NULL,
  amount_due INTEGER NOT NULL DEFAULT 0, amount_paid INTEGER NOT NULL DEFAULT 0, currency TEXT NOT NULL DEFAULT 'usd',
  hosted_invoice_url TEXT, created_at TEXT NOT NULL, last_event_id TEXT, FOREIGN KEY (customer_id) REFERENCES customers(id)
);
CREATE INDEX IF NOT EXISTS idx_invoices_customer ON invoices(customer_id, created_at);
CREATE TABLE IF NOT EXISTS billing_events (
  id TEXT PRIMARY KEY, type TEXT NOT NULL, created_at TEXT NOT NULL, raw_sha256 TEXT NOT NULL, result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS billing_outbox (
  id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, stripe_customer_id TEXT NOT NULL, event_name TEXT NOT NULL,
  value INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','sent')),
  attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT, created_at TEXT NOT NULL, sent_at TEXT,
  FOREIGN KEY (customer_id) REFERENCES customers(id)
);
CREATE INDEX IF NOT EXISTS idx_billing_outbox_status ON billing_outbox(status, created_at);
