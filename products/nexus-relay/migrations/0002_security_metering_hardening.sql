PRAGMA foreign_keys = ON;

ALTER TABLE usage_events ADD COLUMN status TEXT NOT NULL DEFAULT 'completed' CHECK (status IN ('reserved','completed'));
ALTER TABLE usage_events ADD COLUMN completed_at TEXT;

DROP INDEX IF EXISTS idx_usage_key_period;
DROP INDEX IF EXISTS idx_usage_key_minute;
CREATE INDEX idx_usage_key_period ON usage_events(api_key_id, period, status);
CREATE INDEX idx_usage_key_minute ON usage_events(api_key_id, minute_bucket, status);

-- Existing events represent completed historical usage.
UPDATE usage_events SET status = 'completed', completed_at = COALESCE(completed_at, created_at);
