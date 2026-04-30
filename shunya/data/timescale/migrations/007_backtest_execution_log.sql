-- Persisted execution log lines for backtest jobs (worker append + GET /logs).

ALTER TABLE api_backtest_jobs
    ADD COLUMN IF NOT EXISTS execution_log JSONB NOT NULL DEFAULT '[]'::jsonb;
