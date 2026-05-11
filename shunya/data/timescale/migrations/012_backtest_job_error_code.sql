-- Machine-readable failure code for api_backtest_jobs (paired with error_message).

ALTER TABLE api_backtest_jobs
    ADD COLUMN IF NOT EXISTS error_code TEXT;
