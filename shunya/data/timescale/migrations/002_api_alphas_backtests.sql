-- FastAPI backtest service: alpha definitions and async job queue.
-- Apply with: shunya-timescale migrate

CREATE TABLE IF NOT EXISTS api_alphas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid (),
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    import_ref TEXT NOT NULL,
    finstrat_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS api_backtest_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid (),
    alpha_id UUID NOT NULL REFERENCES api_alphas (id) ON DELETE RESTRICT,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
    request_payload JSONB NOT NULL,
    result_payload JSONB,
    error_message TEXT,
    result_summary JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_api_backtest_jobs_alpha_created
    ON api_backtest_jobs (alpha_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_api_backtest_jobs_status_created
    ON api_backtest_jobs (status, created_at);
