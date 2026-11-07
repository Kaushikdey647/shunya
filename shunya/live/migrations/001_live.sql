-- Live daily runner tables (plain Postgres; no Timescale required).
-- Apply with: shunya-live migrate

CREATE TABLE IF NOT EXISTS live_control (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    state TEXT NOT NULL DEFAULT 'armed' CHECK (state IN ('armed', 'halted')),
    level TEXT CHECK (level IN ('halt', 'cancel', 'flatten')),
    reason TEXT,
    actor TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO live_control (id, state)
VALUES (1, 'armed')
ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS live_audit (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    detail JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS live_heartbeat (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_tick_at TIMESTAMPTZ,
    last_result TEXT
);

INSERT INTO live_heartbeat (id)
VALUES (1)
ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS live_strategy_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid (),
    name TEXT NOT NULL,
    alpha_source TEXT NOT NULL,
    finstrat_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    tickers TEXT[] NOT NULL,
    lookback_days INTEGER NOT NULL DEFAULT 400 CHECK (lookback_days > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS live_books (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid (),
    name TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive')),
    capital_fraction DOUBLE PRECISION NOT NULL DEFAULT 1.0
        CHECK (capital_fraction > 0 AND capital_fraction <= 1),
    risk_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    hwm_equity DOUBLE PRECISION,
    last_equity DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- v1: one active book per Alpaca account (positions are netted per symbol by the broker).
CREATE UNIQUE INDEX IF NOT EXISTS idx_live_books_one_active ON live_books ((status))
    WHERE status = 'active';

CREATE TABLE IF NOT EXISTS live_sleeves (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid (),
    book_id UUID NOT NULL REFERENCES live_books (id) ON DELETE CASCADE,
    version_id UUID NOT NULL REFERENCES live_strategy_versions (id),
    weight DOUBLE PRECISION NOT NULL CHECK (weight >= 0 AND weight <= 1),
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'paused', 'draining', 'retired')),
    ramp_days INTEGER NOT NULL DEFAULT 1 CHECK (ramp_days >= 1),
    runs_completed INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_live_sleeves_book ON live_sleeves (book_id);

CREATE TABLE IF NOT EXISTS live_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid (),
    book_id UUID NOT NULL REFERENCES live_books (id) ON DELETE CASCADE,
    session_date DATE NOT NULL,
    signal_date DATE NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'submitted', 'reconciling', 'reconciled', 'failed', 'skipped')),
    error TEXT,
    equity DOUBLE PRECISION,
    detail JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (book_id, session_date)
);

CREATE TABLE IF NOT EXISTS live_sleeve_targets (
    run_id UUID NOT NULL REFERENCES live_runs (id) ON DELETE CASCADE,
    sleeve_id UUID NOT NULL REFERENCES live_sleeves (id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    target_usd DOUBLE PRECISION NOT NULL,
    target_shares DOUBLE PRECISION NOT NULL,
    prev_shares DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (run_id, sleeve_id, symbol)
);

CREATE TABLE IF NOT EXISTS live_orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid (),
    run_id UUID NOT NULL REFERENCES live_runs (id) ON DELETE CASCADE,
    client_order_id TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    qty INTEGER NOT NULL CHECK (qty > 0),
    alpaca_id TEXT,
    status TEXT NOT NULL,
    filled_qty DOUBLE PRECISION NOT NULL DEFAULT 0,
    filled_avg_price DOUBLE PRECISION,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_live_orders_run ON live_orders (run_id);

CREATE TABLE IF NOT EXISTS live_sleeve_positions (
    sleeve_id UUID NOT NULL REFERENCES live_sleeves (id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    shares DOUBLE PRECISION NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (sleeve_id, symbol)
);

CREATE TABLE IF NOT EXISTS live_sleeve_snapshots (
    sleeve_id UUID NOT NULL REFERENCES live_sleeves (id) ON DELETE CASCADE,
    session_date DATE NOT NULL,
    market_value DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (sleeve_id, session_date)
);
