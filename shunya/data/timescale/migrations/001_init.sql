-- Shunya local store: symbols, OHLCV hypertable, fundamentals EAV, classifications, ingestion audit.
-- Apply with: python -m shunya.data.timescale.cli migrate

CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS symbols (
    id BIGSERIAL PRIMARY KEY,
    ticker TEXT NOT NULL UNIQUE,
    mic TEXT,
    name TEXT,
    active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ohlcv_bars (
    symbol_id BIGINT NOT NULL REFERENCES symbols (id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    interval TEXT NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    source TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ohlcv_bars_unique_bar UNIQUE (symbol_id, ts, interval, source)
);

SELECT public.create_hypertable(
    'ohlcv_bars',
    'ts',
    if_not_exists => TRUE,
    migrate_data => TRUE,
    chunk_time_interval => INTERVAL '1 month'
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_ts ON ohlcv_bars (symbol_id, ts DESC);

CREATE TABLE IF NOT EXISTS fundamentals_field_values (
    id BIGSERIAL PRIMARY KEY,
    symbol_id BIGINT NOT NULL REFERENCES symbols (id) ON DELETE CASCADE,
    period_end DATE NOT NULL,
    freq TEXT NOT NULL,
    field TEXT NOT NULL,
    value DOUBLE PRECISION,
    source TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT fundamentals_freq_chk CHECK (freq IN ('quarterly', 'yearly')),
    CONSTRAINT fundamentals_unique_cell UNIQUE (symbol_id, period_end, freq, field, source)
);

CREATE INDEX IF NOT EXISTS idx_fund_sym_period ON fundamentals_field_values (symbol_id, period_end DESC);

CREATE TABLE IF NOT EXISTS symbol_classifications (
    id BIGSERIAL PRIMARY KEY,
    symbol_id BIGINT NOT NULL REFERENCES symbols (id) ON DELETE CASCADE,
    as_of DATE NOT NULL,
    sector TEXT,
    industry TEXT,
    sub_industry TEXT,
    source TEXT NOT NULL DEFAULT 'yfinance',
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT symbol_classifications_unique UNIQUE (symbol_id, source, as_of)
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id BIGSERIAL PRIMARY KEY,
    job TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    provider TEXT,
    params JSONB,
    rows_upserted BIGINT,
    status TEXT NOT NULL DEFAULT 'running',
    error TEXT
);
