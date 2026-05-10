-- Wide fundamentals (daily / quarterly / annual) and event tables.
-- Legacy ``fundamentals_field_values`` (EAV) remains for dual-write / backfill compatibility.

CREATE TABLE IF NOT EXISTS fundamentals_daily (
    symbol_id BIGINT NOT NULL REFERENCES symbols (id) ON DELETE CASCADE,
    as_of_ts TIMESTAMPTZ NOT NULL,
    source TEXT NOT NULL,
    market_cap DOUBLE PRECISION,
    enterprise_value DOUBLE PRECISION,
    trailing_pe DOUBLE PRECISION,
    forward_pe DOUBLE PRECISION,
    peg_ratio DOUBLE PRECISION,
    price_to_book DOUBLE PRECISION,
    dividend_yield DOUBLE PRECISION,
    beta DOUBLE PRECISION,
    shares_outstanding DOUBLE PRECISION,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT fundamentals_daily_unique UNIQUE (symbol_id, as_of_ts, source)
);

SELECT public.create_hypertable(
    'fundamentals_daily',
    'as_of_ts',
    if_not_exists => TRUE,
    migrate_data => TRUE,
    chunk_time_interval => INTERVAL '3 months'
);

CREATE INDEX IF NOT EXISTS idx_fund_daily_symbol_ts
    ON fundamentals_daily (symbol_id, as_of_ts DESC);

CREATE TABLE IF NOT EXISTS fundamentals_quarterly (
    symbol_id BIGINT NOT NULL REFERENCES symbols (id) ON DELETE CASCADE,
    fiscal_period_end DATE NOT NULL,
    source TEXT NOT NULL,
    revenue DOUBLE PRECISION,
    net_income DOUBLE PRECISION,
    eps_diluted DOUBLE PRECISION,
    operating_cash_flow DOUBLE PRECISION,
    free_cash_flow DOUBLE PRECISION,
    total_assets DOUBLE PRECISION,
    total_equity DOUBLE PRECISION,
    total_debt DOUBLE PRECISION,
    current_ratio DOUBLE PRECISION,
    gross_margin DOUBLE PRECISION,
    operating_margin DOUBLE PRECISION,
    return_on_assets DOUBLE PRECISION,
    return_on_equity DOUBLE PRECISION,
    debt_to_equity DOUBLE PRECISION,
    free_cash_flow_yield DOUBLE PRECISION,
    price_to_earnings DOUBLE PRECISION,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT fundamentals_quarterly_unique UNIQUE (symbol_id, fiscal_period_end, source)
);

CREATE INDEX IF NOT EXISTS idx_fund_q_symbol_period
    ON fundamentals_quarterly (symbol_id, fiscal_period_end DESC);

CREATE TABLE IF NOT EXISTS fundamentals_annual (
    symbol_id BIGINT NOT NULL REFERENCES symbols (id) ON DELETE CASCADE,
    fiscal_period_end DATE NOT NULL,
    source TEXT NOT NULL,
    revenue DOUBLE PRECISION,
    net_income DOUBLE PRECISION,
    eps_diluted DOUBLE PRECISION,
    operating_cash_flow DOUBLE PRECISION,
    free_cash_flow DOUBLE PRECISION,
    total_assets DOUBLE PRECISION,
    total_equity DOUBLE PRECISION,
    total_debt DOUBLE PRECISION,
    current_ratio DOUBLE PRECISION,
    gross_margin DOUBLE PRECISION,
    operating_margin DOUBLE PRECISION,
    return_on_assets DOUBLE PRECISION,
    return_on_equity DOUBLE PRECISION,
    debt_to_equity DOUBLE PRECISION,
    free_cash_flow_yield DOUBLE PRECISION,
    price_to_earnings DOUBLE PRECISION,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT fundamentals_annual_unique UNIQUE (symbol_id, fiscal_period_end, source)
);

CREATE INDEX IF NOT EXISTS idx_fund_a_symbol_period
    ON fundamentals_annual (symbol_id, fiscal_period_end DESC);

CREATE TABLE IF NOT EXISTS corporate_actions (
    id BIGSERIAL PRIMARY KEY,
    symbol_id BIGINT NOT NULL REFERENCES symbols (id) ON DELETE CASCADE,
    action_ts TIMESTAMPTZ NOT NULL,
    kind TEXT NOT NULL,
    amount DOUBLE PRECISION,
    split_ratio DOUBLE PRECISION,
    currency TEXT,
    source TEXT NOT NULL,
    raw JSONB,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT corporate_actions_kind_chk CHECK (kind IN ('dividend', 'split')),
    CONSTRAINT corporate_actions_unique UNIQUE (symbol_id, action_ts, kind, source)
);

CREATE INDEX IF NOT EXISTS idx_corp_actions_symbol_ts
    ON corporate_actions (symbol_id, action_ts DESC);

CREATE TABLE IF NOT EXISTS insider_transactions (
    id BIGSERIAL PRIMARY KEY,
    symbol_id BIGINT NOT NULL REFERENCES symbols (id) ON DELETE CASCADE,
    report_date DATE,
    transaction_start_date DATE,
    owner_name TEXT,
    transaction_type TEXT,
    shares DOUBLE PRECISION,
    value DOUBLE PRECISION,
    position TEXT,
    source TEXT NOT NULL,
    row_fingerprint TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT insider_transactions_unique UNIQUE (symbol_id, row_fingerprint, source)
);

CREATE INDEX IF NOT EXISTS idx_insider_symbol_report
    ON insider_transactions (symbol_id, report_date DESC);

CREATE TABLE IF NOT EXISTS earnings_dates (
    symbol_id BIGINT NOT NULL REFERENCES symbols (id) ON DELETE CASCADE,
    earnings_date DATE NOT NULL,
    source TEXT NOT NULL,
    eps_estimate DOUBLE PRECISION,
    reported_eps DOUBLE PRECISION,
    surprise_pct DOUBLE PRECISION,
    quarter_label TEXT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT earnings_dates_unique UNIQUE (symbol_id, earnings_date, source)
);

CREATE INDEX IF NOT EXISTS idx_earnings_symbol_date
    ON earnings_dates (symbol_id, earnings_date DESC);
