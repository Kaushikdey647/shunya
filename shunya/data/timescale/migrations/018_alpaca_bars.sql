-- Alpaca daily bars (raw, unadjusted), corporate actions, trading calendar and per-symbol coverage.
-- Adjusted series are computed at read time (shunya.data.adjustments), so stored history never goes stale.

CREATE TABLE IF NOT EXISTS alpaca_bars_daily (
    symbol_id BIGINT NOT NULL REFERENCES symbols (id) ON DELETE CASCADE,
    session_date DATE NOT NULL,
    feed TEXT NOT NULL,  -- sip | iex
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    vwap DOUBLE PRECISION,
    trade_count BIGINT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT alpaca_bars_daily_pk PRIMARY KEY (symbol_id, session_date, feed)
);

SELECT public.create_hypertable(
    'alpaca_bars_daily',
    'session_date',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 year'
);

CREATE INDEX IF NOT EXISTS idx_alpaca_bars_daily_feed_date ON alpaca_bars_daily (feed, session_date DESC);

-- One row per Alpaca corporate action (all types). ``symbol`` is the affected (old / source /
-- acquiree) symbol, ``other_symbol`` the new / acquirer one. ``ex_date`` falls back to
-- effective_date, then process_date, for types without an ex date (mergers, name changes, removals).
CREATE TABLE IF NOT EXISTS alpaca_corporate_actions (
    id TEXT PRIMARY KEY,
    action_type TEXT NOT NULL,  -- forward_split | reverse_split | unit_split | stock_dividend | cash_dividend | spin_off | ...
    symbol TEXT NOT NULL,
    symbol_id BIGINT REFERENCES symbols (id) ON DELETE SET NULL,
    other_symbol TEXT,
    ex_date DATE NOT NULL,
    process_date DATE,
    record_date DATE,
    payable_date DATE,
    old_rate DOUBLE PRECISION,
    new_rate DOUBLE PRECISION,
    rate DOUBLE PRECISION,  -- cash per share (dividend / cash merger) or stock-dividend fraction
    special BOOLEAN,
    payload JSONB NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_alpaca_ca_symbol_ex ON alpaca_corporate_actions (symbol_id, ex_date);
CREATE INDEX IF NOT EXISTS idx_alpaca_ca_symbol_text_ex ON alpaca_corporate_actions (symbol, ex_date);

-- Alpaca /v2/calendar: the session list coverage and gaps are measured against.
CREATE TABLE IF NOT EXISTS market_sessions (
    session_date DATE PRIMARY KEY,
    open_at TIMESTAMPTZ NOT NULL,
    close_at TIMESTAMPTZ NOT NULL
);

ALTER TABLE symbols
    ADD COLUMN IF NOT EXISTS has_alpaca_data BOOLEAN NOT NULL DEFAULT false;

CREATE INDEX IF NOT EXISTS idx_symbols_has_alpaca_data ON symbols (ticker) WHERE has_alpaca_data;

CREATE TABLE IF NOT EXISTS symbol_data_coverage (
    symbol_id BIGINT NOT NULL REFERENCES symbols (id) ON DELETE CASCADE,
    feed TEXT NOT NULL,
    first_session DATE NOT NULL,
    last_session DATE NOT NULL,
    n_sessions INTEGER NOT NULL,
    missing_sessions INTEGER NOT NULL,  -- calendar sessions in [first, last] without a bar
    refreshed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT symbol_data_coverage_pk PRIMARY KEY (symbol_id, feed)
);

-- Recompute coverage for one feed (optionally only some symbols) and set symbols.has_alpaca_data.
CREATE OR REPLACE FUNCTION refresh_symbol_data_coverage(p_feed TEXT, p_symbol_ids BIGINT[] DEFAULT NULL)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    n INTEGER;
BEGIN
    INSERT INTO symbol_data_coverage (symbol_id, feed, first_session, last_session, n_sessions, missing_sessions, refreshed_at)
    SELECT
        b.symbol_id,
        p_feed,
        min(b.session_date),
        max(b.session_date),
        count(*),
        GREATEST(
            (SELECT count(*) FROM market_sessions m
             WHERE m.session_date BETWEEN min(b.session_date) AND max(b.session_date)) - count(*),
            0
        ),
        now()
    FROM alpaca_bars_daily b
    WHERE b.feed = p_feed
      AND (p_symbol_ids IS NULL OR b.symbol_id = ANY (p_symbol_ids))
    GROUP BY b.symbol_id
    ON CONFLICT (symbol_id, feed) DO UPDATE SET
        first_session = EXCLUDED.first_session,
        last_session = EXCLUDED.last_session,
        n_sessions = EXCLUDED.n_sessions,
        missing_sessions = EXCLUDED.missing_sessions,
        refreshed_at = EXCLUDED.refreshed_at;
    GET DIAGNOSTICS n = ROW_COUNT;

    UPDATE symbols s
    SET has_alpaca_data = EXISTS (SELECT 1 FROM symbol_data_coverage c WHERE c.symbol_id = s.id)
    WHERE p_symbol_ids IS NULL OR s.id = ANY (p_symbol_ids);
    RETURN n;
END;
$$;

-- Per-event factors, computed at ingest from the raw prior close (shunya.data.adjustments.event_factors).
-- Bars before ex_date are multiplied by the product of later events' factors. NULL = no price effect / unknown.
ALTER TABLE alpaca_corporate_actions
    ADD COLUMN IF NOT EXISTS price_factor DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS volume_factor DOUBLE PRECISION;

-- symbol_reference gains has_alpaca_data (appended: CREATE OR REPLACE VIEW only allows new trailing columns).
CREATE OR REPLACE VIEW symbol_reference AS
SELECT
    s.id AS symbol_id,
    s.ticker,
    COALESCE(c.name, s.name) AS name,
    s.cik,
    s.is_alpaca_tradable,
    c.sic,
    k.owner_org AS sector,
    k.description AS industry,
    c.entity_type,
    c.category,
    c.exchanges,
    s.has_alpaca_data
FROM symbols s
LEFT JOIN sec_companies c ON c.cik = s.cik
LEFT JOIN sic_codes k ON k.sic = c.sic;
