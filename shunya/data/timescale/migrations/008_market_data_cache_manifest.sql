-- OHLCV refresh manifest (per symbol, interval, source) and yfinance instrument JSON cache.

CREATE TABLE IF NOT EXISTS ohlcv_symbol_interval_refresh (
    symbol_id BIGINT NOT NULL REFERENCES symbols (id) ON DELETE CASCADE,
    interval TEXT NOT NULL,
    source TEXT NOT NULL,
    last_refresh_at TIMESTAMPTZ NOT NULL,
    last_error TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol_id, interval, source)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_refresh_last_at
    ON ohlcv_symbol_interval_refresh (last_refresh_at DESC);

CREATE TABLE IF NOT EXISTS instrument_yfinance_documents (
    symbol_id BIGINT NOT NULL REFERENCES symbols (id) ON DELETE CASCADE,
    source TEXT NOT NULL DEFAULT 'yfinance',
    resource_type TEXT NOT NULL,
    resource_key TEXT NOT NULL DEFAULT '',
    payload JSONB NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT instrument_yfinance_documents_resource_chk CHECK (
        resource_type IN (
            'overview',
            'financials_income',
            'financials_balance',
            'financials_cashflow',
            'holders',
            'option_expirations',
            'option_chain'
        )
    ),
    CONSTRAINT instrument_yfinance_documents_unique UNIQUE (symbol_id, source, resource_type, resource_key)
);

CREATE INDEX IF NOT EXISTS idx_instrument_yf_docs_symbol_fetched
    ON instrument_yfinance_documents (symbol_id, fetched_at DESC);
