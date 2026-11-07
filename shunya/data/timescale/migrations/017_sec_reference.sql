-- SEC reference data (company_tickers.json + submissions) and the Alpaca tradability flag.
-- Populate with: shunya-timescale sync-reference

CREATE TABLE IF NOT EXISTS sic_codes (
    sic TEXT PRIMARY KEY,
    description TEXT,
    owner_org TEXT
);

CREATE TABLE IF NOT EXISTS sec_companies (
    cik TEXT PRIMARY KEY,  -- 10-digit, zero padded
    name TEXT,
    entity_type TEXT,
    sic TEXT REFERENCES sic_codes (sic),
    category TEXT,
    fiscal_year_end TEXT,
    state_of_incorporation TEXT,
    ein TEXT,
    exchanges TEXT[],
    fetched_at TIMESTAMPTZ  -- NULL until submissions are fetched
);

CREATE INDEX IF NOT EXISTS idx_sec_companies_sic ON sec_companies (sic);

ALTER TABLE symbols
    ADD COLUMN IF NOT EXISTS cik TEXT REFERENCES sec_companies (cik),
    ADD COLUMN IF NOT EXISTS in_sec_tickers BOOLEAN NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS is_alpaca_tradable BOOLEAN NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS alpaca_asset_id TEXT,
    ADD COLUMN IF NOT EXISTS alpaca_checked_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_symbols_alpaca_tradable ON symbols (ticker) WHERE is_alpaca_tradable;
CREATE INDEX IF NOT EXISTS idx_symbols_cik ON symbols (cik);

-- One row per symbol: sector = SEC owner org (coarse), industry = SIC description.
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
    c.exchanges
FROM symbols s
LEFT JOIN sec_companies c ON c.cik = s.cik
LEFT JOIN sic_codes k ON k.sic = c.sic;
