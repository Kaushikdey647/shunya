-- Equity index catalog and symbol membership; extend symbol_classifications with yfinance info columns.
-- Apply with: python -m shunya.data.timescale.cli migrate

CREATE TABLE IF NOT EXISTS equity_indexes (
    code TEXT PRIMARY KEY,
    display_name TEXT NOT NULL UNIQUE
);

INSERT INTO equity_indexes (code, display_name)
VALUES
    ('SP100', 'S&P 100'),
    ('SP500', 'S&P 500'),
    ('SP600', 'S&P 600')
ON CONFLICT (code) DO NOTHING;

CREATE TABLE IF NOT EXISTS symbol_index_membership (
    symbol_id BIGINT NOT NULL REFERENCES symbols (id) ON DELETE CASCADE,
    index_code TEXT NOT NULL REFERENCES equity_indexes (code) ON DELETE CASCADE,
    PRIMARY KEY (symbol_id, index_code)
);

CREATE INDEX IF NOT EXISTS idx_symbol_index_membership_index ON symbol_index_membership (index_code);

ALTER TABLE symbol_classifications
    ADD COLUMN IF NOT EXISTS sector_disp TEXT,
    ADD COLUMN IF NOT EXISTS sector_key TEXT,
    ADD COLUMN IF NOT EXISTS industry_disp TEXT,
    ADD COLUMN IF NOT EXISTS industry_key TEXT,
    ADD COLUMN IF NOT EXISTS quote_type TEXT,
    ADD COLUMN IF NOT EXISTS type_disp TEXT,
    ADD COLUMN IF NOT EXISTS exchange TEXT,
    ADD COLUMN IF NOT EXISTS full_exchange_name TEXT,
    ADD COLUMN IF NOT EXISTS currency TEXT,
    ADD COLUMN IF NOT EXISTS region TEXT,
    ADD COLUMN IF NOT EXISTS market TEXT,
    ADD COLUMN IF NOT EXISTS country TEXT,
    ADD COLUMN IF NOT EXISTS state TEXT,
    ADD COLUMN IF NOT EXISTS city TEXT,
    ADD COLUMN IF NOT EXISTS zip TEXT,
    ADD COLUMN IF NOT EXISTS website TEXT,
    ADD COLUMN IF NOT EXISTS phone TEXT,
    ADD COLUMN IF NOT EXISTS ir_website TEXT,
    ADD COLUMN IF NOT EXISTS long_name TEXT,
    ADD COLUMN IF NOT EXISTS short_name TEXT,
    ADD COLUMN IF NOT EXISTS full_time_employees BIGINT;
