-- User-defined equity universes and membership; optional default on api_alphas.
-- Apply with: python -m shunya.data.timescale.cli migrate

CREATE TABLE IF NOT EXISTS api_universes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid (),
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS api_universe_members (
    universe_id UUID NOT NULL REFERENCES api_universes (id) ON DELETE CASCADE,
    symbol_id BIGINT NOT NULL REFERENCES symbols (id) ON DELETE CASCADE,
    added_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (universe_id, symbol_id)
);

CREATE INDEX IF NOT EXISTS idx_api_universe_members_symbol ON api_universe_members (symbol_id);

ALTER TABLE api_alphas
    ADD COLUMN IF NOT EXISTS default_universe_id UUID REFERENCES api_universes (id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_api_alphas_default_universe ON api_alphas (default_universe_id)
    WHERE default_universe_id IS NOT NULL;
