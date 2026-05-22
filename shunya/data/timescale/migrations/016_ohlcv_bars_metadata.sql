-- Optional JSONB for feed tier, adjustment convention, session flags (see docs/market_data_routing.md).
ALTER TABLE ohlcv_bars ADD COLUMN IF NOT EXISTS metadata JSONB;
