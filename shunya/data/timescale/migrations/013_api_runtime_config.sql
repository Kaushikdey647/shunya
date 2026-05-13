-- Singleton JSON payload for operator-tunable API settings (non-secrets).
-- Edited via PATCH /settings/app; merged with SHUNYA_API_* env at read time.

CREATE TABLE IF NOT EXISTS api_runtime_config (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO api_runtime_config (id, payload)
VALUES (1, '{}'::jsonb)
ON CONFLICT (id) DO NOTHING;
