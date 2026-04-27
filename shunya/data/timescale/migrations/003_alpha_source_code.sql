-- Inline alpha source in DB (Monaco editor in UI). Apply with: shunya-timescale migrate
ALTER TABLE api_alphas
    ADD COLUMN IF NOT EXISTS source_code TEXT;

-- Allow code-only rows: import_ref may be null when source_code is set
ALTER TABLE api_alphas
    ALTER COLUMN import_ref DROP NOT NULL;

ALTER TABLE api_alphas
    ADD CONSTRAINT api_alphas_ref_or_source_chk CHECK (
        (import_ref IS NOT NULL AND btrim(import_ref) <> '')
        OR (source_code IS NOT NULL AND btrim(source_code) <> '')
    );
