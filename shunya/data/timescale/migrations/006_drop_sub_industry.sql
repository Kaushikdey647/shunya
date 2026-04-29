-- Remove sub_industry from company / classification rows (sector + industry only).
-- Apply with: python -m shunya.data.timescale.cli migrate

ALTER TABLE symbol_classifications
    DROP COLUMN IF EXISTS sub_industry;
