-- Seeded saved universe SP100: members mirror symbol_index_membership for index_code SP100.
-- Apply with: python -m shunya.data.timescale.cli migrate
-- Idempotent: safe to re-run on every migrate.
-- Constituents appear only after symbol_index_membership is populated (e.g. sync-index-memberships);
-- re-run migrate after sync to pick up new membership rows.
--
-- If a user already created api_universes.name = SP100, the universe INSERT is skipped; the member
-- INSERT still targets that row by name (index members merged on migrate).

INSERT INTO api_universes (name, description)
SELECT
    'SP100',
    'S&P 100; membership copied from symbol_index_membership for index_code SP100.'
WHERE NOT EXISTS (SELECT 1 FROM api_universes WHERE name = 'SP100');

INSERT INTO api_universe_members (universe_id, symbol_id)
SELECT u.id, m.symbol_id
FROM api_universes u
JOIN symbol_index_membership m ON m.index_code = 'SP100'
WHERE u.name = 'SP100'
ON CONFLICT (universe_id, symbol_id) DO NOTHING;
