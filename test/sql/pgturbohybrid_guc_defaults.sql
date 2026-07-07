-- Golden snapshot of every turbohybrid.* GUC's compiled-in default (boot_val).
--
-- pgturbohybrid_gucs.sql pins a representative handful; this test pins the FULL
-- set, so a silent change to any default -- many of which steer recall or
-- resource use (fusion weights, candidate budgets, storage modes, bit widths) --
-- is caught in CI instead of shipping unnoticed.  boot_val is the compiled-in
-- default and is environment-independent.  After an intentional default change,
-- re-bless the expected output (copy results/ to expected/).
SET client_min_messages = warning;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
RESET client_min_messages;

-- Ensure the shared library (and therefore _PG_init -> GUC registration) is
-- loaded into this backend even when the extension already exists from an
-- earlier test in the same regression database.
LOAD 'pgturbohybrid';

SELECT name, boot_val
FROM pg_settings
WHERE name LIKE 'turbohybrid.%'
ORDER BY name;
