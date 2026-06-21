-- Regression coverage for turbohybrid.* GUC registration.
--
-- After the GUC/profile machinery was extracted from pgturbohybrid_am.c into
-- pgturbohybrid_guc.c, this test guards that a representative set of
-- turbohybrid.* GUCs still register with their expected boot-time defaults and
-- their default-profile (latency) settings.
SET client_min_messages = warning;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
RESET client_min_messages;

-- Ensure the shared library (and therefore _PG_init -> GUC registration) is
-- loaded into this backend even when the extension already exists from an
-- earlier test in the same regression database.
LOAD 'pgturbohybrid';

SELECT name, setting, boot_val
FROM pg_settings
WHERE name IN ('turbohybrid.profile',
               'turbohybrid.default_dense_k',
               'turbohybrid.default_bm25_k',
               'turbohybrid.default_rrf_k',
               'turbohybrid.enable_wand',
               'turbohybrid.simd',
               'turbohybrid.native_cache_max_mb')
ORDER BY name;
