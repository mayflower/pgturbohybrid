-- The 0.2.0 boundary separates objects formerly owned by one extension into
-- core and experimental companions. PostgreSQL cannot atomically create the
-- companion extension from inside an extension update script. Refuse before
-- changing ownership or catalogs so the 0.1.2 installation remains intact.
DO $pgturbohybrid_split$
BEGIN
	RAISE EXCEPTION 'pgturbohybrid 0.2.0 requires an explicit core/experimental migration'
		USING ERRCODE = '0A000',
			  DETAIL = 'The existing 0.1.2 extension owns sparse and multivector objects that move to pgturbohybrid_experimental.',
			  HINT = 'Dump dependent data, install pgturbohybrid 0.2.0 and pgturbohybrid_experimental 0.2.0 in a new database, then restore. The failed ALTER EXTENSION is transactional and leaves 0.1.2 unchanged.';
END
$pgturbohybrid_split$;
