use strict;
use warnings FATAL => 'all';
use Test::More;

BEGIN
{
	eval {
		require PostgreSQL::Test::Cluster;
		PostgreSQL::Test::Cluster->import();
		1;
	} or plan skip_all => 'PostgreSQL TAP test modules are not available';
}

my $node = PostgreSQL::Test::Cluster->new('node');
$node->init;
$node->start;

sub feature_objects
{
	my ($db) = @_;

	return $node->safe_psql($db, q(
		SELECT concat_ws(',',
			to_regtype('hybrid_query') IS NOT NULL,
			EXISTS (SELECT 1 FROM pg_am WHERE amname = 'turboquant'),
			EXISTS (SELECT 1 FROM pg_am WHERE amname = 'turbohybrid'),
			EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'hybrid_query'),
			EXISTS (SELECT 1 FROM pg_opclass WHERE opcname = 'bm25_tsvector_ops'),
			NOT EXISTS (
				SELECT 1
				FROM pg_proc p
				JOIN pg_depend d ON d.objid = p.oid AND d.deptype = 'e'
				JOIN pg_extension e ON e.oid = d.refobjid
				WHERE e.extname = 'vector' AND p.proname LIKE '%debug%'
			),
			NOT EXISTS (
				SELECT 1
				FROM pg_proc p
				JOIN pg_depend d ON d.objid = p.oid AND d.deptype = 'e'
				JOIN pg_extension e ON e.oid = d.refobjid
				WHERE e.extname = 'vector' AND p.proname IN (
					'tq_last_scan_stats',
					'tq_index_stats',
					'tq_simd_capabilities',
					'tq_last_simd_stats',
					'hybrid_last_scan_stats'
				)
			)
		);
	));
}

sub extension_version
{
	my ($db) = @_;

	return $node->safe_psql($db,
		"SELECT extversion FROM pg_extension WHERE extname = 'vector';");
}

my $present = 't,t,t,t,t,t,t';

$node->safe_psql('postgres', 'CREATE DATABASE vector_create;');
$node->safe_psql('vector_create', 'CREATE EXTENSION vector;');

is(extension_version('vector_create'), '0.8.2',
	'CREATE EXTENSION keeps upstream default version');
is(feature_objects('vector_create'), $present,
	'CREATE EXTENSION from development SQL includes TurboHybrid objects');

$node->safe_psql('vector_create', 'ALTER EXTENSION vector UPDATE;');

is(extension_version('vector_create'), '0.8.2',
	'ALTER EXTENSION UPDATE without a target keeps upstream default version');
is(feature_objects('vector_create'), $present,
	'plain update preserves development SQL objects');

is($node->safe_psql('vector_create', q(
	SELECT path IS NOT NULL
	FROM pg_extension_update_paths('vector')
	WHERE source = '0.8.2' AND target = '0.8.2-turbohybrid';
)), 't', 'explicit maintainer-owned placeholder upgrade path is available');

done_testing();
