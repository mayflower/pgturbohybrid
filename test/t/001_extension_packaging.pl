use strict;
use warnings FATAL => 'all';
use Test::More;

BEGIN
{
	eval {
		require PostgreSQL::Test::Cluster;
		PostgreSQL::Test::Cluster->import();
		1;
	} or plan skip_all => 'PostgreSQL::Test::Cluster is not available';
}

my $node = PostgreSQL::Test::Cluster->new('node');
$node->init;
$node->start;

sub feature_objects
{
	my ($db) = @_;

	return $node->safe_psql($db, q(
		SELECT concat_ws(',',
			to_regtype('turbohybrid_query') IS NOT NULL,
			EXISTS (SELECT 1 FROM pg_am WHERE amname = 'turbohybrid'),
			EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'turbohybrid_query'),
			EXISTS (SELECT 1 FROM pg_opclass WHERE opcname = 'vector_l2_turbohybrid_ops'),
			EXISTS (SELECT 1 FROM pg_opclass WHERE opcname = 'vector_ip_turbohybrid_ops'),
			EXISTS (SELECT 1 FROM pg_opclass WHERE opcname = 'vector_cosine_turbohybrid_ops'),
			EXISTS (SELECT 1 FROM pg_opclass WHERE opcname = 'bm25_tsvector_turbohybrid_ops'),
			EXISTS (SELECT 1 FROM pg_operator WHERE oprname = '<~>'),
			EXISTS (SELECT 1 FROM pg_operator WHERE oprname = '<~->'),
			EXISTS (SELECT 1 FROM pg_operator WHERE oprname = '<~#>'),
			to_regtype('vector') IS NOT NULL,
			NOT EXISTS (
				SELECT 1
				FROM pg_extension e
				WHERE e.extname = 'pgturbohybrid'
				  AND e.extversion <> '0.2.0'
			)
		);
	));
}

$node->safe_psql('postgres', 'CREATE DATABASE pgturbohybrid_create;');
$node->safe_psql('pgturbohybrid_create', 'CREATE EXTENSION vector;');
my $vector_version = $node->safe_psql('pgturbohybrid_create',
	"SELECT extversion FROM pg_extension WHERE extname = 'vector';");
$node->safe_psql('pgturbohybrid_create', 'CREATE EXTENSION pgturbohybrid;');

is($node->safe_psql('pgturbohybrid_create', q(
	SELECT concat_ws(',',
		to_regtype('turbohybrid_sparse_vector') IS NULL,
		to_regtype('turbohybrid_multivector') IS NULL,
		NOT EXISTS (SELECT 1 FROM pg_opclass WHERE opcname = 'sparse_ip_turbohybrid_ops'),
		NOT EXISTS (SELECT 1 FROM pg_opclass WHERE opcname IN
			('multivector_cosine_turbohybrid_ops',
			 'multivector_maxsim_ip_turbohybrid_ops')));
)), 't,t,t,t', 'core-only install excludes experimental catalog objects');

is(feature_objects('pgturbohybrid_create'), 't,t,t,t,t,t,t,t,t,t,t,t',
	'CREATE EXTENSION installs only pgturbohybrid objects on top of vector');

is($node->safe_psql('pgturbohybrid_create', q(
	SELECT concat_ws(',',
		NOT EXISTS (
			SELECT 1
			FROM pg_depend d
			JOIN pg_extension e ON e.oid = d.refobjid
			WHERE e.extname = 'pgturbohybrid'
			  AND d.deptype = 'e'
			  AND d.classid = 'pg_type'::regclass
			  AND d.objid = 'vector'::regtype::oid
		),
		EXISTS (
			SELECT 1
			FROM pg_depend d
			JOIN pg_extension e ON e.oid = d.refobjid
			JOIN pg_operator o ON o.oid = d.objid
			WHERE e.extname = 'pgturbohybrid'
			  AND d.deptype = 'e'
			  AND d.classid = 'pg_operator'::regclass
			  AND o.oprname IN ('<~>', '<~->', '<~#>')
		),
		NOT EXISTS (
			SELECT 1
			FROM pg_depend d
			JOIN pg_extension e ON e.oid = d.refobjid
			JOIN pg_proc p ON p.oid = d.objid
			WHERE e.extname = 'pgturbohybrid'
			  AND d.deptype = 'e'
			  AND d.classid = 'pg_proc'::regclass
			  AND p.proname LIKE '%debug%'
		)
	);
)), 't,t,t', 'pgturbohybrid owns no vector type, owns its hybrid operators, and exposes no debug functions');

is($node->safe_psql('pgturbohybrid_create',
	"SELECT extversion FROM pg_extension WHERE extname = 'vector';"),
	$vector_version, 'CREATE EXTENSION pgturbohybrid does not alter vector');

$node->safe_psql('pgturbohybrid_create', 'CREATE EXTENSION pgturbohybrid_experimental;');
is($node->safe_psql('pgturbohybrid_create', q(
	SELECT concat_ws(',',
		to_regtype('turbohybrid_sparse_vector') IS NOT NULL,
		to_regtype('turbohybrid_multivector') IS NOT NULL,
		EXISTS (SELECT 1 FROM pg_opclass WHERE opcname = 'sparse_ip_turbohybrid_ops'),
		EXISTS (SELECT 1 FROM pg_opclass WHERE opcname = 'multivector_maxsim_ip_turbohybrid_ops'));
)), 't,t,t,t', 'companion install adds sparse and multivector objects');

$node->safe_psql('pgturbohybrid_create', 'DROP EXTENSION pgturbohybrid_experimental;');

$node->safe_psql('pgturbohybrid_create', 'DROP EXTENSION pgturbohybrid;');

is($node->safe_psql('pgturbohybrid_create', q(
	SELECT concat_ws(',',
		NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pgturbohybrid'),
		to_regtype('turbohybrid_query') IS NULL,
		NOT EXISTS (SELECT 1 FROM pg_am WHERE amname = 'turbohybrid'),
		NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname LIKE 'pgturbohybrid%'),
		NOT EXISTS (SELECT 1 FROM pg_opclass WHERE opcname LIKE '%pgturbohybrid%'),
		NOT EXISTS (
			SELECT 1
			FROM pg_operator o
			JOIN pg_proc p ON p.oid = o.oprcode
			WHERE p.proname LIKE 'pgturbohybrid%'
		),
		EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector'),
		to_regtype('vector') IS NOT NULL,
		('[1,2,3]'::vector <-> '[1,2,4]'::vector) = 1
	);
)), 't,t,t,t,t,t,t,t,t', 'DROP EXTENSION pgturbohybrid removes pgturbohybrid objects and leaves vector usable');

done_testing();
