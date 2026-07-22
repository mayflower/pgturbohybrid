use strict;
use warnings FATAL => 'all';
use Test::More;
use File::Temp qw(tempfile);

BEGIN
{
	eval {
		require PostgreSQL::Test::Cluster;
		PostgreSQL::Test::Cluster->import();
		1;
	} or plan skip_all => 'PostgreSQL::Test::Cluster is not available';
}

my $node = PostgreSQL::Test::Cluster->new('multivector_dump_restore');
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE DATABASE pgturbohybrid_mv_dump_src;');
$node->safe_psql('pgturbohybrid_mv_dump_src', q(
	CREATE EXTENSION vector;
	CREATE EXTENSION pgturbohybrid; CREATE EXTENSION pgturbohybrid_experimental;

	CREATE TABLE mv_dump_docs (
		id int PRIMARY KEY,
		colbert turbohybrid_multivector
	);

	INSERT INTO mv_dump_docs VALUES
		(1, turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector])),
		(2, turbohybrid_multivector(ARRAY['[0.5,0.5]'::vector, '[0.25,0.75]'::vector]));

	CREATE INDEX mv_dump_docs_idx ON mv_dump_docs USING turbohybrid
		(colbert multivector_maxsim_ip_turbohybrid_ops);
));

my ($dump_fh, $dump_path) = tempfile('pgturbohybrid-mv-dump-XXXXXX',
	SUFFIX => '.sql', TMPDIR => 1, UNLINK => 1);
close $dump_fh;

my $dump_rc = system('pg_dump', '-f', $dump_path,
	$node->connstr('pgturbohybrid_mv_dump_src'));
is($dump_rc, 0, 'pg_dump succeeds for turbohybrid_multivector table');

$node->safe_psql('postgres', 'CREATE DATABASE pgturbohybrid_mv_dump_dst;');
my $restore_rc = system('psql', '-X', '-v', 'ON_ERROR_STOP=1', '-f',
	$dump_path, $node->connstr('pgturbohybrid_mv_dump_dst'));
is($restore_rc, 0, 'restore succeeds for turbohybrid_multivector table');

is($node->safe_psql('pgturbohybrid_mv_dump_dst', q(
	SELECT string_agg(
		id::text || ':' ||
		turbohybrid_multivector_dims(colbert)::text || ':' ||
		turbohybrid_multivector_count(colbert)::text || ':' ||
		colbert::text,
		';' ORDER BY id
	)
	FROM mv_dump_docs;
)), '1:2:2:turbohybrid_multivector(dim=2,count=2,values=[[1,0],[0,1]]);2:2:2:turbohybrid_multivector(dim=2,count=2,values=[[0.5,0.5],[0.25,0.75]])',
	'restored multivector dims, counts, and text output match');

is($node->safe_psql('pgturbohybrid_mv_dump_dst', q(
	WITH q AS (
		SELECT turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]) AS mv
	)
	SELECT turbohybrid_multivector_maxsim(q.mv, colbert)::text
	FROM mv_dump_docs, q
	WHERE id = 1;
)), '2', 'restored multivector exact MaxSim matches');

is($node->safe_psql('pgturbohybrid_mv_dump_dst', q(
	SET enable_seqscan = off;
	SELECT id
	FROM mv_dump_docs
	ORDER BY colbert <~> turbohybrid_experimental_query(
		multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]),
		dense_k => 4,
		final_k => 1
	)
	LIMIT 1;
)), '1', 'restored multivector index returns expected nearest row');

done_testing();
