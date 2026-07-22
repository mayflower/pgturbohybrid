# Physical replication, promotion, and base-backup recovery contract.
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

sub validate
{
	my ($node) = @_;
	return $node->safe_psql('postgres',
		"SELECT (turbohybrid_validate_index('repl_idx'::regclass, true)->>'ok')::boolean;");
}

sub nearest
{
	my ($node) = @_;
	return $node->safe_psql('postgres', q(
		SET enable_seqscan=off;
		SELECT id FROM repl_docs
		ORDER BY embedding <~-> turbohybrid_query(
			vector_query=>'[0,0,0]'::vector, dense_k=>64, bm25_k=>0, final_k=>1)
		LIMIT 1;
	));
}

my $primary = PostgreSQL::Test::Cluster->new('repl_primary');
$primary->init(allows_streaming => 1);
$primary->start;
$primary->safe_psql('postgres', q(
	CREATE EXTENSION vector;
	CREATE EXTENSION pgturbohybrid;
	CREATE TABLE repl_docs(id int PRIMARY KEY, embedding vector(3));
	INSERT INTO repl_docs SELECT g, ('[' || g || ',0,0]')::vector FROM generate_series(1,128) g;
	CREATE INDEX repl_idx ON repl_docs USING turbohybrid (embedding vector_l2_turbohybrid_ops)
		WITH (quantization_bits=4, exact_storage=on);
	DELETE FROM repl_docs WHERE id % 11 = 0;
	VACUUM repl_docs;
));
is(validate($primary), 't', 'primary deep validator passes before backup');

$primary->backup('physical');
my $standby = PostgreSQL::Test::Cluster->new('repl_standby');
$standby->init_from_backup($primary, 'physical', has_streaming => 1);
$standby->start;
$primary->safe_psql('postgres', "INSERT INTO repl_docs VALUES (0, '[0,0,0]');");
$primary->wait_for_catchup($standby);
is(nearest($standby), '0', 'standby replays insert and serves native query');
is(validate($standby), 't', 'standby deep validator passes');
$standby->safe_psql('postgres', "SELECT turbohybrid_prewarm('repl_idx'::regclass);");

$primary->stop;
$standby->promote;
$standby->safe_psql('postgres', "INSERT INTO repl_docs VALUES (-1, '[-1,0,0]'); VACUUM repl_docs;");
is(nearest($standby), '0', 'promoted standby accepts writes and queries');
is(validate($standby), 't', 'promoted standby deep validator passes');
my $cache_identity = $standby->safe_psql('postgres', q(
	SELECT (turbohybrid_prewarm('repl_idx'::regclass)->>'native_cache_used')::boolean;
));
is($cache_identity, 't', 'promoted node builds a local cache with current identity');

$standby->backup('restored');
my $restore = PostgreSQL::Test::Cluster->new('repl_restore');
$restore->init_from_backup($standby, 'restored');
$restore->start;
is(nearest($restore), '0', 'base-backup restore serves the expected result');
is(validate($restore), 't', 'base-backup restore deep validator passes');
$restore->stop;
$standby->stop;
done_testing();
