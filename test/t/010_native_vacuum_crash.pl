# Deterministic crash recovery while native graph VACUUM is in progress.
use strict;
use warnings FATAL => 'all';
use Test::More;
use Time::HiRes qw(time usleep);
use Cwd qw(abs_path);

BEGIN
{
	eval {
		require PostgreSQL::Test::Cluster;
		PostgreSQL::Test::Cluster->import();
		require PostgreSQL::Test::Utils;
		PostgreSQL::Test::Utils->import();
		1;
	} or plan skip_all => 'PostgreSQL::Test::Cluster is not available';
}

sub wait_for_file
{
	my ($path, $seconds) = @_;
	my $deadline = time() + $seconds;

	while (time() < $deadline)
	{
		return 1 if -e $path;
		usleep(10_000);
	}
	return 0;
}

my $node = PostgreSQL::Test::Cluster->new('native_vacuum_crash');
$node->init;
my $stage_dir = $node->data_dir . '/vacuum-test-stages';
mkdir($stage_dir, 0700) or die "mkdir $stage_dir: $!";
$stage_dir = abs_path($stage_dir);
$ENV{PGTURBOHYBRID_TEST_VACUUM_STAGE_DIR} = $stage_dir;
$node->append_conf('postgresql.conf', "restart_after_crash = on\n");
$node->start;
$node->safe_psql('postgres', q(
	CREATE EXTENSION vector;
	CREATE EXTENSION pgturbohybrid;
	CREATE TABLE vacuum_docs (id int PRIMARY KEY, embedding vector(3));
	INSERT INTO vacuum_docs
	SELECT g, ('[' || g || ',' || (g % 7) || ',0]')::vector
	FROM generate_series(0, 127) g;
	CREATE INDEX vacuum_idx ON vacuum_docs
	USING turbohybrid (embedding vector_l2_turbohybrid_ops)
	WITH (quantization_bits = 4, exact_storage = on);
	DELETE FROM vacuum_docs WHERE id % 4 = 0;
));

my $request = "$stage_dir/after_delete_page.request";
my $reached = "$stage_dir/after_delete_page.reached";
open(my $request_fh, '>', $request) or die "create $request: $!";
close($request_fh) or die "close $request: $!";

my $session = $node->background_psql('postgres');
my $backend_pid = $session->query_safe('SELECT pg_backend_pid();');
$session->query_until(qr/VACUUM_STARTED/,
	"\\echo VACUUM_STARTED\nVACUUM vacuum_docs;\n");
ok(wait_for_file($reached, 10), 'VACUUM reaches controlled post-WAL stage');
is(kill(9, $backend_pid), 1, 'VACUUM backend receives SIGKILL');
unlink($request) or die "remove $request: $!";
eval { $session->quit; };
ok($node->poll_query_until('postgres', 'SELECT true;'),
	'server recovers after interrupted VACUUM');

$node->safe_psql('postgres', 'VACUUM vacuum_docs;');
is($node->safe_psql('postgres', q(
	SET enable_seqscan = off;
	SELECT id FROM vacuum_docs
	ORDER BY embedding <~-> turbohybrid_query(
		vector_query => '[1,1,0]'::vector,
		dense_k => 128, bm25_k => 0, final_k => 1)
	LIMIT 1;
)), '1', 'recovered graph returns the nearest live tuple');
is($node->safe_psql('postgres', q(
	SELECT (turbohybrid_index_stats('vacuum_idx'::regclass)->>'live_nodes')::int;
)), '96', 'rerun VACUUM reports the authoritative live-node count');
is($node->safe_psql('postgres', q(
	SELECT (turbohybrid_validate_index('vacuum_idx'::regclass, true)->>'ok')::boolean;
)), 't', 'deep validator passes after interrupted VACUUM recovery');

$node->restart;
is($node->safe_psql('postgres', q(
	SET enable_seqscan = off;
	SELECT id FROM vacuum_docs
	ORDER BY embedding <~-> turbohybrid_query(
		vector_query => '[1,1,0]'::vector,
		dense_k => 128, bm25_k => 0, final_k => 1)
	LIMIT 1;
)), '1', 'repaired graph remains correct after restart');
is($node->safe_psql('postgres', q(
	SELECT (turbohybrid_validate_index('vacuum_idx'::regclass, true)->>'ok')::boolean;
)), 't', 'deep validator remains green after restart');

$node->stop;
done_testing();
