# Shared native-cache identity and layout rejection.
use strict;
use warnings FATAL => 'all';
use Test::More;
use JSON::PP qw(decode_json);
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

my $node = PostgreSQL::Test::Cluster->new('shared_cache_identity');
$node->init;
my $stage_dir = $node->data_dir . '/shared-cache-test-stages';
mkdir($stage_dir, 0700) or die "mkdir $stage_dir: $!";
$stage_dir = abs_path($stage_dir);
$ENV{PGTURBOHYBRID_TEST_SHARED_CACHE_STAGE_DIR} = $stage_dir;
$node->append_conf('postgresql.conf', "restart_after_crash = on\n");
$node->start;

sub setup_database
{
	my ($db) = @_;
	$node->safe_psql('postgres', "CREATE DATABASE $db;");
	$node->safe_psql($db, q(
		CREATE EXTENSION vector;
		CREATE EXTENSION pgturbohybrid;
		CREATE TABLE cache_docs (id int PRIMARY KEY, embedding vector(3));
		INSERT INTO cache_docs
		SELECT g, ('[' || g || ',0,0]')::vector
		FROM generate_series(0, 39) g;
		CREATE INDEX cache_idx ON cache_docs
		USING turbohybrid (embedding vector_l2_turbohybrid_ops)
		WITH (quantization_bits = 4, exact_storage = on);
	));
}

sub prewarm
{
	my ($db, $index) = @_;
	return decode_json($node->safe_psql($db,
		"SELECT turbohybrid_prewarm('$index'::regclass)::text;"));
}

sub nearest_id
{
	my ($db) = @_;
	return $node->safe_psql($db, q(
		SET enable_seqscan = off;
		SELECT id FROM cache_docs
		ORDER BY embedding <~-> turbohybrid_query(
			vector_query => '[-1,0,0]'::vector,
			dense_k => 16, bm25_k => 0, final_k => 1)
		LIMIT 1;
	));
}

sub scan_cache_mode
{
	my ($db, $policy) = @_;
	my $output = $node->safe_psql($db, qq(
		SET turbohybrid.native_cache_scope = '$policy';
		SET enable_seqscan = off;
		SELECT id FROM cache_docs
		ORDER BY embedding <~-> turbohybrid_query(
			vector_query => '[-1,0,0]'::vector,
			dense_k => 16, bm25_k => 0, final_k => 1)
		LIMIT 1;
		SELECT turbohybrid_last_scan_stats()->>'native_cache_scope';
	));
	my @lines = grep { length $_ } split /\n/, $output;
	return $lines[-1];
}

sub cache_path
{
	my ($db, $index) = @_;
	my ($dboid, $spcoid, $relno) = split /\|/, $node->safe_psql($db, qq(
		SELECT d.oid, COALESCE(NULLIF(c.reltablespace, 0),
			(SELECT dattablespace FROM pg_database WHERE datname = current_database())),
			pg_relation_filenode(c.oid)
		FROM pg_class c CROSS JOIN pg_database d
		WHERE c.oid = '$index'::regclass AND d.datname = current_database();
	));
	my $dir = $node->data_dir . "/pg_turbohybrid_cache/$dboid/$spcoid";
	opendir(my $dh, $dir) or die "open cache directory $dir: $!";
	my @files = grep { /^${relno}_0_\d+_2_[0-9a-f]+\.tqcache$/ } readdir($dh);
	closedir($dh);
	is(scalar @files, 1, "$db has one cache file for its physical index");
	return "$dir/$files[0]";
}

sub write_at
{
	my ($path, $offset, $bytes) = @_;
	sysopen(my $fh, $path, 2) or die "open $path: $!";
	binmode $fh;
	sysseek($fh, $offset, 0) == $offset or die "seek $path: $!";
	syswrite($fh, $bytes) == length($bytes) or die "write $path: $!";
	close($fh) or die "close $path: $!";
}

sub read_u64
{
	my ($path, $offset) = @_;
	sysopen(my $fh, $path, 0) or die "open $path: $!";
	binmode $fh;
	sysseek($fh, $offset, 0) == $offset or die "seek $path: $!";
	sysread($fh, my $bytes, 8) == 8 or die "read $path: $!";
	close($fh);
	return unpack('Q', $bytes);
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

sub kill_builder_at_stage
{
	my ($db, $index, $path, $stage) = @_;
	my $request = "$stage_dir/$stage.request";
	my $reached = "$stage_dir/$stage.reached";

	unlink($path) if -e $path;
	unlink($reached) if -e $reached;
	open(my $request_fh, '>', $request) or die "create $request: $!";
	close($request_fh) or die "close $request: $!";

	my $session = $node->background_psql($db);
	my $backend_pid = $session->query_safe('SELECT pg_backend_pid();');
	$session->query_until(qr/BUILD_STARTED/,
		"\\echo BUILD_STARTED\nSELECT turbohybrid_prewarm('$index'::regclass);\n");
	ok(wait_for_file($reached, 10), "$stage builder reaches controlled stage");
	is(kill(9, $backend_pid), 1, "$stage builder backend receives SIGKILL");
	unlink($request) or die "remove $request: $!";
	eval { $session->quit; };
	ok($node->poll_query_until('postgres', 'SELECT true;'),
		"server recovers after $stage SIGKILL");

	my $result = prewarm($db, $index);
	ok($result->{native_cache_built},
		"fresh builder publishes after $stage owner death");
	ok($result->{native_cache_used},
		"no partial cache hit after $stage owner death");
	is($result->{failure_reason}, 'none',
		"$stage recovery has no residual publication failure");
	unlink($reached) if -e $reached;
}

setup_database('cache_a');
setup_database('cache_b');

is(nearest_id('cache_a'), '0', 'first database returns its own nearest row');
is(nearest_id('cache_b'), '0', 'second database returns its own nearest row');
ok(prewarm('cache_a', 'cache_idx')->{native_cache_used},
	'first database builds a shared cache');
ok(prewarm('cache_b', 'cache_idx')->{native_cache_used},
	'second database builds a shared cache');
my $path_a = cache_path('cache_a', 'cache_idx');
my $path_b = cache_path('cache_b', 'cache_idx');
isnt($path_a, $path_b, 'database OID hierarchy separates equivalent indexes');

my $prewarm_contract = prewarm('cache_a', 'cache_idx');
ok(exists $prewarm_contract->{built}, 'prewarm reports built');
ok(exists $prewarm_contract->{attached}, 'prewarm reports attached');
ok(exists $prewarm_contract->{reused}, 'prewarm reports reused');
ok(exists $prewarm_contract->{invalidated}, 'prewarm reports invalidated');
ok(exists $prewarm_contract->{gc_removed_files}, 'prewarm reports GC removals');
ok(exists $prewarm_contract->{failure_reason}, 'prewarm reports failure reason');
ok(exists $prewarm_contract->{effective_bytes}, 'prewarm reports effective bytes');

# Lock files are persistent coordination inodes, not ownership markers. A
# leftover inode from a killed builder must not prevent a new builder.
my $stale_lock = $path_a;
my $cache_index_oid = $node->safe_psql('cache_a',
	"SELECT 'cache_idx'::regclass::oid;");
$stale_lock =~ s{/[^/]+$}{/$cache_index_oid\_0.lock};
unlink($path_a) or die "unlink cache for stale-lock test: $!";
open(my $lock_fh, '>', $stale_lock) or die "create stale lock: $!";
close($lock_fh) or die "close stale lock: $!";
my $after_stale_lock = prewarm('cache_a', 'cache_idx');
ok($after_stale_lock->{native_cache_built},
	'persistent unlocked lock inode cannot strand cache construction');
ok($after_stale_lock->{native_cache_used},
	'cache remains usable after stale lock recovery');

for my $stage (qw(after_lock after_ftruncate after_mmap before_fsync before_rename))
{
	$path_a = cache_path('cache_a', 'cache_idx');
	kill_builder_at_stage('cache_a', 'cache_idx', $path_a, $stage);
}

# A template copy deliberately reuses physical relation numbers. Its database
# OID must still produce a distinct cache and later mutations must be isolated.
$node->safe_psql('postgres', 'CREATE DATABASE cache_copy TEMPLATE cache_a;');
$node->safe_psql('cache_copy',
	"INSERT INTO cache_docs VALUES (100, '[-1,0,0]'::vector);");
is(nearest_id('cache_copy'), '100', 'template copy sees its independent insert');
is(nearest_id('cache_a'), '0', 'template source remains unchanged');
ok(prewarm('cache_copy', 'cache_idx')->{native_cache_used},
	'template copy builds its own cache');
my $path_copy = cache_path('cache_copy', 'cache_idx');
isnt($path_copy, $path_a, 'template copy cannot attach the source database cache');

# Header offsets are the stable cache-format-v2 ABI asserted by this test:
# prefix 24, identity db/tablespace/generation at 24/28/48, adj count at 132,
# and the first two {offset,length} segments at 152 and 168.
my @corruptions = (
	['database OID', sub { write_at($_[0], 24, pack('L', 0x7ffffffe)); }],
	['tablespace OID', sub { write_at($_[0], 28, pack('L', 0x7ffffffd)); }],
	['generation', sub { write_at($_[0], 48, pack('Q', 0)); }],
	['declared file size', sub {
		write_at($_[0], 16, pack('Q', (-s $_[0]) + 8));
	}],
	['offset beyond EOF', sub {
		write_at($_[0], 152, pack('Q', (-s $_[0]) + 8));
	}],
	['overlapping segments', sub {
		my $nodes_offset = read_u64($_[0], 152);
		write_at($_[0], 168, pack('Q', $nodes_offset));
	}],
	['adjacency record count', sub { write_at($_[0], 132, pack('L', 0)); }],
	['truncated file', sub { truncate($_[0], 200) or die "truncate $_[0]: $!"; }],
);

for my $case (@corruptions)
{
	my ($label, $mutate) = @$case;
	$path_a = cache_path('cache_a', 'cache_idx');
	$mutate->($path_a);
	my $result = prewarm('cache_a', 'cache_idx');
	ok($result->{native_cache_invalidated}, "$label cache is invalidated");
	ok($result->{native_cache_built}, "$label cache is rebuilt, not hit");
	ok($result->{native_cache_used}, "$label does not disable usable queries");
	is(nearest_id('cache_a'), '0', "$label cannot change scoring");
	is($node->safe_psql('cache_a', 'SELECT 1;'), '1',
		"$label leaves a fresh session usable");
}

# Keep one backend alive while other backends advance and publish generations.
# Its next scan must observe the new metapage and detach the superseded mmap.
my $reader = $node->background_psql('cache_a');
my $reader_initial = decode_json($reader->query_safe(
	"SELECT turbohybrid_prewarm('cache_idx'::regclass)::text;"));
is($reader_initial->{backend_mappings}, 1,
	'long-lived reader starts with one shared mapping');
$node->safe_psql('cache_a',
	"INSERT INTO cache_docs VALUES (150, '[-1,0,0]'::vector);");
ok(prewarm('cache_a', 'cache_idx')->{native_cache_used},
	'other backend publishes the insert generation');
is($reader->query_safe(q(
	SET enable_seqscan = off;
	SELECT id FROM cache_docs
	ORDER BY embedding <~-> turbohybrid_query(
		vector_query => '[-1,0,0]'::vector,
		dense_k => 16, bm25_k => 0, final_k => 1)
	LIMIT 1;
)), '150', 'long-lived reader sees the inserted generation on its next scan');
my $reader_after_insert = decode_json($reader->query_safe(
	"SELECT turbohybrid_prewarm('cache_idx'::regclass)::text;"));
is($reader_after_insert->{backend_mappings}, 1,
	'long-lived reader detaches its pre-insert mapping');
$node->safe_psql('cache_a', 'REINDEX INDEX cache_idx;');
ok(prewarm('cache_a', 'cache_idx')->{native_cache_used},
	'other backend publishes the REINDEX generation');
my $reader_after_reindex = decode_json($reader->query_safe(
	"SELECT turbohybrid_prewarm('cache_idx'::regclass)::text;"));
is($reader_after_reindex->{backend_mappings}, 1,
	'long-lived reader detaches its pre-REINDEX mapping');
$reader->quit;

# The authoritative maximum layout feeds both admission and estimates. A small
# graph fits a 1 MB cap; a deterministic larger graph exceeds that same cap.
my $small = decode_json($node->safe_psql('cache_a', q(
	SET turbohybrid.native_cache_max_mb = 1;
	SELECT turbohybrid_prewarm('cache_idx'::regclass)::text;
)));
ok($small->{native_cache_used}, 'layout just below the configured cap is admitted');

$node->safe_psql('cache_a', q(
	CREATE TABLE large_docs (id int PRIMARY KEY, embedding vector(64));
	INSERT INTO large_docs
	SELECT g, array_fill((g % 11)::real, ARRAY[64])::vector
	FROM generate_series(1, 4000) g;
	CREATE INDEX large_idx ON large_docs
	USING turbohybrid (embedding vector_l2_turbohybrid_ops)
	WITH (quantization_bits = 4, exact_storage = off);
));
my $large = decode_json($node->safe_psql('cache_a', q(
	SET turbohybrid.native_cache_max_mb = 1;
	SELECT turbohybrid_prewarm('large_idx'::regclass)::text;
)));
ok(!$large->{native_cache_used}, 'layout over the configured cap is rejected');
is($large->{native_cache_reason}, 'exceeds_max_mb',
	'over-cap rejection is visible');
is($node->safe_psql('cache_a', 'SELECT 1;'), '1',
	'over-cap prewarm leaves the session usable');

# Make shared publication fail before open/mmap without affecting index data.
# Auto must use its documented per-backend fallback, while explicit shared
# must remain uncached and must never hide a process-local full copy.
my $cache_root = $node->data_dir . '/pg_turbohybrid_cache';
my $cache_backup = $node->data_dir . '/pg_turbohybrid_cache.policy-test';
rename($cache_root, $cache_backup) or die "rename cache root: $!";
open(my $root_fh, '>', $cache_root) or die "replace cache root with file: $!";
close($root_fh) or die "close cache root replacement: $!";
is(scan_cache_mode('cache_a', 'auto'), 'per_backend',
	'auto falls back to per_backend after forced shared failure');
is(scan_cache_mode('cache_a', 'shared'), 'per_scan',
	'explicit shared failure remains uncached without fallback');
unlink($cache_root) or die "remove cache root replacement: $!";
rename($cache_backup, $cache_root) or die "restore cache root: $!";

# Each generation publication removes older generations for the same physical
# index, bounding files across ordinary insert/query/prewarm cycles.
for my $id (200 .. 204)
{
	$node->safe_psql('cache_a',
		"INSERT INTO cache_docs VALUES ($id, '[$id,0,0]'::vector);");
	my $cycle = prewarm('cache_a', 'cache_idx');
	ok($cycle->{native_cache_used}, "insert cycle $id publishes a usable cache");
	cache_path('cache_a', 'cache_idx');
}
$node->safe_psql('cache_a', 'VACUUM cache_docs;');
ok(prewarm('cache_a', 'cache_idx')->{native_cache_used},
	'VACUUM cycle leaves a usable bounded cache');
$node->safe_psql('cache_a', 'REINDEX INDEX cache_idx;');
ok(prewarm('cache_a', 'cache_idx')->{native_cache_used},
	'REINDEX cycle leaves a usable bounded cache');
$path_a = cache_path('cache_a', 'cache_idx');
my ($cache_dir_a) = $path_a =~ m{^(.*)/[^/]+$};
opendir(my $cache_dir_fh, $cache_dir_a) or die "open $cache_dir_a: $!";
my @temporary_files = grep { /\.tqcache\.tmp\./ } readdir($cache_dir_fh);
rewinddir($cache_dir_fh);
my @lock_files = grep { /\.lock$/ } readdir($cache_dir_fh);
closedir($cache_dir_fh);
is(scalar @temporary_files, 0, 'completed cycles leave no temporary files');
is(scalar @lock_files, 1, 'generation-independent builder lock stays bounded');

# A cache belonging to a dropped index is recognized and removed lazily by a
# fresh backend's first access; an unrelated live database provides that access.
$path_a = cache_path('cache_a', 'cache_idx');
$node->safe_psql('cache_a', q(
	CREATE INDEX cache_gc_idx ON cache_docs
	USING turbohybrid (embedding vector_l2_turbohybrid_ops)
	WITH (quantization_bits = 4, exact_storage = on);
));
$node->safe_psql('cache_a', 'DROP INDEX cache_idx;');
ok(-e $path_a, 'dropped index cache remains until lazy GC runs');
prewarm('cache_a', 'cache_gc_idx');
ok(!-e $path_a, 'lazy GC removes the dropped index cache');

$node->stop;
done_testing();
