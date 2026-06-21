# On-disk metadata corruption hardening (prompt 7).
#
# For three index shapes -- a dense turbohybrid index (graph metapage path), a
# hybrid dense+tsvector index (BM25 metadata path), and a sparse-only index
# (sparse node-map / sparse meta path) -- we build the index, confirm a normal
# query works, stop the node, scribble raw bytes over the relevant on-disk
# page(s), restart, and assert that a subsequent query/diagnostic EITHER raises a
# clean PostgreSQL error OR completes without crashing.  The essential property
# proven here: corrupt metadata yields a clean error or controlled behavior,
# never a backend crash / PANIC / assertion failure.  After every failing
# statement we confirm the postmaster is still alive via a fresh `SELECT 1`.
use strict;
use warnings FATAL => 'all';
use Test::More;

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

my $node = PostgreSQL::Test::Cluster->new('metadata_corruption');
$node->init;
# Surface any backend crash as a hard failure rather than a silent restart.
$node->append_conf('postgresql.conf', "restart_after_crash = off\n");
$node->start;

my $DB = 'pgturbohybrid_corrupt';
$node->safe_psql('postgres', "CREATE DATABASE $DB;");
$node->safe_psql($DB, q(
	CREATE EXTENSION vector;
	CREATE EXTENSION pgturbohybrid;
));

# BLCKSZ default and the offset at which the graph metapage struct begins
# (PageGetContents = MAXALIGN(SizeOfPageHeaderData) = 24 on a standard build).
my $BLCKSZ      = 8192;
my $META_OFFSET = 24;

# Locate the on-disk file backing an index relation (absolute path).
sub index_file_path
{
	my ($idxname) = @_;
	my $rel = $node->safe_psql($DB, "SELECT pg_relation_filepath('$idxname');");
	chomp $rel;
	die "could not resolve filepath for $idxname" unless length $rel;
	return $node->data_dir . '/' . $rel;
}

# Overwrite $len bytes at ($block * BLCKSZ + $within) of the index file with the
# given fill byte.  Node must be stopped before calling.
sub scribble
{
	my ($path, $block, $within, $len, $fill) = @_;
	sysopen(my $fh, $path, 2)    # O_RDWR
		or die "open $path: $!";
	binmode $fh;
	sysseek($fh, $block * $BLCKSZ + $within, 0)
		or die "seek $path: $!";
	my $buf = chr($fill) x $len;
	syswrite($fh, $buf) == $len or die "write $path: $!";
	close($fh) or die "close $path: $!";
	return;
}

# Run a statement that touches the (possibly corrupt) index and assert the
# desired safety property:
#   * the statement either succeeds or raises a clean error (no crash), AND
#   * a fresh connection can still run `SELECT 1` (postmaster alive, no PANIC).
# When $expect_error is true we additionally require the statement to fail with a
# corruption-flavoured message (used for the deterministic version-field case).
sub assert_clean
{
	my ($label, $sql, $expect_error) = @_;
	my ($rc, $stdout, $stderr) =
		$node->psql($DB, $sql, on_error_die => 0);

	# Whatever happened, the server must still be up.
	my $alive = $node->safe_psql($DB, 'SELECT 1;');
	is($alive, '1', "$label: postmaster still alive after statement");

	if ($expect_error)
	{
		isnt($rc, 0, "$label: corrupt metadata raised an error");
		like($stderr,
			qr/corrupt|invalid|unexpected|magic|version|REINDEX/i,
			"$label: error message is corruption-flavoured");
	}
	else
	{
		# Dispatch-returns-false shapes may legitimately not error (the reader
		# treats a clobbered-magic page as non-native).  Either way: no crash,
		# and if it failed it must be a clean PG error (a crash would have shown
		# up as the SELECT 1 above failing).
		ok(1, "$label: completed without crashing (rc=$rc)");
		if ($rc != 0)
		{
			like($stderr,
				qr/corrupt|invalid|unexpected|magic|version|REINDEX|not.*native|index/i,
				"$label: any error is a clean PG error");
		}
	}
	return;
}

# ---------------------------------------------------------------------------
# Shape (a): dense turbohybrid index -- graph metapage path.
# ---------------------------------------------------------------------------
$node->safe_psql($DB, q(
	CREATE TABLE dense_docs (id int PRIMARY KEY, embedding vector(3));
	INSERT INTO dense_docs
	SELECT g, ('[' || g || ',0,0]')::vector FROM generate_series(0, 49) g;
	CREATE INDEX dense_idx ON dense_docs
	USING turbohybrid (embedding vector_l2_turbohybrid_ops)
	WITH (quantization_bits = 4, exact_storage = on);
));

my $dense_query = q(
	SET enable_seqscan = off;
	SELECT id FROM dense_docs
	ORDER BY embedding <~-> turbohybrid_query(
		vector_query => '[0,0,0]'::vector, dense_k => 16, bm25_k => 0, final_k => 3)
	LIMIT 3;
);
is($node->safe_psql($DB, $dense_query), "0\n1\n2",
	'dense index returns expected result before corruption');

my $dense_path = index_file_path('dense_idx');

# (a1) Corrupt ONLY the metapage format-version field (offset 28..31), leaving
# magic + storageKind intact.  This must hit the new version check and raise a
# clean DATA_CORRUPTED error with a REINDEX hint -- a deterministic ERROR path.
$node->stop;
scribble($dense_path, 0, $META_OFFSET + 4, 4, 0xFF);    # version = 0xFFFFFFFF
$node->start;
assert_clean('dense version-field', $dense_query, 1);
assert_clean('dense version-field stats',
	"SELECT turbohybrid_index_stats('dense_idx');", 1);

# Repair by rebuilding, then prove a fresh query is correct again (no behaviour
# change for a valid index).
$node->safe_psql($DB, 'REINDEX INDEX dense_idx;');
is($node->safe_psql($DB, $dense_query), "0\n1\n2",
	'dense index correct again after REINDEX');

# (a2) Broadly clobber the metapage header region (magic included).  With magic
# gone the reader dispatches "not native"; we only require no crash.  REINDEX
# above changed the relfilenode, so re-resolve the on-disk path (while the node
# is still up) before stopping to scribble.
$dense_path = index_file_path('dense_idx');
$node->stop;
scribble($dense_path, 0, $META_OFFSET, 64, 0xFF);
$node->start;
assert_clean('dense magic-clobber', $dense_query, 0);
assert_clean('dense magic-clobber stats',
	"SELECT turbohybrid_index_stats('dense_idx');", 0);
$node->safe_psql($DB, 'REINDEX INDEX dense_idx;');

# ---------------------------------------------------------------------------
# Shape (b): hybrid dense + tsvector index -- BM25 metadata path.
# ---------------------------------------------------------------------------
$node->safe_psql($DB, q(
	CREATE TABLE hybrid_docs (
		id int PRIMARY KEY, embedding vector(3), body_tsv tsvector);
	INSERT INTO hybrid_docs
	SELECT g, ('[' || g || ',0,0]')::vector,
		to_tsvector('english', 'alpha beta term' || g)
	FROM generate_series(0, 49) g;
	CREATE INDEX hybrid_idx ON hybrid_docs
	USING turbohybrid (
		embedding vector_l2_turbohybrid_ops,
		body_tsv bm25_tsvector_turbohybrid_ops)
	WITH (quantization_bits = 4, exact_storage = on);
));

my $hybrid_query = q(
	SET enable_seqscan = off;
	SELECT count(*) FROM (
		SELECT id FROM hybrid_docs
		ORDER BY embedding <~-> turbohybrid_query(
			vector_query => '[0,0,0]'::vector,
			text_query => to_tsquery('english', 'alpha'),
			dense_k => 16, bm25_k => 16, final_k => 5)
		LIMIT 5) q;
);
is($node->safe_psql($DB, $hybrid_query), '5',
	'hybrid index returns expected result before corruption');

# The BM25 metadata lives in its own block chain anchored from the metapage.
# Walk every data block and 0xFF the page payload region of each: this corrupts
# the BM25 meta + posting pages.  The metapage (block 0) is left valid so the
# scan reaches the BM25 reader, which is heavily hardened and must raise a clean
# error or skip -- never crash.
my $hybrid_path = index_file_path('hybrid_idx');
my $hybrid_blocks = $node->safe_psql($DB,
	"SELECT pg_relation_size('hybrid_idx') / $BLCKSZ;");
chomp $hybrid_blocks;

$node->stop;
for (my $b = 1; $b < $hybrid_blocks; $b++)
{
	# Clobber a chunk of each non-meta block's contents (after the header).
	scribble($hybrid_path, $b, $META_OFFSET, 128, 0xFF);
}
$node->start;
assert_clean('hybrid bm25-clobber', $hybrid_query, 0);
assert_clean('hybrid bm25-clobber stats',
	"SELECT turbohybrid_index_stats('hybrid_idx');", 0);
$node->safe_psql($DB, 'REINDEX INDEX hybrid_idx;');
is($node->safe_psql($DB, $hybrid_query), '5',
	'hybrid index correct again after REINDEX');

# ---------------------------------------------------------------------------
# Shape (c): sparse-only index -- sparse node-map / sparse meta path.
# ---------------------------------------------------------------------------
$node->safe_psql($DB, q(
	CREATE TABLE sparse_docs (id int, s turbohybrid_sparse_vector);
	INSERT INTO sparse_docs
	SELECT g, turbohybrid_sparse_vector_build(
		ARRAY[1, (g % 5) + 2]::int4[], ARRAY[g::float4, 1.0::float4])
	FROM generate_series(1, 60) g;
	CREATE INDEX sparse_idx ON sparse_docs
	USING turbohybrid (s sparse_ip_turbohybrid_ops)
	WITH (sparse_quant_bits = 8, sparse_block_size = 8);
));

my $sparse_query = q(
	SET enable_seqscan = off;
	SELECT count(*) FROM (
		SELECT id FROM sparse_docs
		ORDER BY s <~*> turbohybrid_query(
			sparse_query => turbohybrid_sparse_vector_build(
				ARRAY[1]::int4[], ARRAY[1.0]::float4[]),
			sparse_k => 5)
		LIMIT 5) q;
);
my $sparse_before = $node->safe_psql($DB, $sparse_query);
chomp $sparse_before;
cmp_ok($sparse_before, '>', '0',
	'sparse index returns rows before corruption');

# Clobber the sparse-side data blocks (node-map chain + sparse meta + postings),
# all of which live past the metapage (block 0).  Leave block 0 intact so the
# scan reaches the sparse readers we hardened.
my $sparse_path = index_file_path('sparse_idx');
my $sparse_blocks = $node->safe_psql($DB,
	"SELECT pg_relation_size('sparse_idx') / $BLCKSZ;");
chomp $sparse_blocks;

$node->stop;
for (my $b = 1; $b < $sparse_blocks; $b++)
{
	scribble($sparse_path, $b, $META_OFFSET, 128, 0xFF);
}
$node->start;
assert_clean('sparse node-map-clobber', $sparse_query, 0);
assert_clean('sparse node-map-clobber stats',
	"SELECT turbohybrid_index_stats('sparse_idx');", 0);
$node->safe_psql($DB, 'REINDEX INDEX sparse_idx;');
cmp_ok($node->safe_psql($DB, $sparse_query), '>', '0',
	'sparse index returns rows again after REINDEX');

# (c2) Also clobber the sparse metapage chain anchor / format region by zeroing
# the whole metapage payload: exercises the dispatch-returns-false path for the
# sparse-primary node-map anchor.  Must not crash.  Re-resolve the path (the
# REINDEX above changed the relfilenode) while the node is still up.
$sparse_path = index_file_path('sparse_idx');
$node->stop;
scribble($sparse_path, 0, $META_OFFSET, 200, 0x00);
$node->start;
assert_clean('sparse meta-zero', $sparse_query, 0);
assert_clean('sparse meta-zero stats',
	"SELECT turbohybrid_index_stats('sparse_idx');", 0);

# Final sanity: the server survived every corruption case.
is($node->safe_psql($DB, 'SELECT 1;'), '1',
	'postmaster alive at end of corruption suite');

$node->stop;
done_testing();
