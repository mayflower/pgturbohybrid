use strict;
use warnings FATAL => 'all';
use Test::More;
use PostgreSQL::Test::Cluster;

my $node = PostgreSQL::Test::Cluster->new('wal_restart');
$node->init;
$node->start;

sub top_ids
{
	return $node->safe_psql('pgturbohybrid_wal', q(
		SET enable_seqscan = off;
		SELECT string_agg(id::text, ',' ORDER BY ord)
		FROM (
			SELECT id, row_number() OVER () AS ord
			FROM (
				SELECT id
				FROM wal_docs
				ORDER BY embedding <~-> turbohybrid_query(
					vector_query => '[0,0,0]'::vector,
					dense_k => 16,
					bm25_k => 0,
					final_k => 3
				)
				LIMIT 3
			) q
		) s;
	));
}

$node->safe_psql('postgres', 'CREATE DATABASE pgturbohybrid_wal;');
$node->safe_psql('pgturbohybrid_wal', q(
	CREATE EXTENSION vector;
	CREATE EXTENSION pgturbohybrid;
));

unlike($node->safe_psql('pgturbohybrid_wal', 'SHOW shared_preload_libraries;'),
	qr/(^|,)pgturbohybrid(,|$)/,
	'pgturbohybrid works without shared_preload_libraries');

$node->safe_psql('pgturbohybrid_wal', q(
	CREATE TABLE wal_docs (
		id int PRIMARY KEY,
		embedding vector(3),
		body_tsv tsvector
	);
	INSERT INTO wal_docs VALUES
		(1, '[0,0,0]', to_tsvector('english', 'zero')),
		(2, '[1,0,0]', to_tsvector('english', 'one')),
		(3, '[2,0,0]', to_tsvector('english', 'two')),
		(4, '[3,0,0]', to_tsvector('english', 'three'));
	CREATE INDEX wal_docs_idx ON wal_docs
	USING turbohybrid (
		embedding vector_l2_turbohybrid_ops,
		body_tsv bm25_tsvector_turbohybrid_ops
	)
	WITH (quantization_bits = 4, exact_storage = on);
));

$node->restart;
is(top_ids(), '1,2,3', 'index remains correct after build and restart');

$node->safe_psql('pgturbohybrid_wal', q(
	CREATE INDEX CONCURRENTLY wal_docs_concurrent_idx ON wal_docs
	USING turbohybrid (
		embedding vector_l2_turbohybrid_ops,
		body_tsv bm25_tsvector_turbohybrid_ops
	)
	WITH (quantization_bits = 4, exact_storage = off);
));
$node->restart;
is(top_ids(), '1,2,3', 'index remains correct after concurrent build and restart');
$node->safe_psql('pgturbohybrid_wal', 'DROP INDEX wal_docs_concurrent_idx;');

$node->safe_psql('pgturbohybrid_wal',
	"INSERT INTO wal_docs VALUES (5, '[0.5,0,0]', to_tsvector('english', 'half'));");
$node->restart;
is(top_ids(), '1,5,2', 'index remains correct after insert and restart');

$node->safe_psql('pgturbohybrid_wal', 'DELETE FROM wal_docs WHERE id = 1;');
$node->restart;
is(top_ids(), '5,2,3', 'index remains correct after delete and restart');

$node->safe_psql('pgturbohybrid_wal', 'VACUUM wal_docs;');
$node->restart;
is(top_ids(), '5,2,3', 'index remains correct after vacuum and restart');

$node->safe_psql('pgturbohybrid_wal', 'REINDEX INDEX wal_docs_idx;');
$node->restart;
is(top_ids(), '5,2,3', 'index remains correct after reindex and restart');

$node->safe_psql('pgturbohybrid_wal', q(
	CREATE UNLOGGED TABLE wal_unlogged_docs (
		id int PRIMARY KEY,
		embedding vector(3),
		body_tsv tsvector
	);
	INSERT INTO wal_unlogged_docs VALUES
		(1, '[0,0,0]', to_tsvector('english', 'zero')),
		(2, '[1,0,0]', to_tsvector('english', 'one'));
	CREATE INDEX wal_unlogged_docs_idx ON wal_unlogged_docs
	USING turbohybrid (
		embedding vector_l2_turbohybrid_ops,
		body_tsv bm25_tsvector_turbohybrid_ops
	);
));
is($node->safe_psql('pgturbohybrid_wal', q(
	SET enable_seqscan = off;
	SELECT id
	FROM wal_unlogged_docs
	ORDER BY embedding <~-> turbohybrid_query(
		vector_query => '[0,0,0]'::vector,
		dense_k => 4,
		bm25_k => 0,
		final_k => 1
	)
	LIMIT 1;
)), '1', 'unlogged table index returns expected result before restart');
$node->restart;
is($node->safe_psql('pgturbohybrid_wal', q(
	SET enable_seqscan = off;
	SELECT id
	FROM wal_unlogged_docs
	ORDER BY embedding <~-> turbohybrid_query(
		vector_query => '[0,0,0]'::vector,
		dense_k => 4,
		bm25_k => 0,
		final_k => 1
	)
	LIMIT 1;
)), '1', 'unlogged table index returns expected result after clean restart');

$node->safe_psql('pgturbohybrid_wal',
	"INSERT INTO wal_docs VALUES (6, '[0.25,0,0]', to_tsvector('english', 'quarter'));");
$node->stop('immediate');
$node->start;
is(top_ids(), '6,5,2', 'index remains correct after immediate stop and recovery');

done_testing();
