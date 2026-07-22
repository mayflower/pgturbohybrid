# Repeated insert/delete/VACUUM churn must preserve graph validity and results.
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

my $node = PostgreSQL::Test::Cluster->new('validate_churn');
$node->init;
$node->start;
$node->safe_psql('postgres', q(
	CREATE EXTENSION vector;
	CREATE EXTENSION pgturbohybrid;
	CREATE TABLE churn_docs (id int PRIMARY KEY, embedding vector(3));
	INSERT INTO churn_docs
	SELECT g, ('[' || g || ',' || (g % 11) || ',0]')::vector
	FROM generate_series(1, 100) g;
	CREATE INDEX churn_idx ON churn_docs
	USING turbohybrid (embedding vector_l2_turbohybrid_ops)
	WITH (graph_m = 8, exact_storage = on);
));

for my $cycle (1 .. 10)
{
	my $base = 100 + $cycle * 10;
	my $delete_start = $cycle * 3;
	my $delete_end = $delete_start + 1;
	$node->safe_psql('postgres', qq(
		DELETE FROM churn_docs WHERE id BETWEEN $delete_start AND $delete_end;
		INSERT INTO churn_docs
		SELECT g, ('[' || g || ',' || (g % 11) || ',0]')::vector
		FROM generate_series($base + 1, $base + 5) g;
		VACUUM churn_docs;
	));
	is($node->safe_psql('postgres', q(
		SELECT (turbohybrid_validate_index('churn_idx'::regclass, true)->>'ok')::boolean;
	)), 't', "deep validator passes after churn cycle $cycle");

	my $indexed = $node->safe_psql('postgres', q(
		SET enable_seqscan = off;
		SELECT id FROM churn_docs
		ORDER BY embedding <~-> turbohybrid_query(
			vector_query => '[50,6,0]'::vector, dense_k => 512, final_k => 1)
		LIMIT 1;
	));
	my $exact = $node->safe_psql('postgres', q(
		SET enable_indexscan = off;
		SET enable_bitmapscan = off;
		SELECT id FROM churn_docs ORDER BY embedding <-> '[50,6,0]'::vector, id LIMIT 1;
	));
	is($indexed, $exact, "nearest result matches exact scan after cycle $cycle");
}

$node->stop;
done_testing();
