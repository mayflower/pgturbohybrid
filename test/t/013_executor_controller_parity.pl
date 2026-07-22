# LIMIT controller is an optimization only: results and order must be identical.
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

my $node = PostgreSQL::Test::Cluster->new('executor_controller_parity');
$node->init;
$node->start;
$node->safe_psql('postgres', q(
	CREATE EXTENSION vector;
	CREATE EXTENSION pgturbohybrid;
	CREATE TABLE controller_docs(id int PRIMARY KEY, tenant int, embedding vector(3));
	INSERT INTO controller_docs
	SELECT g, g % 4, ('[' || (g % 17) || ',' || (g % 7) || ',' || (g % 3) || ']')::vector
	FROM generate_series(1,256) g;
	CREATE INDEX controller_idx ON controller_docs
	USING turbohybrid (embedding vector_l2_turbohybrid_ops)
	WITH (quantization_bits=4, exact_storage=on);
));

sub run_mode
{
	my ($disabled, $sql) = @_;
	return $node->safe_psql('postgres',
		"SET enable_seqscan=off; SET turbohybrid.dev.disable_executor_controller=$disabled; $sql");
}

my $order = q(
	SELECT string_agg(id::text, ',' ORDER BY ordinal)
	FROM (SELECT id, row_number() OVER () ordinal FROM (
		SELECT id FROM controller_docs
		ORDER BY embedding <~-> turbohybrid_query(
			vector_query=>'[1,2,0]'::vector,dense_k=>128,bm25_k=>0,final_k=>40)
		LIMIT 20 OFFSET 3) ranked) ordered
);

my @cases = (
	['limit-offset', $order],
	['filter', "SELECT string_agg(id::text,',' ORDER BY id) FROM (SELECT id FROM controller_docs WHERE tenant=2 ORDER BY embedding <~-> turbohybrid_query(vector_query=>'[1,2,0]'::vector,dense_k=>200,bm25_k=>0,final_k=>50) LIMIT 12) s"],
	['result-subquery', "SELECT string_agg(id::text,',' ORDER BY id) FROM (SELECT id FROM controller_docs ORDER BY embedding <~-> turbohybrid_query(vector_query=>'[1,2,0]'::vector,dense_k=>100,bm25_k=>0,final_k=>15) LIMIT 15) s WHERE id>0"],
	['cte', "WITH s AS MATERIALIZED (SELECT id FROM controller_docs ORDER BY embedding <~-> turbohybrid_query(vector_query=>'[1,2,0]'::vector,dense_k=>100,bm25_k=>0,final_k=>15) LIMIT 15) SELECT string_agg(id::text,',' ORDER BY id) FROM s"],
	['lateral-nested-loop', "SELECT string_agg(hit.id::text,',' ORDER BY q.n,hit.id) FROM generate_series(1,2) q(n) CROSS JOIN LATERAL (SELECT id FROM controller_docs ORDER BY embedding <~-> turbohybrid_query(vector_query=>format('[%s,2,0]',q.n)::vector,dense_k=>100,bm25_k=>0,final_k=>5) LIMIT 5) hit"],
	['two-scans', "SELECT string_agg(id::text,',' ORDER BY id) FROM ((SELECT id FROM controller_docs ORDER BY embedding <~-> turbohybrid_query(vector_query=>'[1,2,0]'::vector,dense_k=>50,bm25_k=>0,final_k=>5) LIMIT 5) UNION ALL (SELECT id FROM controller_docs ORDER BY embedding <~-> turbohybrid_query(vector_query=>'[2,1,0]'::vector,dense_k=>50,bm25_k=>0,final_k=>5) LIMIT 5)) s"],
);

for my $case (@cases)
{
	my ($name, $sql) = @$case;
	is(run_mode('on', $sql), run_mode('off', $sql), "$name IDs and order match");
}

for my $plan_mode (qw(force_custom_plan force_generic_plan))
{
	my $prepared = "SET plan_cache_mode=$plan_mode; PREPARE controller_plan(vector) AS SELECT id FROM controller_docs ORDER BY embedding <~-> turbohybrid_query(vector_query=>\$1,dense_k=>100,bm25_k=>0,final_k=>10) LIMIT 10; EXECUTE controller_plan('[1,2,0]'::vector)";
	my $enabled = run_mode('off', $prepared);
	my $disabled = run_mode('on', $prepared);
	is($disabled, $enabled, "$plan_mode prepared output matches");
}

$node->stop;
done_testing();
