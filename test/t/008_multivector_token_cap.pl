# Per-document token cap on the exact_symmetric multivector document graph build
# (adopted from Qdrant's StrictModeMultivectorConfig.max_vectors). Exact
# symmetric MaxSim build cost is O(tokens_a * tokens_b * dim) per document pair,
# so a single token-heavy document is expensive regardless of document count.
# turbohybrid.multivector_exact_symmetric_build_max_tokens bounds the per-pair
# cost; 0 (the default) leaves historical behavior unchanged.
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

my $node = PostgreSQL::Test::Cluster->new('multivector_token_cap');
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector; CREATE EXTENSION pgturbohybrid; CREATE EXTENSION pgturbohybrid_experimental;');

# Two documents: one with 2 tokens, one with 4 tokens. Document count (2) stays
# well under the default doc-count guard (1000), so only the token cap can
# block the exact_symmetric build.
$node->safe_psql('postgres', q(
	CREATE TABLE tc (id int PRIMARY KEY, embedding turbohybrid_multivector);
	INSERT INTO tc VALUES
		(1, turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector])),
		(2, turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector,
		                                  '[1,1]'::vector, '[0,0]'::vector]));
));

my $mk_index = q(
	CREATE INDEX tc_idx ON tc
		USING turbohybrid (embedding multivector_maxsim_ip_turbohybrid_ops)
		WITH (multivector_graph = 'document_nodes',
		      multivector_doc_build_scorer = 'exact_symmetric')
);

# A cap below the largest document's token count rejects the build.
my ($rc, $out, $err) = $node->psql('postgres',
	"SET turbohybrid.multivector_exact_symmetric_build_max_tokens = 3;\n$mk_index;");
isnt($rc, 0, 'exact_symmetric build rejected when a document exceeds the token cap');
like($err, qr/token count/, 'rejection error is the token-cap guard');
is($node->safe_psql('postgres', 'SELECT 1'), '1', 'backend still alive after rejection');

# With the cap unset (default 0 = unlimited), the same build is allowed at this
# scale -- proving the cap is the only thing that blocked it above.
my ($rc2, $out2, $err2) = $node->psql('postgres', "$mk_index;");
is($rc2, 0, 'exact_symmetric build allowed with the default unlimited token cap')
	or diag($err2);
is($node->safe_psql('postgres',
	"SELECT count(*) FROM pg_class WHERE relname = 'tc_idx';"), '1',
	'index was created when the token cap is unlimited');

$node->stop;
done_testing();
