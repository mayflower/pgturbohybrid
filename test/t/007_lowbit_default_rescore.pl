# Bit-width-keyed default rescore (adopted from Qdrant's
# tq_bits_default_rescoring): aggressive low-bit (1/2-bit) indexes heap-rescore
# by default under the auto, non-latency profiles at ANY dimension -- because
# low-bit codes are too lossy to rank on directly -- while the default 4-bit
# index stays code-only at high dimension. The rescore decision is data
# independent (profile x bit-width x dimension), so the reported reason is the
# stable thing to assert.
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

my $node = PostgreSQL::Test::Cluster->new('lowbit_default_rescore');
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector; CREATE EXTENSION pgturbohybrid;');

# 384 dims is above the auto-heap-rescore low-dimension threshold (256), so the
# rescore decision is driven by bit-width, not dimension.
$node->safe_psql('postgres', q(
	CREATE TABLE lbr (id int PRIMARY KEY, embedding vector(384));
	INSERT INTO lbr(id, embedding)
	SELECT g,
	       (SELECT array_agg(((g * 7 + d * 13) % 17 - 8)::float4 / 8.0)
	          FROM generate_series(1, 384) d)::vector
	FROM generate_series(1, 60) g;
	CREATE INDEX lbr_2bit ON lbr
		USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
		WITH (quantization_bits = 2);
));

# Scan once under $profile, then read the heap-rescore reason from the same
# backend (turbohybrid_last_scan_stats is backend-local, so it must share the
# connection with the scan).
sub rescore_reason
{
	my ($profile) = @_;
	return $node->safe_psql('postgres', qq(
		SET enable_seqscan = off;
		SET turbohybrid.profile = $profile;
		DO \$\$ BEGIN
			PERFORM id FROM lbr
			ORDER BY embedding <~> turbohybrid_query(
				vector_query => (SELECT embedding FROM lbr WHERE id = 1))
			LIMIT 10;
		END \$\$;
		SELECT turbohybrid_last_scan_stats()->>'heap_rescore_reason';
	));
}

# 2-bit: latency keeps its no-rescore contract; balanced/quality auto-rescore.
is(rescore_reason('latency'), 'profile_latency',
	'2-bit latency profile stays code-only (no auto rescore)');
is(rescore_reason('balanced'), 'profile_lowbit',
	'2-bit balanced auto-rescores by bit-width at high dimension');
is(rescore_reason('quality'), 'profile_lowbit',
	'2-bit quality auto-rescores by bit-width at high dimension');

# 4-bit control on the same column: the default bit width must NOT auto-rescore
# at high dimension under balanced (mirrors Qdrant leaving 4-bit off).
$node->safe_psql('postgres', q(
	CREATE INDEX lbr_4bit ON lbr
		USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
		WITH (quantization_bits = 4);
	DROP INDEX lbr_2bit;
));
is(rescore_reason('balanced'), 'profile_balanced_highdim',
	'4-bit high-dimensional balanced stays code-only (matches Qdrant)');

$node->stop;
done_testing();
