use strict;
use warnings FATAL => 'all';
use Test::More;
use File::Basename qw(basename dirname);
use File::Copy qw(copy);
use Cwd qw(abs_path);
use JSON::PP qw(decode_json);
use PostgreSQL::Test::Cluster;

my $model_path = $ENV{PG_COLBERT_LLAMA_TEST_MODEL}
  or plan skip_all => 'PG_COLBERT_LLAMA_TEST_MODEL is not set';
my $resolved_model_path = abs_path($model_path);
if (!defined $resolved_model_path && $model_path !~ m{\A/} && defined $ENV{TH_ROOT})
{
	$resolved_model_path = abs_path("$ENV{TH_ROOT}/$model_path");
}
$model_path = $resolved_model_path
  or die "could not resolve PG_COLBERT_LLAMA_TEST_MODEL: $ENV{PG_COLBERT_LLAMA_TEST_MODEL}";

my $model_dir = dirname($model_path);
my $alias = basename($model_path);
$alias =~ s/\.gguf\z//;
my $expected_dim = $ENV{PG_COLBERT_LLAMA_EXPECTED_DIM} // 128;
my $profile_path = $ENV{PG_COLBERT_LLAMA_TEST_PROFILE};
my $token_plan_golden = $ENV{PG_COLBERT_LLAMA_TOKEN_PLAN_GOLDEN};

if (defined $profile_path && $profile_path ne '')
{
	my $sidecar_path = "$model_dir/$alias.gguf.colbert_profile.json";
	if ($profile_path ne $sidecar_path)
	{
		copy($profile_path, $sidecar_path)
		  or die "copy profile sidecar failed: $!";
	}
}

sub sql_literal
{
	my ($value) = @_;
	$value =~ s/'/''/g;
	return "'$value'";
}

my $node = PostgreSQL::Test::Cluster->new('pg_colbert_llama_live');
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE DATABASE pg_colbert_llama_live;');
$node->safe_psql('pg_colbert_llama_live', q(
	CREATE EXTENSION vector;
	CREATE EXTENSION pgturbohybrid;
	CREATE EXTENSION pg_colbert_llama;
));
$node->safe_psql('postgres', sprintf(q(
	ALTER DATABASE pg_colbert_llama_live SET pg_colbert_llama.model_dir = %s;
	ALTER DATABASE pg_colbert_llama_live SET pg_colbert_llama.expected_dim = %d;
), sql_literal($model_dir), $expected_dim));

is($node->safe_psql('pg_colbert_llama_live', sprintf(q(
	SELECT turbohybrid_multivector_dims(colbert_mv(%s, 'test')) = %d;
), sql_literal("$alias:query"), $expected_dim)), 't',
	'live query encoding returns expected ColBERT dimension');

is($node->safe_psql('pg_colbert_llama_live', sprintf(q(
	WITH info AS (
		SELECT colbert_model_info(%s) AS payload
	)
	SELECT payload->>'engine' = 'llama'
	       AND (payload->>'n_embd_out')::int = %d
	       AND payload->>'projection_status' = 'ok'
	       AND (payload->>'require_normalized')::boolean
	FROM info;
), sql_literal("$alias:query"), $expected_dim)), 't',
	'live model info reports llama projection output and normalization');

if (defined $profile_path && $profile_path ne '')
{
	is($node->safe_psql('pg_colbert_llama_live', sprintf(q(
		WITH info AS (
			SELECT colbert_model_info(%s) AS payload
		)
		SELECT payload->>'profile_loaded' = 'true'
		       AND payload->>'profile_source' IN ('sidecar', 'gguf')
		       AND payload->>'profile_schema' = 'pg_colbert_profile_v1'
		FROM info;
	), sql_literal("$alias:query"))), 't',
		'live model info reports profile loaded from sidecar or GGUF');
}

is($node->safe_psql('pg_colbert_llama_live', sprintf(q(
	SELECT turbohybrid_multivector_count(colbert_mv(%s, 'test')) > 0;
), sql_literal("$alias:query"))), 't',
	'live query encoding returns at least one token vector');

my $batch_parity = $node->safe_psql('pg_colbert_llama_live', sprintf(q(
	WITH scalar AS (
		SELECT colbert_mv(%s, 'Mars is the red planet.') AS doc1,
		       colbert_mv(%s, 'Earth has blue oceans.') AS doc2
	),
	batched AS (
		SELECT colbert_mv_batch(
			%s,
			ARRAY['Mars is the red planet.', 'Earth has blue oceans.']
		) AS docs
	)
	SELECT turbohybrid_multivector_dims(doc1) = %d
	       AND turbohybrid_multivector_count(doc1) =
	           turbohybrid_multivector_count(docs[1])
	       AND turbohybrid_multivector_count(doc2) =
	           turbohybrid_multivector_count(docs[2])
	       AND abs(turbohybrid_multivector_maxsim(doc1, docs[1]) -
	               turbohybrid_multivector_count(doc1)) < 1e-3
	       AND abs(turbohybrid_multivector_maxsim(doc2, docs[2]) -
	               turbohybrid_multivector_count(doc2)) < 1e-3
	       AS ok,
	       turbohybrid_multivector_count(doc1) AS doc1_count,
	       turbohybrid_multivector_count(docs[1]) AS batch1_count,
	       turbohybrid_multivector_maxsim(doc1, docs[1]) AS doc1_batch1_maxsim,
	       turbohybrid_multivector_count(doc2) AS doc2_count,
	       turbohybrid_multivector_count(docs[2]) AS batch2_count,
	       turbohybrid_multivector_maxsim(doc2, docs[2]) AS doc2_batch2_maxsim
	FROM scalar, batched;
), sql_literal("$alias:doc"), sql_literal("$alias:doc"),
	sql_literal("$alias:doc"), $expected_dim));
my ($batch_ok) = split /\|/, $batch_parity;
is($batch_ok, 't',
	'live batched document encoding matches scalar multivectors')
	or diag("batch parity details: $batch_parity");

is($node->safe_psql('pg_colbert_llama_live', sprintf(q(
	WITH debug AS (
		SELECT colbert_debug(%s, 'red planet') AS payload
	)
	SELECT payload ? 'token_plan'
	       AND jsonb_typeof(payload->'token_plan'->'tokens') = 'array'
	       AND jsonb_array_length(payload->'token_plan'->'tokens') >=
	           (payload->>'vector_count')::int
	FROM debug;
), sql_literal("$alias:query"))), 't',
	'live colbert_debug exposes a token plan with at least vector-count tokens');

if (defined $token_plan_golden && $token_plan_golden ne '')
{
	open my $golden_file, '<', $token_plan_golden
	  or die "open token plan golden failed: $!";
	local $/;
	my $golden_json = <$golden_file>;
	close $golden_file or die "close token plan golden failed: $!";
	my $golden = decode_json($golden_json);
	my $plan = $golden->{plans}->[0];
	my $role = $plan->{role} // ($golden->{role} // 'query');
	my $text = $plan->{input_text} // 'red planet';
	my $debug_json = $node->safe_psql(
		'pg_colbert_llama_live',
		sprintf(q(SELECT colbert_debug(%s, %s)::text;),
			sql_literal("$alias:$role"),
			sql_literal($text)),
	);
	my $debug = decode_json($debug_json);
	my @tokens = @{ $debug->{token_plan}->{tokens} };
	my @actual_ids = map { $_->{id} + 0 } @tokens;
	my @actual_retain = map { $_->{retained} ? 1 : 0 } @tokens;
	my @expected_ids = map { $_ + 0 } @{ $plan->{token_ids_after_padding_truncation} };
	my @expected_retain = map { $_ + 0 } @{ $plan->{retain_mask} };

	is_deeply(\@actual_ids, \@expected_ids,
		'live colbert_debug token ids match token-plan golden');
	is_deeply(\@actual_retain, \@expected_retain,
		'live colbert_debug retain mask matches token-plan golden');
	is($debug->{vector_count} + 0, $plan->{final_vector_count} + 0,
		'live colbert_debug vector count matches token-plan golden');
}

is($node->safe_psql('pg_colbert_llama_live', sprintf(q(
	WITH encoded AS (
		SELECT colbert(%s, 'red planet') AS payload
	),
	vectors AS (
		SELECT token_ord,
		       bool_and(value = value
		                AND value < 'Infinity'::float8
		                AND value > '-Infinity'::float8) AS finite,
		       sqrt(sum(value * value)) AS norm
		FROM encoded,
		     LATERAL jsonb_array_elements(payload->'vectors')
				WITH ORDINALITY AS token(vector_json, token_ord),
		     LATERAL (
				SELECT elem.value::float8 AS value
				FROM jsonb_array_elements_text(token.vector_json) elem(value)
		     ) AS vals
		GROUP BY token_ord
	)
	SELECT bool_and(finite AND abs(norm - 1.0) < 1e-3)
	FROM vectors;
), sql_literal("$alias:query"))), 't',
	'live query vectors are finite and L2-normalized');

SKIP:
{
	skip 'token stream fixture is for sauerkraut-modern 128-dim GGUF', 3
	  unless $alias eq 'sauerkraut-modern' && $expected_dim == 128;

	is($node->safe_psql('pg_colbert_llama_live', sprintf(q(
		WITH encoded AS (
			SELECT colbert(%s, 'red planet') AS payload
		)
		SELECT payload->>'count' = '32'
		       AND payload->'token_ids' = '[101, 30522, 2417, 4774, 102, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103]'::jsonb
		FROM encoded;
	), sql_literal("$alias:query"))), 't',
		'live query token stream keeps prefix, specials, and mask expansion');

	is($node->safe_psql('pg_colbert_llama_live', sprintf(q(
		WITH encoded AS (
			SELECT colbert(%s, 'red planet') AS payload
		)
		SELECT payload->>'count' = '5'
		       AND payload->'token_ids' = '[101, 30523, 2417, 4774, 102]'::jsonb
		FROM encoded;
	), sql_literal("$alias:doc"))), 't',
		'live document token stream keeps prefix and specials');

	is($node->safe_psql('pg_colbert_llama_live', sprintf(q(
		WITH encoded AS (
			SELECT colbert(%s, 'Mars is often called the red planet.') AS payload
		)
		SELECT payload->>'count' = '10'
		       AND NOT EXISTS (
		           SELECT 1
		           FROM jsonb_array_elements_text(payload->'token_ids') elem(token_id)
		           WHERE elem.token_id = '1012'
		       )
		FROM encoded;
	), sql_literal("$alias:doc"))), 't',
		'live document encoding drops punctuation skiplist tokens');
}

is($node->safe_psql('pg_colbert_llama_live', sprintf(q(
	WITH q AS (
		SELECT colbert(%s, 'same passage text') AS payload
	),
	d AS (
		SELECT colbert(%s, 'same passage text') AS payload
	),
	qv AS (
		SELECT ord, value::float8 AS q_value
		FROM q,
		     LATERAL jsonb_array_elements_text((payload->'vectors')->0)
				WITH ORDINALITY AS elem(value, ord)
	),
	dv AS (
		SELECT ord, value::float8 AS d_value
		FROM d,
		     LATERAL jsonb_array_elements_text((payload->'vectors')->0)
				WITH ORDINALITY AS elem(value, ord)
	)
	SELECT max(abs(qv.q_value - dv.d_value)) > 1e-7
	FROM qv
	JOIN dv USING (ord);
), sql_literal("$alias:query"), sql_literal("$alias:doc"))), 't',
	'live query and document prefixes produce different embeddings');

$node->safe_psql('pg_colbert_llama_live', sprintf(q(
	DO $$
	DECLARE
		query_count int;
		doc_count int;
	BEGIN
		PERFORM set_config('pg_colbert_llama.max_query_vectors', '1', true);
		SELECT turbohybrid_multivector_count(
			colbert_mv(%s, 'red planet token cap check')
		) INTO query_count;
		IF query_count <> 1 THEN
			RAISE EXCEPTION 'expected query cap to produce 1 vector, got %%',
				query_count;
		END IF;

		PERFORM set_config('pg_colbert_llama.max_doc_vectors', '2', true);
		SELECT turbohybrid_multivector_count(
			colbert_mv(%s, 'Mars is the red planet with rusty dust')
		) INTO doc_count;
		IF doc_count < 1 OR doc_count > 2 THEN
			RAISE EXCEPTION 'expected doc cap to produce 1..2 vectors, got %%',
				doc_count;
		END IF;
	END
	$$;
), sql_literal("$alias:query"), sql_literal("$alias:doc")));
pass('live vector count caps are enforced');

$node->safe_psql('pg_colbert_llama_live', sprintf(q(
	CREATE TABLE passages (
		id int PRIMARY KEY,
		body text NOT NULL,
		body_tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', body)) STORED,
		colbert turbohybrid_multivector NOT NULL
	);

	INSERT INTO passages (id, body, colbert) VALUES
		(1, 'Mars is the red planet.', colbert_mv(%s, 'Mars is the red planet.')),
		(2, 'Earth has blue oceans.', colbert_mv(%s, 'Earth has blue oceans.')),
		(3, 'Jupiter is a gas giant.', colbert_mv(%s, 'Jupiter is a gas giant.'));

	CREATE INDEX passages_live_idx ON passages USING turbohybrid (
		colbert multivector_maxsim_ip_turbohybrid_ops,
		body_tsv bm25_tsvector_turbohybrid_ops
	);
), sql_literal("$alias:doc"), sql_literal("$alias:doc"), sql_literal("$alias:doc")));

$node->safe_psql('pg_colbert_llama_live', sprintf(q(
	DO $$
	DECLARE
		top_id int;
		stats jsonb;
	BEGIN
		SET LOCAL enable_seqscan = off;

		SELECT id INTO top_id
		FROM passages
		ORDER BY colbert <~> turbohybrid_query(
			multivector_query => colbert_mv(%s, 'red planet'),
			dense_k => 20,
			final_k => 1
		)
		LIMIT 1;

		IF top_id IS NULL THEN
			RAISE EXCEPTION 'expected live dense multivector search to return a row';
		END IF;

		stats := turbohybrid_last_scan_stats();
		IF stats->>'multivector_enabled' <> 'true' OR
		   stats->>'multivector_branch_used' <> 'true' OR
		   stats->>'index_used' <> 'true' THEN
			RAISE EXCEPTION 'expected live dense search to use multivector index, stats=%%',
				stats;
		END IF;
	END
	$$;
), sql_literal("$alias:query")));
pass('live ColBERT multivector output can drive indexed dense search');

$node->safe_psql('pg_colbert_llama_live', sprintf(q(
	DO $$
	DECLARE
		top_id int;
		stats jsonb;
	BEGIN
		SET LOCAL enable_seqscan = off;

		SELECT id INTO top_id
		FROM passages
		ORDER BY colbert <~> turbohybrid_query(
			multivector_query => colbert_mv(%s, 'red planet'),
			text_query => websearch_to_tsquery('simple', 'red planet'),
			fusion => 'rrf',
			dense_k => 20,
			bm25_k => 20,
			final_k => 1
		)
		LIMIT 1;

		IF top_id <> 1 THEN
			RAISE EXCEPTION 'expected live hybrid search to return id 1, got %%', top_id;
		END IF;

		stats := turbohybrid_last_scan_stats();
		IF stats->>'multivector_enabled' <> 'true' OR
		   stats->>'multivector_branch_used' <> 'true' OR
		   stats->>'index_used' <> 'true' THEN
			RAISE EXCEPTION 'expected live hybrid search to use multivector dense index, stats=%%',
				stats;
		END IF;
	END
	$$;
), sql_literal("$alias:query")));
pass('live ColBERT multivector output can drive indexed pgturbohybrid hybrid search');

done_testing();
