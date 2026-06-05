use strict;
use warnings FATAL => 'all';
use Test::More;
use File::Copy qw(copy);
use File::Temp qw(tempdir);
use PostgreSQL::Test::Cluster;

my $srcdir = $ENV{TESTDIR} // '.';
my $fixture = "$srcdir/test/fixtures/profile_sauerkraut_15m_minimal.json";
my $model_dir = tempdir(CLEANUP => 1);

copy($fixture, "$model_dir/fake-model.gguf.colbert_profile.json")
  or die "copy profile fixture failed: $!";

open my $invalid, '>', "$model_dir/bad-model.gguf.colbert_profile.json"
  or die "open invalid profile failed: $!";
print {$invalid} '{"schema":"wrong","output_dim":4,"query":{"max_length":32},"document":{"max_length":300}}';
close $invalid or die "close invalid profile failed: $!";

open my $bad_projection, '>', "$model_dir/bad-projection.gguf.colbert_profile.json"
  or die "open bad projection profile failed: $!";
print {$bad_projection} q({
  "schema":"pg_colbert_profile_v1",
  "output_dim":4,
  "query":{"prefix":"[Q] ","max_length":32,"pad_to":32},
  "document":{"prefix":"[D] ","max_length":300},
  "projection":{
    "kind":"dense",
    "output_dim":4,
    "modules":[{"type":"linear","input_dim":4,"output_dim":4,"activation":"gelu"}]
  }
});
close $bad_projection or die "close bad projection profile failed: $!";

open my $bias_projection, '>', "$model_dir/bias-projection.gguf.colbert_profile.json"
  or die "open bias projection profile failed: $!";
print {$bias_projection} q({
  "schema":"pg_colbert_profile_v1",
  "output_dim":2,
  "query":{"prefix":"[Q] ","max_length":32,"pad_to":32},
  "document":{"prefix":"[D] ","max_length":300},
  "projection":{
    "kind":"module_chain",
    "output_dim":2,
    "modules":[
      {
        "type":"dense",
        "input_dim":4,
        "output_dim":4,
        "weight_tensor":"custom.proj.weight",
        "bias_tensor":"custom.proj.bias",
        "activation":"identity"
      },
      {"type":"truncate","output_dim":2}
    ]
  }
});
close $bias_projection or die "close bias projection profile failed: $!";

sub sql_literal
{
	my ($value) = @_;
	$value =~ s/'/''/g;
	return "'$value'";
}

my $node = PostgreSQL::Test::Cluster->new('pg_colbert_llama_profile_stub_' . $$);
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE DATABASE pg_colbert_llama_profile_stub;');
$node->safe_psql('pg_colbert_llama_profile_stub', q(
	CREATE EXTENSION vector;
	CREATE EXTENSION pgturbohybrid;
	CREATE EXTENSION pg_colbert_llama;
));
$node->safe_psql('postgres', sprintf(q(
	ALTER DATABASE pg_colbert_llama_profile_stub SET pg_colbert_llama.model_dir = %s;
	ALTER DATABASE pg_colbert_llama_profile_stub SET pg_colbert_llama.expected_dim = 4;
), sql_literal($model_dir)));

is($node->safe_psql('pg_colbert_llama_profile_stub', q(
	WITH info AS (
		SELECT colbert_model_info('fake-model:query') AS payload
	)
	SELECT payload->>'profile_loaded' = 'true'
	       AND payload->>'profile_source' = 'sidecar'
	       AND payload->>'profile_schema' = 'pg_colbert_profile_v1'
	       AND payload->>'compatibility_level' = 'profile_loaded'
	       AND payload->>'query_length_source' = 'profile'
	       AND payload->>'document_length_source' = 'profile'
	       AND payload->>'skiplist_token_count' = '1'
	       AND payload->>'attention_mask_status' = 'approximated'
	       AND payload->>'strict_profile' = 'false'
	FROM info;
)), 't', 'stub model info reports loaded sidecar profile fields');

is($node->safe_psql('pg_colbert_llama_profile_stub', q(
	WITH debug AS (
		SELECT colbert_debug('fake-model:query', 'red planet') AS payload
	)
	SELECT payload->>'engine' = 'stub'
	       AND payload->>'profile_source' = 'sidecar'
	       AND payload->>'prefix' = '[Q] '
	       AND payload->'token_plan' ? 'tokens'
	       AND jsonb_array_length(payload->'token_plan'->'tokens') = (payload->>'vector_count')::int
	       AND (payload->'token_plan'->'tokens'->0)->>'retain_reason' = 'retained_query'
	FROM debug;
)), 't', 'colbert_debug exposes token plan shape in stub mode');

is($node->safe_psql('pg_colbert_llama_profile_stub', q(
	WITH info AS (
		SELECT colbert_model_info('missing-model:query') AS payload
	)
	SELECT payload->>'profile_loaded' = 'false'
	       AND payload->>'profile_source' = 'guc_fallback'
	       AND payload->>'query_length_source' = 'guc'
	       AND payload->>'document_length_source' = 'guc'
	FROM info;
)), 't', 'missing sidecar falls back to GUC-derived profile');

my ($ret, $stdout, $stderr) = $node->psql(
	'pg_colbert_llama_profile_stub',
	q(SELECT colbert_model_info('bad-model:query');),
);
isnt($ret, 0, 'invalid sidecar profile fails cleanly');
like($stderr, qr/profile schema must be pg_colbert_profile_v1/,
	'invalid sidecar profile reports schema error');

($ret, $stdout, $stderr) = $node->psql(
	'pg_colbert_llama_profile_stub',
	q(SELECT colbert_model_info('bad-projection:query');),
);
isnt($ret, 0, 'unsupported projection activation fails cleanly');
like($stderr, qr/unsupported activation "gelu"/,
	'unsupported projection activation is named');

is($node->safe_psql('pg_colbert_llama_profile_stub', q(
	SET pg_colbert_llama.expected_dim = 2;
	WITH info AS (
		SELECT colbert_model_info('bias-projection:query') AS payload
	)
	SELECT payload->>'profile_loaded' = 'true'
	       AND payload->>'projection_kind' = 'module_chain'
	       AND payload->>'projection_module_count' = '2'
	       AND payload->'projection_modules'->0->>'type' = 'dense'
	       AND payload->'projection_modules'->0->>'weight_tensor' = 'custom.proj.weight'
	       AND payload->'projection_modules'->0->>'bias_tensor' = 'custom.proj.bias'
	       AND payload->'projection_modules'->0->>'bias' = 'true'
	       AND payload->'projection_modules'->1->>'type' = 'truncate'
	       AND payload->'projection_modules'->1->>'output_dim' = '2'
	FROM info;
)), 't', 'profile parser reports dense bias and truncate projection modules');

$node->safe_psql('pg_colbert_llama_profile_stub', q(
	SET pg_colbert_llama.strict_profile = off;
	SELECT colbert_model_info('fake-model:query');
));
pass('strict_profile off allows approximated query expansion mask');

($ret, $stdout, $stderr) = $node->psql(
	'pg_colbert_llama_profile_stub',
	q(SET pg_colbert_llama.strict_profile = on; SELECT colbert_model_info('fake-model:query');),
);
isnt($ret, 0, 'strict_profile on rejects unsupported exact expansion mask');
like($stderr, qr/query expansion attention/,
	'strict_profile rejection names the attention mask limitation');

done_testing();
