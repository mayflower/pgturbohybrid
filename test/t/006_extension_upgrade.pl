# Exercises the extension upgrade path: ALTER EXTENSION pgturbohybrid UPDATE.
#
# 0.1.1 is the first version with a predecessor, so this is the first test of
# the upgrade machinery -- PostgreSQL resolving an update path, running the
# sql/pgturbohybrid--0.1.0--0.1.1.sql migration, and bumping the recorded
# catalog version. It confirms (a) a fresh install lands on the new default,
# (b) the prior version can still be installed and upgraded, and (c) the
# index/AM remain fully functional across the upgrade.
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

my $node = PostgreSQL::Test::Cluster->new('extension_upgrade');
$node->init;
$node->start;

sub ext_version
{
	my ($db) = @_;
	return $node->safe_psql($db,
		"SELECT extversion FROM pg_extension WHERE extname = 'pgturbohybrid';");
}

# A real index scan, used to prove the AM works both before and after upgrade.
sub nearest_id
{
	my ($db) = @_;
	return $node->safe_psql($db, q(
		SET enable_seqscan = off;
		SELECT id FROM docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector, final_k => 1)
		LIMIT 1;
	));
}

# --- a fresh install lands on the new default version ----------------------
$node->safe_psql('postgres', 'CREATE DATABASE fresh;');
$node->safe_psql('fresh', 'CREATE EXTENSION vector; CREATE EXTENSION pgturbohybrid;');
is(ext_version('fresh'), '0.2.0',
	'fresh CREATE EXTENSION installs the new default version 0.2.0');

# --- upgrade path: install the prior version, then ALTER ... UPDATE --------
$node->safe_psql('postgres', 'CREATE DATABASE upgrade;');
$node->safe_psql('upgrade', q(
	CREATE EXTENSION vector;
	CREATE EXTENSION pgturbohybrid VERSION '0.1.0';
	CREATE TABLE docs (id int PRIMARY KEY, embedding vector(3));
	INSERT INTO docs VALUES (1, '[1,0,0]'), (2, '[0,1,0]'), (3, '[0,0,1]');
	CREATE INDEX docs_idx ON docs
		USING turbohybrid (embedding vector_cosine_turbohybrid_ops);
));
is(ext_version('upgrade'), '0.1.0',
	"CREATE EXTENSION ... VERSION '0.1.0' installs the prior version");
is(nearest_id('upgrade'), '1', 'turbohybrid index scan works at 0.1.0 (pre-upgrade)');

# the upgrade itself
$node->safe_psql('upgrade', "ALTER EXTENSION pgturbohybrid UPDATE TO '0.1.1';");
is(ext_version('upgrade'), '0.1.1',
	"ALTER EXTENSION ... UPDATE TO '0.1.1' bumps the recorded catalog version");

# 0.1.2 is the last monolithic catalog version.
$node->safe_psql('upgrade', "ALTER EXTENSION pgturbohybrid UPDATE TO '0.1.2';");
is(ext_version('upgrade'), '0.1.2', 'upgrade reaches the final monolithic version');

# The ownership split cannot create a second extension from inside ALTER
# EXTENSION. It must abort transactionally before leaving unowned objects.
my ($upgrade_rc, $upgrade_out, $upgrade_err) = $node->psql('upgrade',
	'ALTER EXTENSION pgturbohybrid UPDATE;', extra_params => ['-v', 'ON_ERROR_STOP=1']);
isnt($upgrade_rc, 0, 'monolithic-to-split update is refused');
like($upgrade_out . $upgrade_err, qr/requires an explicit core\/experimental migration/,
	'upgrade refusal names the required migration');
is(ext_version('upgrade'), '0.1.2', 'failed split update leaves extension version intact');

# the public catalog surface is intact and the index still scans after upgrade
is($node->safe_psql('upgrade', q(
	SELECT concat_ws(',',
		EXISTS (SELECT 1 FROM pg_am WHERE amname = 'turbohybrid'),
		to_regtype('turbohybrid_query') IS NOT NULL,
		EXISTS (SELECT 1 FROM pg_opclass WHERE opcname = 'vector_cosine_turbohybrid_ops'),
		EXISTS (SELECT 1 FROM pg_operator WHERE oprname = '<~>')
	);
)), 't,t,t,t', 'public catalog surface intact after upgrade');
is(nearest_id('upgrade'), '1',
	'turbohybrid index scan still works after transactional split refusal');

$node->stop;
done_testing();
