# Exercises the turbohybrid_multivector binary RECEIVE path
# (pgturbohybrid_multivector_recv) via COPY ... WITH (FORMAT binary).
#
# The plain-text dump/restore test (003) and the fuzz suite only cover the
# text input function; pg_dump even in -Fc custom format replays data as TEXT
# COPY, so neither reaches recv. COPY FROM a binary file is the one path that
# feeds bytes straight into recv. This test confirms (a) a well-formed binary
# multivector round-trips through recv, and (b) malformed binary input is
# rejected with a clean error and never crashes the backend.
#
# Wire layout recv expects (see pgturbohybrid_multivector_recv): four 4-byte
# big-endian ints -- formatVersion, dim, count, flags -- then count*dim
# big-endian float4 values (non-context format, version 1).
use strict;
use warnings FATAL => 'all';
use Test::More;
use File::Temp qw(tempfile);

BEGIN
{
	eval {
		require PostgreSQL::Test::Cluster;
		PostgreSQL::Test::Cluster->import();
		1;
	} or plan skip_all => 'PostgreSQL::Test::Cluster is not available';
}

my $MV_VERSION          = 1;    # PGTURBOHYBRID_MULTIVECTOR_BINARY_VERSION
my $MV_FLAG_CONTEXTS    = 1;    # PGTURBOHYBRID_MULTIVECTOR_FLAG_CONTEXTS

# Build the binary SEND/RECEIVE representation of a (non-context) multivector.
sub mv_field
{
	my (%a) = @_;
	my $version = defined $a{version} ? $a{version} : $MV_VERSION;
	my $dim     = defined $a{dim}     ? $a{dim}     : 2;
	my $count   = defined $a{count}   ? $a{count}   : 1;
	my $flags   = defined $a{flags}   ? $a{flags}   : 0;
	my @floats  = defined $a{floats}  ? @{ $a{floats} } : (1.0, 0.0);

	my $bytes = pack('l>', $version)
		. pack('l>', $dim)
		. pack('l>', $count)
		. pack('l>', $flags);
	$bytes .= pack('l>', $a{context_count}) if defined $a{context_count};
	$bytes .= pack('f>', $_) for @floats;
	$bytes .= $a{trailing} if defined $a{trailing};
	return $bytes;
}

# Wrap one field in a single-row COPY BINARY stream and write it to a temp file.
# $declared_len overrides the 4-byte field-length prefix (to forge truncation).
sub copy_binary_file
{
	my ($field, $declared_len) = @_;
	$declared_len = length($field) unless defined $declared_len;

	my $data = "PGCOPY\n\377\r\n\0";    # 11-byte signature
	$data .= pack('N', 0);              # flags
	$data .= pack('N', 0);              # header extension length
	$data .= pack('n', 1);             # field count for the row
	$data .= pack('l>', $declared_len);
	$data .= $field;
	$data .= pack('n', 0xFFFF);        # file trailer (-1)

	my ($fh, $path) = tempfile('pgth-recv-XXXXXX',
		SUFFIX => '.bin', TMPDIR => 1, UNLINK => 1);
	binmode $fh;
	print $fh $data;
	close $fh;
	return $path;
}

my $node = PostgreSQL::Test::Cluster->new('multivector_recv_binary');
$node->init;
$node->start;

$node->safe_psql('postgres', q(
	CREATE EXTENSION vector;
	CREATE EXTENSION pgturbohybrid; CREATE EXTENSION pgturbohybrid_experimental;
	CREATE TABLE recv_t (col turbohybrid_multivector);
));

# COPY a single binary field and report (rc, stderr).
sub copy_one
{
	my ($field, $declared_len) = @_;
	my $path = copy_binary_file($field, $declared_len);
	my ($rc, $stdout, $stderr) = $node->psql('postgres',
		"COPY recv_t FROM '$path' WITH (FORMAT binary)");
	return ($rc, $stderr);
}

# Each malformed input must be rejected with a clean ERROR, and the backend
# must remain alive afterwards (no crash / no restart).
sub recv_rejects
{
	my ($label, $field, $declared_len) = @_;
	my ($rc, $stderr) = copy_one($field, $declared_len);
	isnt($rc, 0, "$label: rejected (non-zero exit)");
	like($stderr, qr/ERROR:/, "$label: clean ERROR, not a crash");
	is($node->safe_psql('postgres', 'SELECT 1'), '1',
		"$label: backend still alive");
}

# --- positive control: a well-formed binary multivector round-trips ---------
{
	my $path = copy_binary_file(mv_field(dim => 2, count => 1, floats => [1.0, 0.0]));
	my ($rc, $stdout, $stderr) = $node->psql('postgres',
		"COPY recv_t FROM '$path' WITH (FORMAT binary)");
	is($rc, 0, 'well-formed binary multivector accepted by recv')
		or diag($stderr);
	is($node->safe_psql('postgres', q(
		SELECT col::text FROM recv_t ORDER BY ctid DESC LIMIT 1;
	)), 'turbohybrid_multivector(dim=2,count=1,values=[[1,0]])',
		'binary-received multivector has the expected value');
}

# --- negatives: every malformed shape errors cleanly, never crashes --------
recv_rejects('unsupported version', mv_field(version => 99));
recv_rejects('non-positive dim', mv_field(dim => 0));
recv_rejects('negative count', mv_field(count => -1));
recv_rejects('truncated float payload',
	mv_field(dim => 2, count => 1, floats => [1.0]));    # 1 float, needs 2
recv_rejects('empty field', '', 0);
recv_rejects('trailing garbage',
	mv_field(dim => 2, count => 1, floats => [1.0, 0.0], trailing => pack('N', 0)));
recv_rejects('contexts flag on a v1 payload',
	mv_field(version => 1, flags => $MV_FLAG_CONTEXTS, context_count => 1));

# Sanity: the one valid row is the only row that made it in.
is($node->safe_psql('postgres', 'SELECT count(*) FROM recv_t;'), '1',
	'exactly the one well-formed row was inserted');

$node->stop;
done_testing();
