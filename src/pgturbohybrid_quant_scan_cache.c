#include "postgres.h"

#include <errno.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#ifndef WIN32
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "access/genam.h"
#include "catalog/pg_class.h"
#include "catalog/pg_database.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "portability/instr_time.h"
#include "storage/bufmgr.h"
#include "storage/fd.h"
#include "utils/fmgrprotos.h"
#include "utils/jsonb.h"
#include "utils/memutils.h"
#include "utils/numeric.h"
#include "utils/rel.h"
#include "utils/syscache.h"
#include "utils/timestamp.h"

#include "pgturbohybrid_jsonb_compat.h"
#include "pgturbohybrid_am.h"
#include "pgturbohybrid_quant.h"

static PgturbohybridGraphNativeCache *pgturbohybridGraphCacheList = NULL;
static PgturbohybridGraphDocSidecarCache *pgturbohybridGraphDocSidecarCacheList = NULL;

#define PGTURBOHYBRID_GRAPH_SHARED_CACHE_MAGIC 0x54485343U
#define PGTURBOHYBRID_GRAPH_SHARED_CACHE_VERSION 2U
#define PGTURBOHYBRID_GRAPH_SHARED_CACHE_ENDIAN 0x01020304U
#define PGTURBOHYBRID_GRAPH_SHARED_CACHE_WAIT_US 5000000L
#define PGTURBOHYBRID_GRAPH_SHARED_CACHE_POLL_US 10000L
#define PGTURBOHYBRID_GRAPH_SHARED_CACHE_GC_INTERVAL_MS 60000L
#define PGTURBOHYBRID_GRAPH_SHARED_CACHE_TEMP_STALE_SECONDS 600

static float
PgturbohybridGraphHalfToFloat(uint16 half)
{
	uint32		sign = ((uint32) half & 0x8000U) << 16;
	uint32		exp = ((uint32) half >> 10) & 0x1fU;
	uint32		mant = (uint32) half & 0x03ffU;
	uint32		bits;
	float		value;

	if (exp == 0)
	{
		if (mant == 0)
			bits = sign;
		else
		{
			exp = 1;
			while ((mant & 0x0400U) == 0)
			{
				mant <<= 1;
				exp--;
			}
			mant &= 0x03ffU;
			bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
		}
	}
	else if (exp == 31)
		bits = sign | 0x7f800000U | (mant << 13);
	else
		bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);

	memcpy(&value, &bits, sizeof(value));
	return value;
}

typedef struct PgturbohybridGraphCacheIdentity
{
	Oid			databaseOid;
	Oid			tablespaceOid;
	Oid			relNumber;
	Oid			relationOid;	/* diagnostics only */
	uint32		forkNumber;
	uint32		graphFormatVersion;
	uint64		graphGeneration;
	uint32		postgresMajor;
	uint32		blockSize;
	uint32		endianness;
	uint32		sharedNodeSize;
	uint32		sharedNodeAlign;
	uint32		payloadRefSize;
	uint32		payloadRefAlign;
} PgturbohybridGraphCacheIdentity;

typedef struct PgturbohybridGraphSharedSegment
{
	uint64		offset;
	uint64		length;
} PgturbohybridGraphSharedSegment;

typedef struct PgturbohybridGraphSharedCacheHeader
{
	uint32		magic;
	uint32		version;
	uint64		key;
	uint64		fileSize;
	PgturbohybridGraphCacheIdentity identity;
	uint32		dimensions;
	uint16		m;
	uint16		graphMaxLevel;
	uint16		graphFlags;
	uint16		tqBits;
	uint32		tqNodeCount;
	uint32		tqEntryNodeId;
	uint16		tqCodeBytes;
	uint16		tqPayloadCount;
	uint16		tqPayloadBytes;
	uint16		tqResidualRerankBytes;
	BlockNumber tqCodeStartBlkno;
	BlockNumber tqAdjStartBlkno;
	BlockNumber tqExactStartBlkno;
	BlockNumber tqCorrectionStartBlkno;
	uint32		adjRecordCount;
	uint64		neighborValueCount;
	uint32		payloadRefCount;
	uint32		exactBytes;
	PgturbohybridGraphSharedSegment nodes;
	PgturbohybridGraphSharedSegment codeArena;
	PgturbohybridGraphSharedSegment payloadArena;
	PgturbohybridGraphSharedSegment residualArena;
	PgturbohybridGraphSharedSegment exactArena;
	PgturbohybridGraphSharedSegment neighborCounts;
	PgturbohybridGraphSharedSegment neighborOffsets;
	PgturbohybridGraphSharedSegment neighborData;
	PgturbohybridGraphSharedSegment payloadRefs;
	uint64		residentCodeBytes;
	uint64		residentAdjBytes;
	uint64		residentExactBytes;
	uint64		residentTotalBytes;
} PgturbohybridGraphSharedCacheHeader;

typedef struct PgturbohybridGraphSharedNode
{
	ItemPointerData heaptid;
	uint16		payloadMask;
	int			level;
	BlockNumber exactBlkno;
	OffsetNumber exactOffno;
	float		scale;
	float		norm;
	float		codeNorm;
	float		ecCorrection;
	uint16		flags;
	bool		loaded;
} PgturbohybridGraphSharedNode;

typedef struct PgturbohybridGraphSharedMap
{
	uint64		key;
	Oid			relid;
	Oid			relfilenumber;
	PgturbohybridGraphCacheIdentity identity;
	void	   *base;
	Size		size;
	PgturbohybridGraphScanStorage view;
	MemoryContext ctx;
	int64		attachUs;
	int64		buildUs;
	int64		waitUs;
	bool		builtThisBackend;
	struct PgturbohybridGraphSharedMap *next;
} PgturbohybridGraphSharedMap;

static PgturbohybridGraphSharedMap *pgturbohybridGraphSharedMapList = NULL;
static TimestampTz pgturbohybridGraphSharedLastGc = 0;

typedef struct PgturbohybridGraphSharedWarning
{
	Oid			databaseOid;
	Oid			relid;
	int			reason;
	struct PgturbohybridGraphSharedWarning *next;
} PgturbohybridGraphSharedWarning;

static PgturbohybridGraphSharedWarning *pgturbohybridGraphSharedWarningList = NULL;

static void
PgturbohybridGraphWarnSharedFailureOnce(Relation index,
										PgturbohybridGraphNativeCacheReason reason)
{
	PgturbohybridGraphSharedWarning *warning;

	for (warning = pgturbohybridGraphSharedWarningList;
		 warning != NULL; warning = warning->next)
		if (warning->databaseOid == MyDatabaseId &&
			warning->relid == RelationGetRelid(index))
			return;

	warning = MemoryContextAlloc(TopMemoryContext, sizeof(*warning));
	warning->databaseOid = MyDatabaseId;
	warning->relid = RelationGetRelid(index);
	warning->reason = (int) reason;
	warning->next = pgturbohybridGraphSharedWarningList;
	pgturbohybridGraphSharedWarningList = warning;
	ereport(WARNING,
			(errmsg("shared native cache unavailable for index %s; using uncached scans",
					RelationGetRelationName(index)),
			 errdetail("cache failure reason code: %d", (int) reason)));
}

static inline int64
PgturbohybridGraphElapsedUsSince(instr_time start)
{
	instr_time	elapsed;

	INSTR_TIME_SET_CURRENT(elapsed);
	INSTR_TIME_SUBTRACT(elapsed, start);
	return (int64) INSTR_TIME_GET_MICROSEC(elapsed);
}

/*
 * Native scan cache size cap, in bytes, from the
 * turbohybrid.native_cache_max_mb GUC (default 2048 MB).  An index whose
 * resident working set fits under the cap can be fully loaded into the
 * selected native cache scope so warm scans read 0 code pages; larger indexes
 * fall back to per-scan page loading.  Raising the cap past ~1 GB requires the
 * huge allocations below (the code arena for 1M x 3072 dims is ~1.5 GB).  When
 * explicitly using per_backend, size the cap to host RAM and connection count.
 */
static inline Size
PgturbohybridGraphNativeCacheMaxBytes(void)
{
	return (Size) pgturbohybrid_native_cache_max_mb * 1024 * 1024;
}

static bool
PgturbohybridGraphShouldCacheMultiVectorDocSidecar(PgturbohybridGraphMetaPageData *meta)
{
	if (meta->tqMultivectorGraphMode !=
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES)
		return true;
	switch ((PgturbohybridMultiVectorDocStorageCacheMode)
			pgturbohybrid_multivector_doc_storage_cache)
	{
		case PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_RESIDENT:
			return true;
		case PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_PAGED:
			return false;
		case PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_AUTO:
		default:
			/*
			 * The native graph cache is shared/per-backend graph state.  In
			 * document_nodes mode, auto must not silently materialize every
			 * full document multivector before the candidate source is known;
			 * proxy_vector can use a paged metadata cache and touch full
			 * vectors only for bounded exact rerank.  Users can still force
			 * full resident sidecar storage with the explicit resident mode.
			 */
			return false;
	}
}

static inline Size
PgturbohybridGraphNativeCacheWarnBytes(void)
{
	return (Size) pgturbohybrid_native_cache_warn_mb * 1024 * 1024;
}

static const char *
PgturbohybridGraphNativeCachePolicyNameForLog(int policy)
{
	switch ((PgturbohybridNativeCachePolicy) policy)
	{
		case PGTURBOHYBRID_NATIVE_CACHE_POLICY_OFF:
			return "off";
		case PGTURBOHYBRID_NATIVE_CACHE_POLICY_PER_BACKEND:
			return "per_backend";
		case PGTURBOHYBRID_NATIVE_CACHE_POLICY_SHARED:
			return "shared";
		case PGTURBOHYBRID_NATIVE_CACHE_POLICY_AUTO:
		default:
			return "auto";
	}
}

static bool
PgturbohybridGraphNativeCacheWarns(PgturbohybridGraphNativeCache *cache)
{
	Size		warnBytes;

	if (cache == NULL || pgturbohybrid_native_cache_warn_mb <= 0)
		return false;
	warnBytes = PgturbohybridGraphNativeCacheWarnBytes();
	return warnBytes > 0 && cache->residentTotalBytes >= warnBytes;
}

static int
PgturbohybridGraphEffectiveScanNativeCachePolicy(void)
{
	if (pgturbohybrid_native_cache_policy != PGTURBOHYBRID_NATIVE_CACHE_POLICY_AUTO)
		return pgturbohybrid_native_cache_policy;

#ifdef WIN32
	/*
	 * The current shared implementation is mmap-backed and disabled on
	 * Windows, so auto keeps the previous per-backend behavior there.
	 */
	return PGTURBOHYBRID_NATIVE_CACHE_POLICY_PER_BACKEND;
#else
	return PGTURBOHYBRID_NATIVE_CACHE_POLICY_SHARED;
#endif
}

static int
PgturbohybridGraphPayloadRefCompare(const void *a, const void *b)
{
	const PgturbohybridGraphPayloadRef *ra = (const PgturbohybridGraphPayloadRef *) a;
	const PgturbohybridGraphPayloadRef *rb = (const PgturbohybridGraphPayloadRef *) b;

	if (ra->payloadSlot < rb->payloadSlot)
		return -1;
	if (ra->payloadSlot > rb->payloadSlot)
		return 1;
	if (ra->payloadValue < rb->payloadValue)
		return -1;
	if (ra->payloadValue > rb->payloadValue)
		return 1;
	return (ra->nodeId > rb->nodeId) - (ra->nodeId < rb->nodeId);
}


static bool
PgturbohybridGraphShouldCacheExactVectors(Relation index, PgturbohybridGraphMetaPageData *meta)
{
	int			policy = PgturbohybridGraphGetGraphExactCache(index);
	Size		exactBytes;
	Size		totalBytes;

	if (policy == PGTURBOHYBRID_GRAPH_EXACT_CACHE_OFF ||
		meta->tqNodeCount == 0 || meta->dimensions == 0 ||
		!BlockNumberIsValid(meta->tqExactStartBlkno))
		return false;
	if (policy == PGTURBOHYBRID_GRAPH_EXACT_CACHE_ON)
		return true;

	exactBytes = PGTURBOHYBRID_VECTOR_SIZE(meta->dimensions);
	if (meta->tqNodeCount > SIZE_MAX / exactBytes)
		return false;
	totalBytes = exactBytes * (Size) meta->tqNodeCount;

	return totalBytes <= PGTURBOHYBRID_GRAPH_EXACT_CACHE_AUTO_MAX_BYTES;
}

static bool
PgturbohybridGraphArenaBytes(uint32 count, Size itemBytes, Size *totalBytes)
{
	if (totalBytes != NULL)
		*totalBytes = 0;
	if (count == 0 || itemBytes == 0)
		return true;
	if ((Size) count > SIZE_MAX / itemBytes)
		return false;
	if (totalBytes != NULL)
		*totalBytes = (Size) count * itemBytes;
	return AllocSizeIsValid((Size) count * itemBytes);
}

/*
 * Same, but allows huge (> MaxAllocSize) sizes, for the contiguous data arenas
 * (code / residual / payload) that are allocated with palloc_extended(HUGE).
 * The per-node metadata arrays keep the 1 GB AllocSize bound, which caps a
 * cacheable index at the point where those arrays would exceed it (graceful
 * fallback to uncached scanning); the code arena is the only one that grows
 * past 1 GB in the realistic range (1M x 3072 dims => ~1.5 GB).
 */
static bool
PgturbohybridGraphArenaBytesHuge(uint32 count, Size itemBytes, Size *totalBytes)
{
	if (totalBytes != NULL)
		*totalBytes = 0;
	if (count == 0 || itemBytes == 0)
		return true;
	if ((Size) count > SIZE_MAX / itemBytes)
		return false;
	if (totalBytes != NULL)
		*totalBytes = (Size) count * itemBytes;
	return AllocHugeSizeIsValid((Size) count * itemBytes);
}

static bool
PgturbohybridGraphCheckedMultiplyU64(uint64 count, uint64 width, uint64 *result)
{
	if (width != 0 && count > UINT64_MAX / width)
		return false;
	*result = count * width;
	return true;
}

static void
PgturbohybridGraphCacheIdentityInit(Relation index,
									PgturbohybridGraphMetaPageData *meta,
									PgturbohybridGraphCacheIdentity *identity)
{
	memset(identity, 0, sizeof(*identity));
	identity->databaseOid = MyDatabaseId;
	PgturbohybridGraphGetRelationLocator(index, &identity->tablespaceOid,
										 &identity->relNumber);
	identity->relationOid = RelationGetRelid(index);
	identity->forkNumber = MAIN_FORKNUM;
	identity->graphFormatVersion = meta->version;
	identity->graphGeneration = meta->graphGeneration;
	identity->postgresMajor = PG_VERSION_NUM / 10000;
	identity->blockSize = BLCKSZ;
	identity->endianness = PGTURBOHYBRID_GRAPH_SHARED_CACHE_ENDIAN;
	identity->sharedNodeSize = sizeof(PgturbohybridGraphSharedNode);
	identity->sharedNodeAlign = MAXIMUM_ALIGNOF;
	identity->payloadRefSize = sizeof(PgturbohybridGraphPayloadRef);
	identity->payloadRefAlign = MAXIMUM_ALIGNOF;
}

static bool
PgturbohybridGraphCacheIdentityEqual(const PgturbohybridGraphCacheIdentity *a,
									 const PgturbohybridGraphCacheIdentity *b)
{
	return memcmp(a, b, sizeof(*a)) == 0;
}

static void *
PgturbohybridGraphSharedCheckedPtr(void *base, uint64 fileSize,
								  uint64 offset, uint64 length)
{
	if (length == 0)
		return NULL;
	if (offset == 0 || offset > fileSize || length > fileSize - offset)
		return NULL;
	return (void *) ((char *) base + offset);
}

static uint64
PgturbohybridGraphHashU64(uint64 hash, uint64 value)
{
	hash ^= value;
	hash *= UINT64CONST(1099511628211);
	return hash;
}

static uint64
PgturbohybridGraphSharedCacheKey(Relation index, PgturbohybridGraphMetaPageData *meta)
{
	uint64		hash = UINT64CONST(1469598103934665603);
	PgturbohybridGraphCacheIdentity identity;

	PgturbohybridGraphCacheIdentityInit(index, meta, &identity);
	for (Size offset = 0; offset < sizeof(identity); offset += sizeof(uint64))
	{
		uint64 word = 0;
		Size remaining = Min(sizeof(uint64), sizeof(identity) - offset);

		memcpy(&word, ((char *) &identity) + offset, remaining);
		hash = PgturbohybridGraphHashU64(hash, word);
	}
	hash = PgturbohybridGraphHashU64(hash, meta->dimensions);
	hash = PgturbohybridGraphHashU64(hash, meta->m);
	hash = PgturbohybridGraphHashU64(hash, meta->graphMaxLevel);
	hash = PgturbohybridGraphHashU64(hash, meta->graphFlags);
	hash = PgturbohybridGraphHashU64(hash, meta->tqNodeCount);
	hash = PgturbohybridGraphHashU64(hash, meta->tqEntryNodeId);
	hash = PgturbohybridGraphHashU64(hash, meta->tqCodeBytes);
	hash = PgturbohybridGraphHashU64(hash, meta->tqBits);
	hash = PgturbohybridGraphHashU64(hash, meta->tqPayloadCount);
	hash = PgturbohybridGraphHashU64(hash, meta->tqPayloadBytes);
	hash = PgturbohybridGraphHashU64(hash, meta->tqResidualRerankBytes);
	hash = PgturbohybridGraphHashU64(hash, meta->tqCodeStartBlkno);
	hash = PgturbohybridGraphHashU64(hash, meta->tqAdjStartBlkno);
	hash = PgturbohybridGraphHashU64(hash, meta->tqExactStartBlkno);
	hash = PgturbohybridGraphHashU64(hash, meta->tqCorrectionStartBlkno);
	return hash;
}

static void
PgturbohybridGraphSharedCacheDir(char *dir, Size dirSize)
{
	snprintf(dir, dirSize, "%s/pg_turbohybrid_cache", DataDir);
}

static bool
PgturbohybridGraphEnsureSharedCacheDir(Relation index, char *dir, Size dirSize)
{
	char root[MAXPGPATH];
	char databaseDir[MAXPGPATH];
	Oid tablespaceOid;
	Oid relNumber;
	int written;

	PgturbohybridGraphSharedCacheDir(root, sizeof(root));
	if (mkdir(root, 0700) != 0 && errno != EEXIST)
		goto fail;
	written = snprintf(databaseDir, sizeof(databaseDir), "%s/%u", root, MyDatabaseId);
	if (written < 0 || written >= (int) sizeof(databaseDir))
		return false;
	if (mkdir(databaseDir, 0700) != 0 && errno != EEXIST)
		goto fail;
	PgturbohybridGraphGetRelationLocator(index, &tablespaceOid, &relNumber);
	written = snprintf(dir, dirSize, "%s/%u", databaseDir, tablespaceOid);
	if (written < 0 || written >= (int) dirSize)
		return false;
	if (mkdir(dir, 0700) == 0 || errno == EEXIST)
		return true;
fail:
	elog(WARNING, "could not create pgturbohybrid shared cache directory: %m");
	return false;
}

static void
PgturbohybridGraphSharedCachePath(Relation index, PgturbohybridGraphMetaPageData *meta,
								  uint64 key, char *path, Size pathSize)
{
	char		dir[MAXPGPATH];
	Oid		tablespaceOid;
	Oid		relNumber;
	int		written;

	PgturbohybridGraphGetRelationLocator(index, &tablespaceOid, &relNumber);
	PgturbohybridGraphSharedCacheDir(dir, sizeof(dir));
	written = snprintf(path, pathSize, "%s/%u/%u/%u_%u_%llu_%u_%016llx.tqcache",
			 dir,
			 MyDatabaseId, tablespaceOid, relNumber, MAIN_FORKNUM,
			 (unsigned long long) meta->graphGeneration,
			 meta->version,
			 (unsigned long long) key);
	if (written < 0 || written >= (int) pathSize)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid shared cache path exceeds MAXPGPATH")));
}

static void
PgturbohybridGraphSharedCacheLockPath(Relation index, PgturbohybridGraphMetaPageData *meta,
									  uint64 key, char *path, Size pathSize)
{
	char		dir[MAXPGPATH];
	Oid			tablespaceOid;
	Oid			relNumber;
	int			written;

	(void) meta;
	(void) key;
	PgturbohybridGraphGetRelationLocator(index, &tablespaceOid, &relNumber);
	(void) relNumber;
	PgturbohybridGraphSharedCacheDir(dir, sizeof(dir));
	written = snprintf(path, pathSize, "%s/%u/%u/%u_%u.lock", dir,
					   MyDatabaseId, tablespaceOid, RelationGetRelid(index),
					   MAIN_FORKNUM);
	if (written < 0 || written >= (int) pathSize)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid shared cache lock path exceeds MAXPGPATH")));
}

#ifndef WIN32
static bool
PgturbohybridGraphSharedTryLock(int fd)
{
	struct flock lock;

	memset(&lock, 0, sizeof(lock));
	lock.l_type = F_WRLCK;
	lock.l_whence = SEEK_SET;
	lock.l_start = 0;
	lock.l_len = 1;
	return fcntl(fd, F_SETLK, &lock) == 0;
}

/*
 * TAP-only synchronization for crash testing.  This deliberately has no GUC
 * or SQL surface: normal servers do not set the environment variable and pay
 * only one getenv().  Fixed stage names prevent an environment value from
 * selecting arbitrary files outside the test-owned directory.
 */
static void
PgturbohybridGraphSharedTestPause(const char *stage)
{
	const char *dir = getenv("PGTURBOHYBRID_TEST_SHARED_CACHE_STAGE_DIR");
	char		requestPath[MAXPGPATH];
	char		reachedPath[MAXPGPATH];
	int			fd;

	if (dir == NULL || dir[0] == '\0')
		return;
	if (strcmp(stage, "after_lock") != 0 &&
		strcmp(stage, "after_ftruncate") != 0 &&
		strcmp(stage, "after_mmap") != 0 &&
		strcmp(stage, "before_fsync") != 0 &&
		strcmp(stage, "before_rename") != 0)
		elog(ERROR, "invalid shared-cache test stage");
	if (snprintf(requestPath, sizeof(requestPath), "%s/%s.request", dir,
				 stage) >= (int) sizeof(requestPath) ||
		snprintf(reachedPath, sizeof(reachedPath), "%s/%s.reached", dir,
				 stage) >= (int) sizeof(reachedPath))
		elog(ERROR, "shared-cache test stage path exceeds MAXPGPATH");
	if (access(requestPath, F_OK) != 0)
		return;

	fd = open(reachedPath, O_CREAT | O_EXCL | O_WRONLY, 0600);
	if (fd >= 0)
	{
		char		pidText[32];
		int			length = snprintf(pidText, sizeof(pidText), "%d\n", MyProcPid);

		if (length > 0)
			(void) write(fd, pidText, (size_t) length);
		close(fd);
	}
	while (access(requestPath, F_OK) == 0)
	{
		CHECK_FOR_INTERRUPTS();
		pg_usleep(PGTURBOHYBRID_GRAPH_SHARED_CACHE_POLL_US);
	}
}

static void
PgturbohybridGraphSharedUnlock(int fd)
{
	struct flock lock;

	memset(&lock, 0, sizeof(lock));
	lock.l_type = F_UNLCK;
	lock.l_whence = SEEK_SET;
	lock.l_start = 0;
	lock.l_len = 1;
	(void) fcntl(fd, F_SETLK, &lock);
}

static bool
PgturbohybridGraphSharedHasSuffix(const char *name, const char *suffix)
{
	Size nameLen = strlen(name);
	Size suffixLen = strlen(suffix);

	return nameLen >= suffixLen &&
		strcmp(name + nameLen - suffixLen, suffix) == 0;
}

static bool
PgturbohybridGraphSharedParseOid(const char *name, Oid *oid)
{
	char	   *end = NULL;
	unsigned long value;

	if (*name == '\0')
		return false;
	errno = 0;
	value = strtoul(name, &end, 10);
	if (errno != 0 || end == name || *end != '\0' || value > UINT_MAX)
		return false;
	*oid = (Oid) value;
	return true;
}

static bool
PgturbohybridGraphSharedReadHeader(const char *path,
									 PgturbohybridGraphSharedCacheHeader *hdr)
{
	int			fd;
	ssize_t		readBytes;

	fd = open(path, O_RDONLY);
	if (fd < 0)
		return false;
	readBytes = read(fd, hdr, sizeof(*hdr));
	close(fd);
	return readBytes == (ssize_t) sizeof(*hdr) &&
		hdr->magic == PGTURBOHYBRID_GRAPH_SHARED_CACHE_MAGIC &&
		hdr->version == PGTURBOHYBRID_GRAPH_SHARED_CACHE_VERSION;
}

static bool
PgturbohybridGraphSharedRelationIdentityExists(
	const PgturbohybridGraphCacheIdentity *identity)
{
	HeapTuple	tuple;
	Form_pg_class classForm;
	Oid			tablespaceOid;
	bool		matches;

	if (identity->databaseOid != MyDatabaseId)
		return false;
	tuple = SearchSysCache1(RELOID, ObjectIdGetDatum(identity->relationOid));
	if (!HeapTupleIsValid(tuple))
		return false;
	classForm = (Form_pg_class) GETSTRUCT(tuple);
	tablespaceOid = classForm->reltablespace != InvalidOid ?
		classForm->reltablespace : MyDatabaseTableSpace;
	matches = tablespaceOid == identity->tablespaceOid &&
		classForm->relfilenode == identity->relNumber;
	ReleaseSysCache(tuple);
	return matches;
}

static int64
PgturbohybridGraphSharedCleanDirectory(const char *dir, bool databaseExists)
{
	DIR		   *directory;
	struct dirent *entry;
	int64		removed = 0;
	time_t		now = time(NULL);

	directory = opendir(dir);
	if (directory == NULL)
		return 0;
	while ((entry = readdir(directory)) != NULL)
	{
		char		path[MAXPGPATH];
		int			written;
		bool		removeFile = false;

		if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
			continue;
		written = snprintf(path, sizeof(path), "%s/%s", dir, entry->d_name);
		if (written < 0 || written >= (int) sizeof(path))
			continue;
		if (PgturbohybridGraphSharedHasSuffix(entry->d_name, ".tqcache"))
		{
			PgturbohybridGraphSharedCacheHeader hdr;

			removeFile = !databaseExists ||
				!PgturbohybridGraphSharedReadHeader(path, &hdr) ||
				!PgturbohybridGraphSharedRelationIdentityExists(&hdr.identity);
		}
		else if (strstr(entry->d_name, ".tqcache.tmp.") != NULL)
		{
			struct stat st;

			removeFile = !databaseExists ||
				(stat(path, &st) == 0 && now - st.st_mtime >=
				 PGTURBOHYBRID_GRAPH_SHARED_CACHE_TEMP_STALE_SECONDS);
		}
		else if (!databaseExists &&
				 (PgturbohybridGraphSharedHasSuffix(entry->d_name, ".lock")))
			removeFile = true;
		else if (databaseExists &&
				 PgturbohybridGraphSharedHasSuffix(entry->d_name, ".lock"))
		{
			unsigned int relationOid;
			unsigned int forkNumber;
			int consumed = 0;

			if (sscanf(entry->d_name, "%u_%u.lock%n", &relationOid,
					   &forkNumber, &consumed) == 2 &&
				entry->d_name[consumed] == '\0' &&
				forkNumber == MAIN_FORKNUM &&
				!SearchSysCacheExists1(RELOID, ObjectIdGetDatum((Oid) relationOid)))
				removeFile = true;
		}

		if (removeFile && unlink(path) == 0)
			removed++;
	}
	closedir(directory);
	return removed;
}

static int64
PgturbohybridGraphSharedRunGc(void)
{
	char		root[MAXPGPATH];
	char		gcPath[MAXPGPATH];
	DIR		   *rootDir;
	struct dirent *databaseEntry;
	TimestampTz now = GetCurrentTimestamp();
	int			gcFd;
	int64		removed = 0;

	if (pgturbohybridGraphSharedLastGc != 0 &&
		!TimestampDifferenceExceeds(pgturbohybridGraphSharedLastGc, now,
									PGTURBOHYBRID_GRAPH_SHARED_CACHE_GC_INTERVAL_MS))
		return 0;
	PgturbohybridGraphSharedCacheDir(root, sizeof(root));
	if (snprintf(gcPath, sizeof(gcPath), "%s/.gc.lock", root) >=
		(int) sizeof(gcPath))
		return 0;
	gcFd = open(gcPath, O_CREAT | O_RDWR, 0600);
	if (gcFd < 0 || !PgturbohybridGraphSharedTryLock(gcFd))
	{
		if (gcFd >= 0)
			close(gcFd);
		return 0;
	}
	pgturbohybridGraphSharedLastGc = now;
	rootDir = opendir(root);
	if (rootDir != NULL)
	{
		while ((databaseEntry = readdir(rootDir)) != NULL)
		{
			Oid			databaseOid;
			char		databaseDir[MAXPGPATH];
			DIR		   *tablespaceDir;
			struct dirent *tablespaceEntry;
			bool		databaseExists;

			if (!PgturbohybridGraphSharedParseOid(databaseEntry->d_name,
												 &databaseOid))
				continue;
			if (snprintf(databaseDir, sizeof(databaseDir), "%s/%s", root,
						 databaseEntry->d_name) >= (int) sizeof(databaseDir))
				continue;
			databaseExists = SearchSysCacheExists1(DATABASEOID,
												ObjectIdGetDatum(databaseOid));
			tablespaceDir = opendir(databaseDir);
			if (tablespaceDir == NULL)
				continue;
			while ((tablespaceEntry = readdir(tablespaceDir)) != NULL)
			{
				Oid			tablespaceOid;
				char		dir[MAXPGPATH];

				if (!PgturbohybridGraphSharedParseOid(tablespaceEntry->d_name,
													 &tablespaceOid))
					continue;
				if (snprintf(dir, sizeof(dir), "%s/%s", databaseDir,
							 tablespaceEntry->d_name) >= (int) sizeof(dir))
					continue;
				if (databaseExists && databaseOid != MyDatabaseId)
					continue;
				removed += PgturbohybridGraphSharedCleanDirectory(dir,
														 databaseExists);
				if (!databaseExists)
					(void) rmdir(dir);
			}
			closedir(tablespaceDir);
			if (!databaseExists)
				(void) rmdir(databaseDir);
		}
		closedir(rootDir);
	}
	PgturbohybridGraphSharedUnlock(gcFd);
	close(gcFd);
	return removed;
}

static int64
PgturbohybridGraphSharedRemoveOldGenerations(Relation index,
											 PgturbohybridGraphMetaPageData *meta,
											 const char *keepPath)
{
	char		dir[MAXPGPATH];
	DIR		   *directory;
	struct dirent *entry;
	Oid			tablespaceOid;
	Oid			relNumber;
	char		prefix[64];
	int64		removed = 0;

	(void) meta;
	if (!PgturbohybridGraphEnsureSharedCacheDir(index, dir, sizeof(dir)))
		return 0;
	PgturbohybridGraphGetRelationLocator(index, &tablespaceOid, &relNumber);
	(void) tablespaceOid;
	if (snprintf(prefix, sizeof(prefix), "%u_%u_", relNumber, MAIN_FORKNUM) >=
		(int) sizeof(prefix))
		return 0;
	directory = opendir(dir);
	if (directory == NULL)
		return 0;
	while ((entry = readdir(directory)) != NULL)
	{
		unsigned int parsedRel;
		unsigned int parsedFork;
		unsigned int parsedFormat;
		unsigned long long parsedGeneration;
		unsigned long long parsedKey;
		int consumed = 0;
		char path[MAXPGPATH];

		if (strncmp(entry->d_name, prefix, strlen(prefix)) == 0 &&
			strstr(entry->d_name, ".tqcache.tmp.") != NULL)
		{
			if (snprintf(path, sizeof(path), "%s/%s", dir, entry->d_name) <
				(int) sizeof(path) && unlink(path) == 0)
				removed++;
			continue;
		}
		if (sscanf(entry->d_name, "%u_%u_%llu_%u_%llx.tqcache%n",
				   &parsedRel, &parsedFork, &parsedGeneration, &parsedFormat,
				   &parsedKey, &consumed) != 5 ||
			entry->d_name[consumed] != '\0' || parsedRel != relNumber ||
			parsedFork != MAIN_FORKNUM)
			continue;
		if (snprintf(path, sizeof(path), "%s/%s", dir, entry->d_name) >=
			(int) sizeof(path) || strcmp(path, keepPath) == 0)
			continue;
		if (unlink(path) == 0)
			removed++;
	}
	closedir(directory);
	return removed;
}
#endif

static bool
PgturbohybridGraphSharedHeaderMatches(PgturbohybridGraphSharedCacheHeader *hdr,
									  Relation index,
									  PgturbohybridGraphMetaPageData *meta,
									  uint64 key)
{
	PgturbohybridGraphCacheIdentity identity;

	PgturbohybridGraphCacheIdentityInit(index, meta, &identity);
	return hdr->magic == PGTURBOHYBRID_GRAPH_SHARED_CACHE_MAGIC &&
		hdr->version == PGTURBOHYBRID_GRAPH_SHARED_CACHE_VERSION &&
		hdr->key == key &&
		PgturbohybridGraphCacheIdentityEqual(&hdr->identity, &identity) &&
		hdr->dimensions == meta->dimensions &&
		hdr->m == meta->m &&
		hdr->graphMaxLevel == meta->graphMaxLevel &&
		hdr->graphFlags == meta->graphFlags &&
		hdr->tqNodeCount == meta->tqNodeCount &&
		hdr->tqEntryNodeId == meta->tqEntryNodeId &&
		hdr->tqCodeBytes == meta->tqCodeBytes &&
		hdr->tqBits == meta->tqBits &&
		hdr->tqPayloadCount == meta->tqPayloadCount &&
		hdr->tqPayloadBytes == meta->tqPayloadBytes &&
		hdr->tqResidualRerankBytes == meta->tqResidualRerankBytes &&
		hdr->tqCodeStartBlkno == meta->tqCodeStartBlkno &&
		hdr->tqAdjStartBlkno == meta->tqAdjStartBlkno &&
		hdr->tqExactStartBlkno == meta->tqExactStartBlkno &&
		hdr->tqCorrectionStartBlkno == meta->tqCorrectionStartBlkno;
}

static bool
PgturbohybridGraphSharedAlign(uint64 offset, uint64 *aligned)
{
	if (offset > UINT64_MAX - (MAXIMUM_ALIGNOF - 1))
		return false;
	*aligned = (offset + (MAXIMUM_ALIGNOF - 1)) & ~((uint64) MAXIMUM_ALIGNOF - 1);
	return true;
}

static bool
PgturbohybridGraphSharedAddBytes(uint64 *offset, uint64 bytes,
								PgturbohybridGraphSharedSegment *segment)
{
	uint64 aligned;

	if (bytes == 0)
	{
		segment->offset = 0;
		segment->length = 0;
		return true;
	}
	if (!PgturbohybridGraphSharedAlign(*offset, &aligned))
		return false;
	*offset = aligned;
	segment->offset = *offset;
	segment->length = bytes;
	if (bytes > UINT64_MAX - *offset)
		return false;
	*offset += bytes;
	return true;
}

static bool
PgturbohybridGraphBuildSharedLayout(PgturbohybridGraphMetaPageData *meta,
									uint64 neighborValueCount,
									uint32 payloadRefCount,
									uint32 exactBytes,
									bool enforceCacheLimit,
									PgturbohybridGraphSharedCacheHeader *hdr)
{
	uint64 offset = sizeof(*hdr);
	uint64 bytes;
	uint64 adjRecordCount = PgturbohybridGraphAdjRecordCount(meta);

#define ADD_ARRAY(segment, count, width) \
	do { \
		if (!PgturbohybridGraphCheckedMultiplyU64((count), (width), &bytes) || \
			!PgturbohybridGraphSharedAddBytes(&offset, bytes, &(segment))) \
			return false; \
	} while (0)

	ADD_ARRAY(hdr->nodes, meta->tqNodeCount, sizeof(PgturbohybridGraphSharedNode));
	ADD_ARRAY(hdr->codeArena, meta->tqNodeCount, meta->tqCodeBytes);
	ADD_ARRAY(hdr->payloadArena, meta->tqNodeCount, meta->tqPayloadBytes);
	ADD_ARRAY(hdr->residualArena, meta->tqNodeCount, meta->tqResidualRerankBytes);
	ADD_ARRAY(hdr->exactArena, meta->tqNodeCount, exactBytes);
	ADD_ARRAY(hdr->neighborCounts, adjRecordCount, sizeof(uint16));
	ADD_ARRAY(hdr->neighborOffsets, adjRecordCount, sizeof(uint64));
	ADD_ARRAY(hdr->neighborData, neighborValueCount, sizeof(uint32));
	ADD_ARRAY(hdr->payloadRefs, payloadRefCount,
			  sizeof(PgturbohybridGraphPayloadRef));
#undef ADD_ARRAY
	if (!PgturbohybridGraphSharedAlign(offset, &hdr->fileSize) ||
		(enforceCacheLimit &&
		 hdr->fileSize > PgturbohybridGraphNativeCacheMaxBytes()) ||
		hdr->fileSize > (uint64) SIZE_MAX)
		return false;
	return true;
}

/*
 * Produce the conservative layout used by admission and diagnostics.  The
 * writer feeds the same calculator the exact counts it observed while loading
 * the graph; admission must reserve for every legal edge and payload ref so it
 * can never admit a graph that the writer later cannot represent.
 */
static bool
PgturbohybridGraphBuildMaximumSharedLayout(PgturbohybridGraphMetaPageData *meta,
										bool cacheExactVectors,
										bool enforceCacheLimit,
										PgturbohybridGraphSharedCacheHeader *hdr)
{
	uint64		neighborCapacityPerNode = 0;
	uint64		neighborValueCount;
	uint64		payloadRefCount;
	uint64		exactTotal;
	uint32		exactBytes = 0;
	int			levelCapacity = PgturbohybridGraphLevelCapacity(meta->m);

	for (int level = 0; level < levelCapacity; level++)
	{
		uint64 levelM = PgturbohybridGraphLevelM(meta->m, level);

		if (levelM > UINT64_MAX - neighborCapacityPerNode)
			return false;
		neighborCapacityPerNode += levelM;
	}
	if (!PgturbohybridGraphCheckedMultiplyU64(meta->tqNodeCount,
											 neighborCapacityPerNode,
											 &neighborValueCount) ||
		!PgturbohybridGraphCheckedMultiplyU64(meta->tqNodeCount,
											 meta->tqPayloadCount,
											 &payloadRefCount) ||
		payloadRefCount > UINT32_MAX)
		return false;
	if (cacheExactVectors && meta->dimensions > 0)
	{
		Size perNode = PGTURBOHYBRID_VECTOR_SIZE(meta->dimensions);

		if (perNode > UINT32_MAX ||
			!PgturbohybridGraphCheckedMultiplyU64(meta->tqNodeCount,
												 perNode, &exactTotal))
			return false;
		exactBytes = (uint32) perNode;
	}

	memset(hdr, 0, sizeof(*hdr));
	return PgturbohybridGraphBuildSharedLayout(meta, neighborValueCount,
										  (uint32) payloadRefCount,
										  exactBytes, enforceCacheLimit, hdr);
}

static bool
PgturbohybridGraphValidateSharedLayout(void *base,
									  PgturbohybridGraphSharedCacheHeader *hdr,
									  PgturbohybridGraphMetaPageData *meta,
									  uint64 fileSize)
{
	PgturbohybridGraphSharedSegment *segments[] = {
		&hdr->nodes, &hdr->codeArena, &hdr->payloadArena,
		&hdr->residualArena, &hdr->exactArena, &hdr->neighborCounts,
		&hdr->neighborOffsets, &hdr->neighborData, &hdr->payloadRefs
	};
	PgturbohybridGraphSharedCacheHeader expected;

	if (hdr->fileSize != fileSize || fileSize < sizeof(*hdr) ||
		hdr->adjRecordCount != PgturbohybridGraphAdjRecordCount(meta) ||
		hdr->payloadRefCount > meta->tqNodeCount * (uint64) meta->tqPayloadCount ||
		hdr->neighborValueCount >
			(uint64) hdr->adjRecordCount * PgturbohybridGraphLevelM(meta->m, 0))
		return false;
	memset(&expected, 0, sizeof(expected));
	if (!PgturbohybridGraphBuildSharedLayout(meta, hdr->neighborValueCount,
										 hdr->payloadRefCount, hdr->exactBytes,
										 true,
										 &expected) ||
		expected.fileSize != hdr->fileSize)
		return false;
	for (Size i = 0; i < lengthof(segments); i++)
	{
		PgturbohybridGraphSharedSegment *segment = segments[i];

		if ((segment->length == 0) != (segment->offset == 0) ||
			(segment->length > 0 &&
			 (segment->offset % MAXIMUM_ALIGNOF != 0 ||
			  segment->offset < sizeof(*hdr) ||
			  segment->offset > fileSize ||
			  segment->length > fileSize - segment->offset)))
			return false;
		for (Size j = 0; j < i; j++)
		{
			PgturbohybridGraphSharedSegment *other = segments[j];

			if (segment->length > 0 && other->length > 0 &&
				segment->offset < other->offset + other->length &&
				other->offset < segment->offset + segment->length)
				return false;
		}
	}
	if (memcmp(&hdr->nodes, &expected.nodes,
			   sizeof(PgturbohybridGraphSharedSegment) * lengthof(segments)) != 0)
		return false;
	if (hdr->adjRecordCount > 0)
	{
		uint16 *counts = (uint16 *) PgturbohybridGraphSharedCheckedPtr(
			base, fileSize, hdr->neighborCounts.offset,
			hdr->neighborCounts.length);
		uint64 *offsets = (uint64 *) PgturbohybridGraphSharedCheckedPtr(
			base, fileSize, hdr->neighborOffsets.offset,
			hdr->neighborOffsets.length);

		if (counts == NULL || offsets == NULL)
			return false;

		for (uint32 slot = 0; slot < hdr->adjRecordCount; slot++)
		{
			if (offsets[slot] > hdr->neighborValueCount ||
				counts[slot] > hdr->neighborValueCount - offsets[slot])
				return false;
		}
	}
	if (hdr->payloadRefCount > 0)
	{
		PgturbohybridGraphPayloadRef *refs =
			(PgturbohybridGraphPayloadRef *) PgturbohybridGraphSharedCheckedPtr(
				base, fileSize, hdr->payloadRefs.offset,
				hdr->payloadRefs.length);

		if (refs == NULL)
			return false;

		for (uint32 i = 0; i < hdr->payloadRefCount; i++)
			if (refs[i].nodeId >= meta->tqNodeCount ||
				refs[i].payloadSlot >= meta->tqPayloadCount)
				return false;
	}
	return true;
}

static bool
PgturbohybridGraphShouldUseNativeCacheWithPolicy(PgturbohybridGraphMetaPageData *meta,
									   bool cacheExactVectors,
									   int policy,
									   PgturbohybridGraphNativeCacheReason *reason)
{
	uint64		totalBytes;
	Size		bytes;
	PgturbohybridGraphSharedCacheHeader layout;

	if (reason != NULL)
		*reason = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_NONE;

	if (policy == PGTURBOHYBRID_NATIVE_CACHE_POLICY_OFF)
	{
		if (reason != NULL)
			*reason = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_POLICY_OFF;
		return false;
	}

	if (!PgturbohybridGraphBuildMaximumSharedLayout(meta, cacheExactVectors,
												 true, &layout))
		goto too_large;
	totalBytes = layout.fileSize;
	if (BlockNumberIsValid(meta->tqMultivectorDocMapStartBlkno) &&
		meta->tqMultivectorDocMapVersion ==
		PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION)
	{
		if (!PgturbohybridGraphArenaBytes(meta->tqNodeCount,
										  sizeof(TqMultiVectorNodeMapEntry),
										  &bytes))
			goto too_large;
		if ((uint64) bytes > UINT64_MAX - totalBytes)
			goto too_large;
		totalBytes += (uint64) bytes;
		if (!PgturbohybridGraphArenaBytes(meta->tqMultivectorDocCount,
										  sizeof(TqMultiVectorDocMapEntry),
										  &bytes))
			goto too_large;
		if ((uint64) bytes > UINT64_MAX - totalBytes)
			goto too_large;
		totalBytes += (uint64) bytes;
	}

	if (totalBytes <= PgturbohybridGraphNativeCacheMaxBytes())
	{
		if (reason != NULL)
		{
			if (policy == PGTURBOHYBRID_NATIVE_CACHE_POLICY_SHARED)
				*reason = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_SHARED_FITS_MAX_MB;
			else if (policy == PGTURBOHYBRID_NATIVE_CACHE_POLICY_PER_BACKEND)
				*reason = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_PER_BACKEND_FITS_MAX_MB;
			else
				*reason = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_AUTO_FITS_MAX_MB;
		}
		return true;
	}

too_large:
	if (reason != NULL)
		*reason = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_EXCEEDS_MAX_MB;
	return false;
}

static bool
PgturbohybridGraphShouldUseNativeCache(PgturbohybridGraphMetaPageData *meta,
									   bool cacheExactVectors,
									   PgturbohybridGraphNativeCacheReason *reason)
{
	return PgturbohybridGraphShouldUseNativeCacheWithPolicy(meta, cacheExactVectors,
															pgturbohybrid_native_cache_policy,
															reason);
}

bool
PgturbohybridGraphEstimateMemory(Relation index,
								 PgturbohybridGraphMetaPageData *meta,
								 PgturbohybridGraphMemoryEstimate *estimate)
{
	uint64		totalBytes = 0;
	uint64		adjRecordCount;
	uint64		neighborCapacityPerNode = 0;
	int			codeTuplesPerPage;
	int			levelCapacity;
	Size		exactBytesPerNode;
	PgturbohybridGraphSharedCacheHeader sharedLayout;

	memset(estimate, 0, sizeof(*estimate));
	if (meta == NULL || !PgturbohybridGraphReadMeta(index, meta))
		return false;

	estimate->available = true;
	estimate->adjacencyEstimated = true;
	estimate->cacheExactVectors = PgturbohybridGraphShouldCacheExactVectors(index, meta);
	estimate->cachePolicy = pgturbohybrid_native_cache_policy;
	estimate->effectiveCachePolicy = PgturbohybridGraphEffectiveScanNativeCachePolicy();
	(void) PgturbohybridGraphShouldUseNativeCache(meta,
												  estimate->cacheExactVectors,
												  &estimate->cacheReason);
	estimate->nodeCount = meta->tqNodeCount;
	estimate->dimensions = meta->dimensions;
	estimate->quantizationBits = meta->tqBits != 0 ? meta->tqBits :
		PGTURBOHYBRID_DEFAULT_BITS;

	estimate->codeBytes =
		(uint64) meta->tqNodeCount * (uint64) meta->tqCodeBytes;
	estimate->payloadBytes =
		(uint64) meta->tqNodeCount * (uint64) meta->tqPayloadBytes;
	estimate->residualBytes =
		(uint64) meta->tqNodeCount * (uint64) meta->tqResidualRerankBytes;
	if (BlockNumberIsValid(meta->tqMultivectorDocMapStartBlkno) &&
		meta->tqMultivectorDocMapVersion ==
		PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION)
		estimate->multivectorDocMapBytes =
			(uint64) meta->tqNodeCount *
			(uint64) sizeof(TqMultiVectorNodeMapEntry) +
			(uint64) meta->tqMultivectorDocCount *
			(uint64) sizeof(TqMultiVectorDocMapEntry);
	estimate->nodeBytes =
		(uint64) meta->tqNodeCount * (uint64) sizeof(PgturbohybridGraphScanNode);
	if (meta->tqNodeCount > 0)
		estimate->visitedGenerationBytes =
			(uint64) meta->tqNodeCount * (uint64) sizeof(uint32);

	exactBytesPerNode = PGTURBOHYBRID_VECTOR_SIZE(meta->dimensions);
	if (estimate->cacheExactVectors &&
		BlockNumberIsValid(meta->tqExactStartBlkno) &&
		meta->tqNodeCount > 0 &&
		exactBytesPerNode > 0 &&
		(uint64) meta->tqNodeCount <= UINT64_MAX / (uint64) exactBytesPerNode &&
		(uint64) meta->tqNodeCount * (uint64) exactBytesPerNode <=
		PGTURBOHYBRID_GRAPH_EXACT_CACHE_AUTO_MAX_BYTES)
		estimate->exactBytes =
			(uint64) meta->tqNodeCount * (uint64) exactBytesPerNode;

	levelCapacity = PgturbohybridGraphLevelCapacity(meta->m);
	estimate->levelCapacity = (uint16) levelCapacity;
	adjRecordCount = (uint64) PgturbohybridGraphAdjRecordCount(meta);
	for (int level = 0; level < levelCapacity; level++)
		neighborCapacityPerNode +=
			(uint64) PgturbohybridGraphLevelM(meta->m, level);
	estimate->adjacencyBytes =
		adjRecordCount *
		((uint64) sizeof(uint32 *) + (uint64) sizeof(uint16) +
		 (uint64) sizeof(BlockNumber) + (uint64) sizeof(OffsetNumber)) +
		(uint64) meta->tqNodeCount * neighborCapacityPerNode *
		(uint64) sizeof(uint32);

	codeTuplesPerPage =
		PgturbohybridGraphTuplesPerPage(PgturbohybridGraphCodeTupleSize(meta->dimensions,
											 meta->tqPayloadCount,
											 meta->tqBits,
											 (meta->tqFlags & PGTURBOHYBRID_GRAPH_TQ_WEIGHTED) != 0,
											 meta->tqResidualRerankBytes));
	estimate->codePageCount =
		(uint32) PgturbohybridGraphPageCount(meta->tqNodeCount,
											 codeTuplesPerPage);
	estimate->pageMapBytes =
		(uint64) estimate->codePageCount *
		((uint64) sizeof(bool) + (uint64) sizeof(BlockNumber));

	estimate->sharedBackendViewBytes =
		estimate->nodeBytes +
		adjRecordCount * (uint64) sizeof(uint32 *) +
		estimate->visitedGenerationBytes +
		(meta->tqNodeCount > 0 ? (uint64) sizeof(uint32) : 0);

	/* Keep the persisted shared portion byte-for-byte aligned with admission. */
	if (!PgturbohybridGraphBuildMaximumSharedLayout(meta,
												 estimate->cacheExactVectors,
												 false, &sharedLayout))
		return false;
	totalBytes = sharedLayout.fileSize;
	totalBytes += estimate->visitedGenerationBytes;
	totalBytes += estimate->multivectorDocMapBytes;
	totalBytes += estimate->pageMapBytes;
	estimate->estimatedTotalBytes = totalBytes;
	return true;
}

static void
PgturbohybridGraphInitScanStorageUncached(PgturbohybridGraphMetaPageData *meta, PgturbohybridGraphScanStorage *storage,
							   bool cacheExactVectors)
{
	Size		arenaBytes;
	int			adjRecordCount;

	memset(storage, 0, sizeof(PgturbohybridGraphScanStorage));
	storage->ctx = CurrentMemoryContext;
	storage->nodes =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridGraphScanNode),
											   meta->tqNodeCount,
											   "pgturbohybrid graph scan node cache"));
	if (meta->tqNodeCount > 0 && meta->tqCodeBytes > 0 &&
		PgturbohybridGraphArenaBytesHuge(meta->tqNodeCount, meta->tqCodeBytes,
										 &arenaBytes) &&
		arenaBytes <= PgturbohybridGraphNativeCacheMaxBytes())
		storage->codeArena = palloc_extended(arenaBytes, MCXT_ALLOC_HUGE | MCXT_ALLOC_ZERO);
	if (meta->tqNodeCount > 0 && meta->tqResidualRerankBytes > 0 &&
		PgturbohybridGraphArenaBytesHuge(meta->tqNodeCount, meta->tqResidualRerankBytes,
										 &arenaBytes) &&
		arenaBytes <= PgturbohybridGraphNativeCacheMaxBytes())
		storage->residualArena = palloc_extended(arenaBytes, MCXT_ALLOC_HUGE | MCXT_ALLOC_ZERO);
	if (meta->tqNodeCount > 0 && meta->tqPayloadBytes > 0 &&
		PgturbohybridGraphArenaBytesHuge(meta->tqNodeCount, meta->tqPayloadBytes,
										 &arenaBytes) &&
		arenaBytes <= PgturbohybridGraphNativeCacheMaxBytes())
		storage->payloadArena = palloc_extended(arenaBytes, MCXT_ALLOC_HUGE | MCXT_ALLOC_ZERO);
	if (cacheExactVectors && meta->tqNodeCount > 0 && meta->dimensions > 0)
	{
		storage->exactBytes = PGTURBOHYBRID_VECTOR_SIZE(meta->dimensions);
		if (PgturbohybridGraphArenaBytes(meta->tqNodeCount, storage->exactBytes,
										 &arenaBytes) &&
			arenaBytes <= PGTURBOHYBRID_GRAPH_EXACT_CACHE_AUTO_MAX_BYTES)
			storage->exactArena = palloc0(arenaBytes);
	}
	storage->levelCount = PgturbohybridGraphLevelCapacity(meta->m);
	adjRecordCount = PgturbohybridGraphAdjRecordCount(meta);
	storage->neighbors =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32 *),
											   adjRecordCount,
											   "pgturbohybrid graph neighbor cache"));
	storage->neighborCounts =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint16),
											   adjRecordCount,
											   "pgturbohybrid graph neighbor count cache"));
	storage->codeTuplesPerPage =
		PgturbohybridGraphTuplesPerPage(PgturbohybridGraphCodeTupleSize(meta->dimensions,
												  meta->tqPayloadCount,
												  meta->tqBits,
												  (meta->tqFlags & PGTURBOHYBRID_GRAPH_TQ_WEIGHTED) != 0,
												  meta->tqResidualRerankBytes));
	storage->codePageCount = PgturbohybridGraphPageCount(meta->tqNodeCount, storage->codeTuplesPerPage);
	storage->adjPageCount = BlockNumberIsValid(meta->tqAdjStartBlkno) ? 1 : 0;
	storage->codePagesLoaded =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(bool),
											   storage->codePageCount,
											   "pgturbohybrid graph code page cache"));
	if (storage->adjPageCount > 0)
		storage->adjPagesLoaded =
			palloc0(PgturbohybridCheckedArrayBytes(sizeof(bool),
												   storage->adjPageCount,
												   "pgturbohybrid graph adjacency page cache"));
	storage->codeBlknos =
		palloc(PgturbohybridCheckedArrayBytes(sizeof(BlockNumber),
											  storage->codePageCount,
											  "pgturbohybrid graph code block map"));
	PgturbohybridGraphInitBlockMap(storage->codeBlknos, storage->codePageCount);
	storage->adjBlknos =
		palloc(PgturbohybridCheckedArrayBytes(sizeof(BlockNumber),
											  adjRecordCount,
											  "pgturbohybrid graph adjacency block map"));
	storage->adjOffnos =
		palloc(PgturbohybridCheckedArrayBytes(sizeof(OffsetNumber),
											  adjRecordCount,
											  "pgturbohybrid graph adjacency offset map"));
	for (int i = 0; i < adjRecordCount; i++)
	{
		storage->adjBlknos[i] = InvalidBlockNumber;
		storage->adjOffnos[i] = InvalidOffsetNumber;
	}
}

static void
PgturbohybridGraphMultiVectorDocMapError(Relation index, const char *detail)
{
	ereport(ERROR,
			(errcode(ERRCODE_DATA_CORRUPTED),
			 errmsg("pgturbohybrid multivector docmap sidecar is invalid for index \"%s\"",
					RelationGetRelationName(index)),
			 errdetail_internal("%s", detail),
				 errhint("REINDEX the index to rebuild the multivector docmap sidecar.")));
}

typedef struct PgturbohybridGraphDecodedMultiVectorDocVectorTuple
{
	uint8		storageKind;
	uint16		count;
	uint32		docId;
	uint32		startFloat;
	float		scale;
	const void *values;
	Size		tupleSize;
} PgturbohybridGraphDecodedMultiVectorDocVectorTuple;

static bool
PgturbohybridGraphDecodeMultiVectorDocVectorTuple(Item item, Size itemSize,
												  PgturbohybridGraphDecodedMultiVectorDocVectorTuple *decoded)
{
	uint8		type;
	PgturbohybridGraphMultiVectorDocMapVectorTuple common;

	if (item == NULL || decoded == NULL ||
		itemSize < offsetof(PgturbohybridGraphMultiVectorDocMapVectorTupleData,
							values))
		return false;

	type = *((uint8 *) item);
	common = (PgturbohybridGraphMultiVectorDocMapVectorTuple) item;
	if (common->magic != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC ||
		common->version != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION ||
		common->count == 0)
		return false;

	memset(decoded, 0, sizeof(*decoded));
	decoded->count = common->count;
	decoded->docId = common->docId;
	decoded->startFloat = common->startFloat;
	decoded->scale = 1.0f;

	if (type == PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VECTOR_TUPLE_TYPE)
	{
		PgturbohybridGraphMultiVectorDocMapVectorTuple tuple =
			(PgturbohybridGraphMultiVectorDocMapVectorTuple) item;

		decoded->storageKind = PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32;
		decoded->tupleSize =
			PgturbohybridGraphMultiVectorDocMapVectorTupleSize(tuple->count);
		decoded->values = tuple->values;
	}
	else if (type == PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VECTOR_F16_TUPLE_TYPE)
	{
		PgturbohybridGraphMultiVectorDocMapVectorF16Tuple tuple =
			(PgturbohybridGraphMultiVectorDocMapVectorF16Tuple) item;

		decoded->storageKind = PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16;
		decoded->tupleSize =
			PgturbohybridGraphMultiVectorDocMapVectorF16TupleSize(tuple->count);
		decoded->values = tuple->values;
	}
	else if (type == PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VECTOR_SQ8_TUPLE_TYPE)
	{
		PgturbohybridGraphMultiVectorDocMapVectorSq8Tuple tuple =
			(PgturbohybridGraphMultiVectorDocMapVectorSq8Tuple) item;

		decoded->storageKind = PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8;
		decoded->tupleSize =
			PgturbohybridGraphMultiVectorDocMapVectorSq8TupleSize(tuple->count);
		decoded->scale = tuple->scale;
		decoded->values = tuple->values;
		if (!isfinite(decoded->scale) || decoded->scale <= 0.0f)
			return false;
	}
	else
		return false;

	return itemSize >= decoded->tupleSize;
}

static void
PgturbohybridGraphDecodeMultiVectorDocValues(float *dest,
											 uint32 startFloat,
											 uint16 count,
											 uint8 storageKind,
											 const void *values,
											 float scale)
{
	if (storageKind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32)
		memcpy(dest + startFloat, values, sizeof(float) * count);
	else if (storageKind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16)
	{
		const uint16 *halfValues = (const uint16 *) values;

		for (uint16 i = 0; i < count; i++)
			dest[startFloat + i] = PgturbohybridGraphHalfToFloat(halfValues[i]);
	}
	else if (storageKind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8)
	{
		const int8 *sq8Values = (const int8 *) values;

		for (uint16 i = 0; i < count; i++)
			dest[startFloat + i] = (float) sq8Values[i] * scale;
	}
	else
		elog(ERROR, "unknown multivector document vector sidecar storage kind %u",
			 storageKind);
}

bool
PgturbohybridGraphLoadMultiVectorDocMap(Relation index,
									 PgturbohybridGraphMetaPageData *meta,
									 PgturbohybridGraphScanStorage *storage,
									 bool require)
{
	return PgturbohybridGraphLoadMultiVectorDocMapWithStats(index, meta,
															storage, require,
															NULL);
}

static void
PgturbohybridGraphRememberMultiVectorDocVectorChunk(Relation index,
													PgturbohybridGraphScanStorage *storage,
													uint32 docId,
													BlockNumber blkno,
													OffsetNumber offno,
													uint32 startFloat,
													uint16 count,
													uint8 storageKind)
{
	uint32		chunkIndex;

	if (storage->multivectorDocVectorChunkCount >=
		storage->multivectorDocVectorChunkCapacity)
	{
		uint32		oldCapacity = storage->multivectorDocVectorChunkCapacity;
		uint32		newCapacity = oldCapacity == 0 ? 1024 : oldCapacity;
		Size		oldBytes;
		Size		newBytes;

		while (newCapacity <= storage->multivectorDocVectorChunkCount)
		{
			if (newCapacity > PG_UINT32_MAX / 2)
				PgturbohybridGraphMultiVectorDocMapError(index,
														 "document vector sidecar has too many chunks");
			newCapacity += newCapacity;
		}
		oldBytes =
			PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridGraphMultiVectorDocVectorChunkRef),
										   oldCapacity,
										   "pgturbohybrid multivector document vector chunk refs");
		newBytes =
			PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridGraphMultiVectorDocVectorChunkRef),
										   newCapacity,
										   "pgturbohybrid multivector document vector chunk refs");
		if (storage->multivectorDocVectorChunks == NULL)
			storage->multivectorDocVectorChunks =
				MemoryContextAllocZero(storage->ctx, newBytes);
		else
		{
			storage->multivectorDocVectorChunks =
				repalloc(storage->multivectorDocVectorChunks, newBytes);
			memset((char *) storage->multivectorDocVectorChunks + oldBytes, 0,
				   newBytes - oldBytes);
		}
		storage->multivectorDocVectorChunkCapacity = newCapacity;
	}

	chunkIndex = storage->multivectorDocVectorChunkCount++;
	if (storage->multivectorDocVectorFirstChunk[docId] == PG_UINT32_MAX)
		storage->multivectorDocVectorFirstChunk[docId] = chunkIndex;
	else if (storage->multivectorDocVectorFirstChunk[docId] +
			 storage->multivectorDocVectorChunkCounts[docId] != chunkIndex)
		PgturbohybridGraphMultiVectorDocMapError(index,
												 "document vector sidecar chunks are not grouped by document");
	storage->multivectorDocVectorChunkCounts[docId]++;
	storage->multivectorDocVectorChunks[chunkIndex].blkno = blkno;
	storage->multivectorDocVectorChunks[chunkIndex].offno = offno;
	storage->multivectorDocVectorChunks[chunkIndex].startFloat = startFloat;
	storage->multivectorDocVectorChunks[chunkIndex].count = count;
	storage->multivectorDocVectorChunks[chunkIndex].storageKind = storageKind;
}

static void
PgturbohybridGraphBuildCentroidDocCodeMap(Relation index,
										  PgturbohybridGraphMetaPageData *meta,
										  PgturbohybridGraphScanStorage *storage,
										  const PgturbohybridGraphMultiVectorCentroidPostingEntry *postings,
										  const uint32 *postingCodewords,
										  uint32 postingCount,
										  uint32 codebookSize)
{
	MemoryContext oldCtx;
	uint32	   *docCodeCounts;
	uint32	   *docCodeWrite;
	uint32	   *docCodeOffsets;
	uint32	   *docCodes;

	if (postings == NULL || postingCodewords == NULL)
		return;

	docCodeCounts =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
											   meta->tqMultivectorDocCount,
											   "pgturbohybrid multivector centroid doc code counts"));
	for (uint32 postingIndex = 0; postingIndex < postingCount; postingIndex++)
	{
		const PgturbohybridGraphMultiVectorCentroidPostingEntry *entry =
			&postings[postingIndex];
		uint32		codeword = postingCodewords[postingIndex];

		CHECK_FOR_INTERRUPTS();
		if (entry->docId >= meta->tqMultivectorDocCount)
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "centroid posting entry references an out-of-range document");
		if (codeword >= codebookSize)
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "centroid posting entry has invalid codeword");
		if (docCodeCounts[entry->docId] == PG_UINT32_MAX)
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "centroid doc code map is too large");
		docCodeCounts[entry->docId]++;
	}

	oldCtx = MemoryContextSwitchTo(storage->ctx);
	docCodeOffsets =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
											   (Size) meta->tqMultivectorDocCount + 1,
											   "pgturbohybrid multivector centroid doc code offsets"));
	docCodes =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
											   Max(postingCount, 1U),
											   "pgturbohybrid multivector centroid doc codes"));
	MemoryContextSwitchTo(oldCtx);

	docCodeOffsets[0] = 0;
	for (uint32 docId = 0; docId < meta->tqMultivectorDocCount; docId++)
	{
		uint64		nextOffset =
			(uint64) docCodeOffsets[docId] + (uint64) docCodeCounts[docId];

		CHECK_FOR_INTERRUPTS();
		if (nextOffset > (uint64) PG_UINT32_MAX)
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "centroid doc code map is too large");
		docCodeOffsets[docId + 1] = (uint32) nextOffset;
	}
	docCodeWrite =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
											   meta->tqMultivectorDocCount,
											   "pgturbohybrid multivector centroid doc code write cursors"));
	memcpy(docCodeWrite, docCodeOffsets,
		   sizeof(uint32) * (Size) meta->tqMultivectorDocCount);
	for (uint32 postingIndex = 0; postingIndex < postingCount; postingIndex++)
	{
		const PgturbohybridGraphMultiVectorCentroidPostingEntry *entry =
			&postings[postingIndex];
		uint32		docId = entry->docId;
		uint32		offset = docCodeWrite[docId]++;

		CHECK_FOR_INTERRUPTS();
		if (offset >= docCodeOffsets[docId + 1])
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "centroid doc code map write cursor is invalid");
		docCodes[offset] = postingCodewords[postingIndex];
	}

	storage->multivectorCentroidDocCodeOffsets = docCodeOffsets;
	storage->multivectorCentroidDocCodes = docCodes;
	storage->multivectorCentroidDocCodeCount = postingCount;
	storage->multivectorCentroidDocCodesLoaded = true;
	pfree(docCodeWrite);
	pfree(docCodeCounts);
}

static void
PgturbohybridGraphBuildQuantizedInvertedDocCodeMap(Relation index,
												   PgturbohybridGraphMetaPageData *meta,
												   PgturbohybridGraphScanStorage *storage,
												   const PgturbohybridGraphMultiVectorQuantizedPostingEntry *postings,
												   const uint32 *postingCodewords,
												   uint32 postingCount,
												   uint32 codebookSize)
{
	MemoryContext oldCtx;
	uint32	   *docCodeCounts;
	uint32	   *docCodeWrite;
	uint32	   *docCodeOffsets;
	uint32	   *docCodes;

	if (postings == NULL || postingCodewords == NULL)
		return;

	docCodeCounts =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
											   meta->tqMultivectorDocCount,
											   "pgturbohybrid multivector quantized doc code counts"));
	for (uint32 postingIndex = 0; postingIndex < postingCount; postingIndex++)
	{
		const PgturbohybridGraphMultiVectorQuantizedPostingEntry *entry =
			&postings[postingIndex];
		uint32		codeword = postingCodewords[postingIndex];

		CHECK_FOR_INTERRUPTS();
		if (entry->docId >= meta->tqMultivectorDocCount)
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "quantized posting entry references an out-of-range document");
		if (codeword >= codebookSize)
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "quantized posting entry has invalid codeword");
		if (docCodeCounts[entry->docId] == PG_UINT32_MAX)
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "quantized doc code map is too large");
		docCodeCounts[entry->docId]++;
	}

	oldCtx = MemoryContextSwitchTo(storage->ctx);
	docCodeOffsets =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
											   (Size) meta->tqMultivectorDocCount + 1,
											   "pgturbohybrid multivector quantized doc-code offsets"));
	docCodes =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
											   Max(postingCount, 1U),
											   "pgturbohybrid multivector quantized doc codes"));
	MemoryContextSwitchTo(oldCtx);

	docCodeOffsets[0] = 0;
	for (uint32 docId = 0; docId < meta->tqMultivectorDocCount; docId++)
	{
		uint64		nextOffset =
			(uint64) docCodeOffsets[docId] + (uint64) docCodeCounts[docId];

		CHECK_FOR_INTERRUPTS();
		if (nextOffset > (uint64) PG_UINT32_MAX)
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "quantized doc code map is too large");
		docCodeOffsets[docId + 1] = (uint32) nextOffset;
	}
	docCodeWrite =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
											   meta->tqMultivectorDocCount,
											   "pgturbohybrid multivector quantized doc-code write cursors"));
	memcpy(docCodeWrite, docCodeOffsets,
		   sizeof(uint32) * (Size) meta->tqMultivectorDocCount);
	for (uint32 postingIndex = 0; postingIndex < postingCount; postingIndex++)
	{
		const PgturbohybridGraphMultiVectorQuantizedPostingEntry *entry =
			&postings[postingIndex];
		uint32		docId = entry->docId;
		uint32		offset = docCodeWrite[docId]++;

		CHECK_FOR_INTERRUPTS();
		if (offset >= docCodeOffsets[docId + 1])
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "quantized doc code map write cursor is invalid");
		docCodes[offset] = postingCodewords[postingIndex];
	}

	storage->multivectorQuantizedInvertedDocCodeOffsets = docCodeOffsets;
	storage->multivectorQuantizedInvertedDocCodes = docCodes;
	storage->multivectorQuantizedInvertedDocCodeCount = postingCount;
	storage->multivectorQuantizedInvertedDocCodesLoaded = true;
	pfree(docCodeWrite);
	pfree(docCodeCounts);
}

static void
PgturbohybridGraphAppendCentroidDocCodeScratch(Relation index,
											   MemoryContext ctx,
											   uint32 docId,
											   const uint32 *codes,
											   uint16 count,
											   uint32 **scratchDocIds,
											   uint32 **scratchCodes,
											   uint32 *scratchCount,
											   uint32 *scratchCapacity)
{
	uint32		required;

	if (count == 0)
		return;
	if ((uint64) *scratchCount + count > (uint64) PG_UINT32_MAX)
		PgturbohybridGraphMultiVectorDocMapError(index,
												 "centroid doc-code sidecar is too large");
	required = *scratchCount + count;
	if (required > *scratchCapacity)
	{
		uint32		oldCapacity = *scratchCapacity;
		uint32		newCapacity = oldCapacity == 0 ? 1024 : oldCapacity;
		Size		oldBytes;
		Size		newBytes;
		MemoryContext oldCtx;

		while (newCapacity < required)
		{
			if (newCapacity > PG_UINT32_MAX / 2)
				PgturbohybridGraphMultiVectorDocMapError(index,
														 "centroid doc-code sidecar is too large");
			newCapacity += newCapacity;
		}
		oldBytes = PgturbohybridCheckedArrayBytes(sizeof(uint32),
												 oldCapacity,
												 "pgturbohybrid multivector centroid doc-code scratch");
		newBytes = PgturbohybridCheckedArrayBytes(sizeof(uint32),
												 newCapacity,
												 "pgturbohybrid multivector centroid doc-code scratch");
		if (*scratchDocIds == NULL)
		{
			oldCtx = MemoryContextSwitchTo(ctx);
			*scratchDocIds = palloc0(newBytes);
			*scratchCodes = palloc0(newBytes);
			MemoryContextSwitchTo(oldCtx);
		}
		else
		{
			*scratchDocIds = repalloc(*scratchDocIds, newBytes);
			memset((char *) *scratchDocIds + oldBytes, 0, newBytes - oldBytes);
			*scratchCodes = repalloc(*scratchCodes, newBytes);
			memset((char *) *scratchCodes + oldBytes, 0, newBytes - oldBytes);
		}
		*scratchCapacity = newCapacity;
	}

	for (uint16 i = 0; i < count; i++)
	{
		(*scratchDocIds)[*scratchCount + i] = docId;
		(*scratchCodes)[*scratchCount + i] = codes[i];
	}
	*scratchCount += count;
}

static void
PgturbohybridGraphMaterializeCentroidDocCodeMap(Relation index,
												PgturbohybridGraphMetaPageData *meta,
												PgturbohybridGraphScanStorage *storage,
												const uint32 *docCodeCounts,
												const uint32 *scratchDocIds,
												const uint32 *scratchCodes,
												uint32 scratchCount,
												uint32 codebookSize)
{
	MemoryContext oldCtx;
	uint32	   *docCodeOffsets;
	uint32	   *docCodeWrite;
	uint32	   *docCodes;

	if (docCodeCounts == NULL || scratchCount == 0)
		return;
	oldCtx = MemoryContextSwitchTo(storage->ctx);
	docCodeOffsets =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
											   (Size) meta->tqMultivectorDocCount + 1,
											   "pgturbohybrid multivector centroid doc-code offsets"));
	docCodes =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
											   Max(scratchCount, 1U),
											   "pgturbohybrid multivector centroid doc codes"));
	MemoryContextSwitchTo(oldCtx);

	docCodeOffsets[0] = 0;
	for (uint32 docId = 0; docId < meta->tqMultivectorDocCount; docId++)
	{
		uint64		nextOffset =
			(uint64) docCodeOffsets[docId] + (uint64) docCodeCounts[docId];

		CHECK_FOR_INTERRUPTS();
		if (docCodeCounts[docId] == 0 || nextOffset > (uint64) PG_UINT32_MAX)
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "centroid doc-code sidecar does not cover every document");
		docCodeOffsets[docId + 1] = (uint32) nextOffset;
	}
	if (docCodeOffsets[meta->tqMultivectorDocCount] != scratchCount)
		PgturbohybridGraphMultiVectorDocMapError(index,
												 "centroid doc-code sidecar count is invalid");
	docCodeWrite =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
											   meta->tqMultivectorDocCount,
											   "pgturbohybrid multivector centroid doc-code write cursors"));
	memcpy(docCodeWrite, docCodeOffsets,
		   sizeof(uint32) * (Size) meta->tqMultivectorDocCount);
	for (uint32 scratchIndex = 0; scratchIndex < scratchCount; scratchIndex++)
	{
		uint32		docId = scratchDocIds[scratchIndex];
		uint32		code = scratchCodes[scratchIndex];
		uint32		offset;

		CHECK_FOR_INTERRUPTS();
		if (docId >= meta->tqMultivectorDocCount || code >= codebookSize)
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "centroid doc-code sidecar is invalid");
		offset = docCodeWrite[docId]++;
		if (offset >= docCodeOffsets[docId + 1])
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "centroid doc-code sidecar write cursor is invalid");
		docCodes[offset] = code;
	}

	storage->multivectorCentroidDocCodeOffsets = docCodeOffsets;
	storage->multivectorCentroidDocCodes = docCodes;
	storage->multivectorCentroidDocCodeCount = scratchCount;
	storage->multivectorCentroidDocCodesLoaded = true;
	pfree(docCodeWrite);
}

bool
PgturbohybridGraphLoadMultiVectorDocMapWithStats(Relation index,
									 PgturbohybridGraphMetaPageData *meta,
									 PgturbohybridGraphScanStorage *storage,
									 bool require,
									 PgturbohybridMultiVectorDocSidecarAccessStats *stats)
{
	MemoryContext oldCtx;
	bool	   *nodeSeen;
	bool	   *docSeen;
	uint32	   *vectorFloatCounts = NULL;
	uint32	   *centroidFloatCounts = NULL;
	uint32	   *centroidPostingListCounts = NULL;
	uint32	   *centroidPostingScratchCodewords = NULL;
	PgturbohybridGraphMultiVectorCentroidPostingEntry *centroidPostingScratch = NULL;
	uint32		centroidPostingScratchCount = 0;
	uint32		centroidPostingScratchCapacity = 0;
	uint32	   *centroidDocCodeCounts = NULL;
	uint32	   *centroidDocCodeScratchDocIds = NULL;
	uint32	   *centroidDocCodeScratchCodes = NULL;
	uint32		centroidDocCodeScratchCount = 0;
	uint32		centroidDocCodeScratchCapacity = 0;
	uint32	   *quantizedPostingListCounts = NULL;
	uint32	   *quantizedPostingScratchCodewords = NULL;
	PgturbohybridGraphMultiVectorQuantizedPostingEntry *quantizedPostingScratch = NULL;
	uint32		quantizedPostingScratchCount = 0;
	uint32		quantizedPostingScratchCapacity = 0;
	uint32		nodesSeen = 0;
	uint32		docsSeen = 0;
	BlockNumber blkno;
	bool		documentNodes =
		meta->tqMultivectorGraphMode ==
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES;
	bool		hasCentroids =
		documentNodes &&
		(meta->tqMultivectorDocMapFlags &
		 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CENTROIDS) != 0;
	bool		hasDocVectors =
		documentNodes &&
		(meta->tqMultivectorDocMapFlags &
		 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_DOC_VECTORS) != 0;
	bool		proxyOnlyDocMap =
		documentNodes &&
		(meta->tqMultivectorDocMapFlags &
		 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_PROXY_ONLY) != 0;
	bool		hasCentroidPostings =
		hasCentroids &&
		(meta->tqMultivectorDocMapFlags &
		 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CENTROID_POSTINGS) != 0;
	bool		hasCentroidDocCodes =
		hasCentroidPostings &&
		(meta->tqMultivectorDocMapFlags &
		 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CENTROID_DOC_CODES) != 0;
	bool		hasQuantizedPostings =
		documentNodes &&
		(meta->tqMultivectorDocMapFlags &
		 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_QUANTIZED_POSTINGS) != 0;
	bool		hasQuantizedCodebook =
		hasQuantizedPostings &&
		(meta->tqMultivectorDocMapFlags &
		 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_QUANTIZED_CODEBOOK) != 0;
	bool		quantizedCodebookSeen = false;
	uint32		centroidPostingCodebookSize =
		documentNodes ? (uint32) meta->dimensions * 2U : 0;
	uint32		quantizedPostingCodebookSize =
		documentNodes ? (uint32) meta->dimensions * 2U : 0;
	bool		pagedDocVectors =
		documentNodes && storage->multivectorDocVectorsPaged;
	bool		skipCentroidVectors =
		documentNodes && storage->multivectorDocCentroidVectorsSkipped;

	if (!BlockNumberIsValid(meta->tqMultivectorDocMapStartBlkno))
	{
		if (require)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("multivector docmap sidecar is not available for index \"%s\"",
							RelationGetRelationName(index)),
					 errhint("REINDEX the index to build the multivector docmap sidecar, or set turbohybrid.multivector_docmap = 'auto' or 'off'.")));
		return false;
	}
	if (meta->tqMultivectorDocMapVersion !=
		PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION)
		PgturbohybridGraphMultiVectorDocMapError(index,
												 "unsupported docmap sidecar version");
	if (meta->tqMultivectorDocMapPageCount == 0 ||
		meta->tqMultivectorDocCount == 0)
		PgturbohybridGraphMultiVectorDocMapError(index,
												 "docmap metapage counts are empty");

	if (storage->multivectorDocMapLoaded)
	{
		if (stats != NULL)
			stats->cacheHits++;
		return !documentNodes ||
			((proxyOnlyDocMap || !hasDocVectors ||
			  storage->multivectorDocVectorsLoaded) &&
			 (!hasCentroids || storage->multivectorDocCentroidsLoaded ||
			  skipCentroidVectors) &&
				 (!hasCentroidPostings ||
				  (storage->multivectorCentroidPostingsLoaded &&
				   storage->multivectorCentroidPostings != NULL &&
				   storage->multivectorCentroidPostingListOffsets != NULL &&
				   storage->multivectorCentroidPostingCodebookSize > 0 &&
				   (!hasCentroidDocCodes ||
					(storage->multivectorCentroidDocCodesLoaded &&
					 storage->multivectorCentroidDocCodeOffsets != NULL &&
					 storage->multivectorCentroidDocCodes != NULL)))) &&
			 (!hasQuantizedPostings ||
			  (storage->multivectorQuantizedInvertedPostingsLoaded &&
			   storage->multivectorQuantizedInvertedPostings != NULL &&
			   storage->multivectorQuantizedInvertedListOffsets != NULL &&
			   storage->multivectorQuantizedInvertedCodebookSize > 0 &&
			   storage->multivectorQuantizedInvertedCodebookDim > 0 &&
			   storage->multivectorQuantizedInvertedDocCodesLoaded &&
			   storage->multivectorQuantizedInvertedDocCodeOffsets != NULL &&
			   storage->multivectorQuantizedInvertedDocCodes != NULL)));
	}

	oldCtx = MemoryContextSwitchTo(storage->ctx);
	storage->multivectorNodeMap =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(TqMultiVectorNodeMapEntry),
											   meta->tqNodeCount,
											   "pgturbohybrid multivector node docmap"));
	storage->multivectorDocMap =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(TqMultiVectorDocMapEntry),
											   meta->tqMultivectorDocCount,
											   "pgturbohybrid multivector docmap"));
	if (documentNodes)
	{
		if (hasDocVectors)
		{
			storage->multivectorDocVectors =
				palloc0(PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridMultiVector *),
													   meta->tqMultivectorDocCount,
													   "pgturbohybrid multivector document vectors"));
			vectorFloatCounts =
				palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
													   meta->tqMultivectorDocCount,
													   "pgturbohybrid multivector document vector coverage"));
			if (pagedDocVectors)
			{
				Size		chunkMetaBytes;

				storage->multivectorDocVectorFirstChunk =
					palloc(PgturbohybridCheckedArrayBytes(sizeof(uint32),
														  meta->tqMultivectorDocCount,
														  "pgturbohybrid multivector document vector first chunks"));
				storage->multivectorDocVectorChunkCounts =
					palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
														   meta->tqMultivectorDocCount,
														   "pgturbohybrid multivector document vector chunk counts"));
				chunkMetaBytes =
					PgturbohybridCheckedArrayBytes(sizeof(uint32),
												   meta->tqMultivectorDocCount,
												   "pgturbohybrid multivector document vector first chunks");
				memset(storage->multivectorDocVectorFirstChunk, 0xff,
					   chunkMetaBytes);
			}
		}
		if (hasCentroids && !skipCentroidVectors)
		{
			storage->multivectorDocCentroids =
				palloc0(PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridMultiVector *),
													   meta->tqMultivectorDocCount,
													   "pgturbohybrid multivector document centroids"));
			storage->multivectorDocCentroidResiduals =
				palloc0(PgturbohybridCheckedArrayBytes(sizeof(float),
													   meta->tqMultivectorDocCount,
													   "pgturbohybrid multivector document centroid residuals"));
			centroidFloatCounts =
				palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
													   meta->tqMultivectorDocCount,
													   "pgturbohybrid multivector document centroid coverage"));
		}
			if (hasCentroidPostings)
				centroidPostingListCounts =
					palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
														   centroidPostingCodebookSize,
														   "pgturbohybrid multivector centroid posting counts"));
			if (hasCentroidDocCodes)
				centroidDocCodeCounts =
					palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
														   meta->tqMultivectorDocCount,
														   "pgturbohybrid multivector centroid doc-code counts"));
			if (hasQuantizedPostings && !hasQuantizedCodebook)
				quantizedPostingListCounts =
					palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
													   quantizedPostingCodebookSize,
													   "pgturbohybrid multivector quantized posting counts"));
	}
	nodeSeen =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(bool),
											   meta->tqNodeCount,
											   "pgturbohybrid multivector node docmap seen"));
	docSeen =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(bool),
											   meta->tqMultivectorDocCount,
											   "pgturbohybrid multivector docmap seen"));
	MemoryContextSwitchTo(oldCtx);

	blkno = meta->tqMultivectorDocMapStartBlkno;
	for (uint32 pageNo = 0; pageNo < meta->tqMultivectorDocMapPageCount; pageNo++)
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;
		BlockNumber nextblkno;
		instr_time	pageReadStart;

		CHECK_FOR_INTERRUPTS();
		if (!BlockNumberIsValid(blkno))
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "docmap page chain ended early");
		if (stats != NULL)
			INSTR_TIME_SET_CURRENT(pageReadStart);
		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = PgturbohybridGraphPageGetOpaque(page);
		if (stats != NULL)
		{
			stats->pagesRead++;
			stats->docMapPagesRead++;
			stats->cacheMisses++;
			stats->pageReadUs +=
				(uint64) PgturbohybridGraphElapsedUsSince(pageReadStart);
		}
		if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) !=
			PGTURBOHYBRID_GRAPH_PAGE_KIND_MULTIVECTOR_DOCMAP)
		{
			UnlockReleaseBuffer(buf);
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "docmap page kind mismatch");
		}

		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff;
			 offno = OffsetNumberNext(offno))
		{
			ItemId		iid = PageGetItemId(page, offno);
			Item		item = PageGetItem(page, iid);
			Size		itemSize = ItemIdGetLength(iid);
			uint8		type = *((uint8 *) item);

			CHECK_FOR_INTERRUPTS();
			if (stats != NULL)
			{
				stats->bytesTouched += itemSize;
				stats->docMapBytesTouched += itemSize;
			}
			if (type ==
				PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_NODE_TUPLE_TYPE)
			{
				PgturbohybridGraphMultiVectorDocMapNodeTuple tuple =
					(PgturbohybridGraphMultiVectorDocMapNodeTuple) item;
				Size		tupleSize;

				if (itemSize < offsetof(PgturbohybridGraphMultiVectorDocMapNodeTupleData,
										entries) ||
					tuple->magic != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC ||
					tuple->version != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION ||
					tuple->count == 0)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "malformed node docmap tuple");
				tupleSize =
					PgturbohybridGraphMultiVectorDocMapNodeTupleSize(tuple->count);
				if (itemSize < tupleSize ||
					(uint64) tuple->firstNodeId + tuple->count >
					meta->tqNodeCount)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "node docmap tuple range is invalid");
				for (uint16 i = 0; i < tuple->count; i++)
				{
					uint32		nodeId = tuple->firstNodeId + i;
					TqMultiVectorNodeMapEntry *entry =
						&tuple->entries[i];

					CHECK_FOR_INTERRUPTS();
					if (nodeSeen[nodeId] ||
						entry->docId >= meta->tqMultivectorDocCount)
						PgturbohybridGraphMultiVectorDocMapError(index,
																 "node docmap entry is invalid");
					nodeSeen[nodeId] = true;
					storage->multivectorNodeMap[nodeId] = *entry;
					nodesSeen++;
				}
			}
			else if (type ==
					 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_DOC_TUPLE_TYPE)
			{
				PgturbohybridGraphMultiVectorDocMapDocTuple tuple =
					(PgturbohybridGraphMultiVectorDocMapDocTuple) item;
				Size		tupleSize;

				if (itemSize < offsetof(PgturbohybridGraphMultiVectorDocMapDocTupleData,
										entries) ||
					tuple->magic != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC ||
					tuple->version != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION ||
					tuple->count == 0)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "malformed document docmap tuple");
				tupleSize =
					PgturbohybridGraphMultiVectorDocMapDocTupleSize(tuple->count);
				if (itemSize < tupleSize ||
					(uint64) tuple->firstDocId + tuple->count >
					meta->tqMultivectorDocCount)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "document docmap tuple range is invalid");
				for (uint16 i = 0; i < tuple->count; i++)
				{
					uint32		docId = tuple->firstDocId + i;
					TqMultiVectorDocMapEntry *entry =
						&tuple->entries[i];

					CHECK_FOR_INTERRUPTS();
					if (docSeen[docId] ||
						entry->tokenCount == 0 ||
						entry->firstNodeId >= meta->tqNodeCount ||
						(!documentNodes &&
						 (uint64) entry->firstNodeId + entry->tokenCount >
						 meta->tqNodeCount))
						PgturbohybridGraphMultiVectorDocMapError(index,
																 "document docmap entry is invalid");
					docSeen[docId] = true;
					storage->multivectorDocMap[docId] = *entry;
					docsSeen++;
				}
			}
				else if (type ==
						 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VECTOR_TUPLE_TYPE ||
						 type ==
						 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VECTOR_F16_TUPLE_TYPE ||
						 type ==
						 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VECTOR_SQ8_TUPLE_TYPE)
				{
					PgturbohybridGraphDecodedMultiVectorDocVectorTuple tuple;
					TqMultiVectorDocMapEntry *entry;
					PgturbohybridMultiVector *mv;
					Size		totalFloats;
					instr_time	vectorReconstructStart;

					if (stats != NULL)
						INSTR_TIME_SET_CURRENT(vectorReconstructStart);
					if (!documentNodes)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "document vector tuple found in token-node docmap");
					if (!hasDocVectors)
						PgturbohybridGraphMultiVectorDocMapError(index,
																 "document vector tuple found without document vector storage flag");
					if (!PgturbohybridGraphDecodeMultiVectorDocVectorTuple(item,
																		   itemSize,
																		   &tuple) ||
						tuple.docId >= meta->tqMultivectorDocCount)
						PgturbohybridGraphMultiVectorDocMapError(index,
																 "malformed document vector tuple");
					if (!docSeen[tuple.docId])
						PgturbohybridGraphMultiVectorDocMapError(index,
																 "document vector tuple range is invalid");
					entry = &storage->multivectorDocMap[tuple.docId];
					totalFloats =
						PgturbohybridMultiVectorFloatCount(entry->tokenCount,
														   meta->dimensions);
					if ((uint64) tuple.startFloat + tuple.count > totalFloats)
						PgturbohybridGraphMultiVectorDocMapError(index,
																 "document vector tuple range is invalid");
					if (pagedDocVectors)
					{
						if (stats != NULL)
							stats->vectorChunkRefBytesTouched += itemSize;
						PgturbohybridGraphRememberMultiVectorDocVectorChunk(index,
																			storage,
																			tuple.docId,
																			blkno,
																			offno,
																			tuple.startFloat,
																			tuple.count,
																			tuple.storageKind);
					}
					else
					{
						mv = storage->multivectorDocVectors[tuple.docId];
						if (mv == NULL)
						{
							Size		mvSize =
								PgturbohybridMultiVectorSize(entry->tokenCount,
														 meta->dimensions);

						mv = MemoryContextAllocZero(storage->ctx, mvSize);
							SET_VARSIZE(mv, mvSize);
							mv->dim = meta->dimensions;
							mv->count = entry->tokenCount;
							storage->multivectorDocVectors[tuple.docId] = mv;
							if (stats != NULL)
								stats->residentVectorsLoaded++;
						}
						PgturbohybridGraphDecodeMultiVectorDocValues(mv->values,
																	 tuple.startFloat,
																	 tuple.count,
																	 tuple.storageKind,
																	 tuple.values,
																	 tuple.scale);
						if (stats != NULL)
							stats->residentVectorBytesLoaded += itemSize;
					}
					if (tuple.startFloat != vectorFloatCounts[tuple.docId])
						PgturbohybridGraphMultiVectorDocMapError(index,
																 "document vector tuple chunks are not contiguous");
					vectorFloatCounts[tuple.docId] += tuple.count;
					if (stats != NULL)
						stats->vectorReconstructUs +=
							(uint64) PgturbohybridGraphElapsedUsSince(vectorReconstructStart);
				}
			else if (type ==
					 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_CENTROID_TUPLE_TYPE)
			{
				PgturbohybridGraphMultiVectorDocMapCentroidTuple tuple =
					(PgturbohybridGraphMultiVectorDocMapCentroidTuple) item;
				PgturbohybridMultiVector *centroids;
				Size		totalFloats;
				Size		tupleSize;

				if (!documentNodes)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "document centroid tuple found in token-node docmap");
				if (!hasCentroids)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "document centroid tuple found without centroid storage flag");
				if (itemSize < offsetof(PgturbohybridGraphMultiVectorDocMapCentroidTupleData,
										values) ||
					tuple->magic != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC ||
					tuple->version != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION ||
					tuple->count == 0 ||
					tuple->docId >= meta->tqMultivectorDocCount ||
					tuple->centroidCount == 0)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "malformed document centroid tuple");
				tupleSize =
					PgturbohybridGraphMultiVectorDocMapCentroidTupleSize(tuple->count);
				if (itemSize < tupleSize || !docSeen[tuple->docId])
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "document centroid tuple range is invalid");
				totalFloats =
					PgturbohybridMultiVectorFloatCount(tuple->centroidCount,
													   meta->dimensions);
				if ((uint64) tuple->startFloat + tuple->count > totalFloats)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "document centroid tuple range is invalid");
				if (skipCentroidVectors)
					continue;
				centroids = storage->multivectorDocCentroids[tuple->docId];
				if (centroids == NULL)
				{
					Size		mvSize =
						PgturbohybridMultiVectorSize(tuple->centroidCount,
													 meta->dimensions);

					centroids = MemoryContextAllocZero(storage->ctx, mvSize);
					SET_VARSIZE(centroids, mvSize);
					centroids->dim = meta->dimensions;
					centroids->count = tuple->centroidCount;
					storage->multivectorDocCentroids[tuple->docId] = centroids;
					storage->multivectorDocCentroidResiduals[tuple->docId] =
						tuple->residualMean;
				}
				else if (centroids->count != tuple->centroidCount)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "document centroid tuple count changed across chunks");
				if (tuple->startFloat != centroidFloatCounts[tuple->docId])
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "document centroid tuple chunks are not contiguous");
				memcpy(centroids->values + tuple->startFloat, tuple->values,
					   sizeof(float) * tuple->count);
				centroidFloatCounts[tuple->docId] += tuple->count;
			}
			else if (type ==
					 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_CENTROID_F16_TUPLE_TYPE)
			{
				PgturbohybridGraphMultiVectorDocMapCentroidF16Tuple tuple =
					(PgturbohybridGraphMultiVectorDocMapCentroidF16Tuple) item;
				PgturbohybridMultiVector *centroids;
				Size		totalFloats;
				Size		tupleSize;

				if (!documentNodes)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "document f16 centroid tuple found in token-node docmap");
				if (!hasCentroids)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "document f16 centroid tuple found without centroid storage flag");
				if ((meta->tqMultivectorDocMapFlags &
					 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CENTROIDS_F16) == 0)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "document f16 centroid tuple found without f16 centroid storage flag");
				if (itemSize < offsetof(PgturbohybridGraphMultiVectorDocMapCentroidF16TupleData,
										values) ||
					tuple->magic != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC ||
					tuple->version != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION ||
					tuple->count == 0 ||
					tuple->docId >= meta->tqMultivectorDocCount ||
					tuple->centroidCount == 0)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "malformed document f16 centroid tuple");
				tupleSize =
					PgturbohybridGraphMultiVectorDocMapCentroidF16TupleSize(tuple->count);
				if (itemSize < tupleSize || !docSeen[tuple->docId])
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "document f16 centroid tuple range is invalid");
				totalFloats =
					PgturbohybridMultiVectorFloatCount(tuple->centroidCount,
													   meta->dimensions);
				if ((uint64) tuple->startFloat + tuple->count > totalFloats)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "document f16 centroid tuple range is invalid");
				if (skipCentroidVectors)
					continue;
				centroids = storage->multivectorDocCentroids[tuple->docId];
				if (centroids == NULL)
				{
					Size		mvSize =
						PgturbohybridMultiVectorSize(tuple->centroidCount,
													 meta->dimensions);

					centroids = MemoryContextAllocZero(storage->ctx, mvSize);
					SET_VARSIZE(centroids, mvSize);
					centroids->dim = meta->dimensions;
					centroids->count = tuple->centroidCount;
					storage->multivectorDocCentroids[tuple->docId] = centroids;
					storage->multivectorDocCentroidResiduals[tuple->docId] =
						tuple->residualMean;
				}
				else if (centroids->count != tuple->centroidCount)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "document f16 centroid tuple count changed across chunks");
				if (tuple->startFloat != centroidFloatCounts[tuple->docId])
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "document f16 centroid tuple chunks are not contiguous");
				for (uint16 i = 0; i < tuple->count; i++)
				{
					CHECK_FOR_INTERRUPTS();
					centroids->values[tuple->startFloat + i] =
						PgturbohybridGraphHalfToFloat(tuple->values[i]);
				}
				centroidFloatCounts[tuple->docId] += tuple->count;
			}
			else if (type ==
					 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_CENTROID_POSTING_TUPLE_TYPE)
			{
				PgturbohybridGraphMultiVectorDocMapCentroidPostingTuple tuple =
					(PgturbohybridGraphMultiVectorDocMapCentroidPostingTuple) item;
				Size		tupleSize;

				if (!documentNodes)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "centroid posting tuple found in token-node docmap");
				if (!hasCentroidPostings)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "centroid posting tuple found without posting storage flag");
				if (itemSize < offsetof(PgturbohybridGraphMultiVectorDocMapCentroidPostingTupleData,
										entries) ||
					tuple->magic != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC ||
					tuple->version != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION ||
					tuple->count == 0 ||
					tuple->codeword >= centroidPostingCodebookSize)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "malformed centroid posting tuple");
				tupleSize =
					PgturbohybridGraphMultiVectorDocMapCentroidPostingTupleSize(tuple->count);
				if (itemSize < tupleSize)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "centroid posting tuple range is invalid");
				if ((uint64) centroidPostingScratchCount + tuple->count >
					(uint64) PG_UINT32_MAX)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "centroid posting sidecar is too large");
				if (centroidPostingScratchCount + tuple->count >
					centroidPostingScratchCapacity)
				{
					uint32		oldCapacity = centroidPostingScratchCapacity;
					uint32		newCapacity =
						centroidPostingScratchCapacity == 0 ?
						1024 : centroidPostingScratchCapacity;
					Size		oldPostingBytes;
					Size		newPostingBytes;
					Size		oldCodewordBytes;
					Size		newCodewordBytes;

					while (newCapacity <
						   centroidPostingScratchCount + tuple->count)
					{
						if (newCapacity > PG_UINT32_MAX / 2)
							PgturbohybridGraphMultiVectorDocMapError(index,
																	 "centroid posting sidecar is too large");
						newCapacity += newCapacity;
					}
					oldPostingBytes =
						PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridGraphMultiVectorCentroidPostingEntry),
													   oldCapacity,
													   "pgturbohybrid multivector centroid posting scratch");
					newPostingBytes =
						PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridGraphMultiVectorCentroidPostingEntry),
													   newCapacity,
													   "pgturbohybrid multivector centroid posting scratch");
					oldCodewordBytes =
						PgturbohybridCheckedArrayBytes(sizeof(uint32),
													   oldCapacity,
													   "pgturbohybrid multivector centroid posting codewords");
					newCodewordBytes =
						PgturbohybridCheckedArrayBytes(sizeof(uint32),
													   newCapacity,
													   "pgturbohybrid multivector centroid posting codewords");
					if (centroidPostingScratch == NULL)
					{
						centroidPostingScratch =
							MemoryContextAllocZero(storage->ctx,
												   newPostingBytes);
						centroidPostingScratchCodewords =
							MemoryContextAllocZero(storage->ctx,
												   newCodewordBytes);
					}
					else
					{
						centroidPostingScratch =
							repalloc(centroidPostingScratch,
									 newPostingBytes);
						memset((char *) centroidPostingScratch + oldPostingBytes,
							   0, newPostingBytes - oldPostingBytes);
						centroidPostingScratchCodewords =
							repalloc(centroidPostingScratchCodewords,
									 newCodewordBytes);
						memset((char *) centroidPostingScratchCodewords +
							   oldCodewordBytes, 0,
							   newCodewordBytes - oldCodewordBytes);
					}
					centroidPostingScratchCapacity = newCapacity;
				}
				for (uint16 i = 0; i < tuple->count; i++)
				{
					PgturbohybridGraphMultiVectorCentroidPostingEntry *entry =
						&tuple->entries[i];

					CHECK_FOR_INTERRUPTS();
					if (entry->docId >= meta->tqMultivectorDocCount)
						PgturbohybridGraphMultiVectorDocMapError(index,
																 "centroid posting entry is invalid");
					centroidPostingScratch[centroidPostingScratchCount] =
						*entry;
					centroidPostingScratchCodewords[centroidPostingScratchCount] =
						tuple->codeword;
					centroidPostingListCounts[tuple->codeword]++;
					centroidPostingScratchCount++;
				}
			}
				else if (type ==
						 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_QUANTIZED_CODEBOOK_TUPLE_TYPE)
				{
					PgturbohybridGraphMultiVectorDocMapQuantizedCodebookTuple tuple =
						(PgturbohybridGraphMultiVectorDocMapQuantizedCodebookTuple) item;

				if (!documentNodes)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "quantized codebook tuple found in token-node docmap");
				if (!hasQuantizedPostings || !hasQuantizedCodebook)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "quantized codebook tuple found without quantized posting metadata flag");
				if (quantizedCodebookSeen)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "duplicate quantized codebook metadata tuple");
				if (itemSize < MAXALIGN(sizeof(PgturbohybridGraphMultiVectorDocMapQuantizedCodebookTupleData)) ||
					tuple->magic != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC ||
					tuple->version != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION ||
					tuple->dim != (uint32) meta->dimensions ||
					tuple->codebookSize == 0 ||
					tuple->topM == 0)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "malformed quantized codebook metadata tuple");
				quantizedPostingCodebookSize = tuple->codebookSize;
				storage->multivectorQuantizedInvertedCodebookSource =
					tuple->source;
				storage->multivectorQuantizedInvertedCodebookDim = tuple->dim;
				storage->multivectorQuantizedInvertedCodebookTopM =
					tuple->topM;
				strlcpy(storage->multivectorQuantizedInvertedCodebookChecksum,
						tuple->checksum,
						sizeof(storage->multivectorQuantizedInvertedCodebookChecksum));
				oldCtx = MemoryContextSwitchTo(storage->ctx);
				quantizedPostingListCounts =
					palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
														   quantizedPostingCodebookSize,
														   "pgturbohybrid multivector quantized posting counts"));
				MemoryContextSwitchTo(oldCtx);
					quantizedCodebookSeen = true;
				}
				else if (type ==
						 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_CENTROID_DOC_CODE_TUPLE_TYPE)
				{
					PgturbohybridGraphMultiVectorDocMapCentroidDocCodeTuple tuple =
						(PgturbohybridGraphMultiVectorDocMapCentroidDocCodeTuple) item;
					Size		tupleSize;

					if (!documentNodes)
						PgturbohybridGraphMultiVectorDocMapError(index,
																 "centroid doc-code tuple found in token-node docmap");
					if (!hasCentroidDocCodes)
						PgturbohybridGraphMultiVectorDocMapError(index,
																 "centroid doc-code tuple found without doc-code storage flag");
					if (itemSize < offsetof(PgturbohybridGraphMultiVectorDocMapCentroidDocCodeTupleData,
											codes) ||
						tuple->magic != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC ||
						tuple->version != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION ||
						tuple->count == 0 ||
						tuple->docId >= meta->tqMultivectorDocCount ||
						tuple->codebookSize != centroidPostingCodebookSize ||
						tuple->startCode != centroidDocCodeCounts[tuple->docId])
						PgturbohybridGraphMultiVectorDocMapError(index,
																 "malformed centroid doc-code tuple");
					tupleSize =
						PgturbohybridGraphMultiVectorDocMapCentroidDocCodeTupleSize(tuple->count);
					if (itemSize < tupleSize)
						PgturbohybridGraphMultiVectorDocMapError(index,
																 "centroid doc-code tuple range is invalid");
					for (uint16 i = 0; i < tuple->count; i++)
					{
						CHECK_FOR_INTERRUPTS();
						if (tuple->codes[i] >= centroidPostingCodebookSize)
							PgturbohybridGraphMultiVectorDocMapError(index,
																	 "centroid doc-code tuple codeword is invalid");
					}
					PgturbohybridGraphAppendCentroidDocCodeScratch(index,
																   storage->ctx,
																   tuple->docId,
																   tuple->codes,
																   tuple->count,
																   &centroidDocCodeScratchDocIds,
																   &centroidDocCodeScratchCodes,
																   &centroidDocCodeScratchCount,
																   &centroidDocCodeScratchCapacity);
					centroidDocCodeCounts[tuple->docId] += tuple->count;
				}
				else if (type ==
						 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_QUANTIZED_POSTING_TUPLE_TYPE)
				{
				PgturbohybridGraphMultiVectorDocMapQuantizedPostingTuple tuple =
					(PgturbohybridGraphMultiVectorDocMapQuantizedPostingTuple) item;
				Size		tupleSize;

				if (!documentNodes)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "quantized posting tuple found in token-node docmap");
				if (!hasQuantizedPostings)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "quantized posting tuple found without posting storage flag");
				if (quantizedPostingListCounts == NULL)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "quantized posting tuple found before codebook metadata");
				if (itemSize < offsetof(PgturbohybridGraphMultiVectorDocMapQuantizedPostingTupleData,
										entries) ||
					tuple->magic != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC ||
					tuple->version != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION ||
					tuple->count == 0 ||
					tuple->codeword >= quantizedPostingCodebookSize)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "malformed quantized posting tuple");
				tupleSize =
					PgturbohybridGraphMultiVectorDocMapQuantizedPostingTupleSize(tuple->count);
				if (itemSize < tupleSize)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "quantized posting tuple range is invalid");
				if ((uint64) quantizedPostingScratchCount + tuple->count >
					(uint64) PG_UINT32_MAX)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "quantized posting sidecar is too large");
				if (quantizedPostingScratchCount + tuple->count >
					quantizedPostingScratchCapacity)
				{
					uint32		oldCapacity = quantizedPostingScratchCapacity;
					uint32		newCapacity =
						quantizedPostingScratchCapacity == 0 ?
						1024 : quantizedPostingScratchCapacity;
					Size		oldPostingBytes;
					Size		newPostingBytes;
					Size		oldCodewordBytes;
					Size		newCodewordBytes;

					while (newCapacity <
						   quantizedPostingScratchCount + tuple->count)
					{
						if (newCapacity > PG_UINT32_MAX / 2)
							PgturbohybridGraphMultiVectorDocMapError(index,
																	 "quantized posting sidecar is too large");
						newCapacity += newCapacity;
					}
					oldPostingBytes =
						PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridGraphMultiVectorQuantizedPostingEntry),
													   oldCapacity,
													   "pgturbohybrid multivector quantized posting scratch");
					newPostingBytes =
						PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridGraphMultiVectorQuantizedPostingEntry),
													   newCapacity,
													   "pgturbohybrid multivector quantized posting scratch");
					oldCodewordBytes =
						PgturbohybridCheckedArrayBytes(sizeof(uint32),
													   oldCapacity,
													   "pgturbohybrid multivector quantized posting codewords");
					newCodewordBytes =
						PgturbohybridCheckedArrayBytes(sizeof(uint32),
													   newCapacity,
													   "pgturbohybrid multivector quantized posting codewords");
					if (quantizedPostingScratch == NULL)
					{
						quantizedPostingScratch =
							MemoryContextAllocZero(storage->ctx,
												   newPostingBytes);
						quantizedPostingScratchCodewords =
							MemoryContextAllocZero(storage->ctx,
												   newCodewordBytes);
					}
					else
					{
						quantizedPostingScratch =
							repalloc(quantizedPostingScratch,
									 newPostingBytes);
						memset((char *) quantizedPostingScratch + oldPostingBytes,
							   0, newPostingBytes - oldPostingBytes);
						quantizedPostingScratchCodewords =
							repalloc(quantizedPostingScratchCodewords,
									 newCodewordBytes);
						memset((char *) quantizedPostingScratchCodewords +
							   oldCodewordBytes, 0,
							   newCodewordBytes - oldCodewordBytes);
					}
					quantizedPostingScratchCapacity = newCapacity;
				}
				for (uint16 i = 0; i < tuple->count; i++)
				{
					PgturbohybridGraphMultiVectorQuantizedPostingEntry *entry =
						&tuple->entries[i];

					CHECK_FOR_INTERRUPTS();
					if (entry->docId >= meta->tqMultivectorDocCount)
						PgturbohybridGraphMultiVectorDocMapError(index,
																 "quantized posting entry is invalid");
					quantizedPostingScratch[quantizedPostingScratchCount] =
						*entry;
					quantizedPostingScratchCodewords[quantizedPostingScratchCount] =
						tuple->codeword;
					quantizedPostingListCounts[tuple->codeword]++;
					quantizedPostingScratchCount++;
				}
			}
			else if (type ==
					 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_CONTEXT_TUPLE_TYPE)
			{
				PgturbohybridGraphMultiVectorDocMapContextTuple tuple =
					(PgturbohybridGraphMultiVectorDocMapContextTuple) item;
				TqMultiVectorDocMapEntry *entry;
				PgturbohybridMultiVector *mv;
				PgturbohybridMultiVector *extended;
				Size		flatSize;
				Size		extendedSize;
				Size		tupleSize;
				bool		hasFields;

				if (!documentNodes)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "document context tuple found in token-node docmap");
				if (!hasDocVectors)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "document context tuple found without document vector storage flag");
				if (itemSize < offsetof(PgturbohybridGraphMultiVectorDocMapContextTupleData,
										values) ||
					tuple->magic != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC ||
					tuple->version != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION ||
					tuple->contextCount == 0 ||
					tuple->docId >= meta->tqMultivectorDocCount ||
					(tuple->flags & PGTURBOHYBRID_MULTIVECTOR_FLAG_CONTEXTS) == 0 ||
					(tuple->flags & ~PGTURBOHYBRID_MULTIVECTOR_KNOWN_FLAGS) != 0)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "malformed document context tuple");
				hasFields =
					(tuple->flags & PGTURBOHYBRID_MULTIVECTOR_FLAG_FIELDS) != 0;
				tupleSize =
					PgturbohybridGraphMultiVectorDocMapContextTupleSize(tuple->contextCount,
																		hasFields);
				if (itemSize < tupleSize || !docSeen[tuple->docId])
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "document context tuple range is invalid");
				entry = &storage->multivectorDocMap[tuple->docId];
				if (pagedDocVectors)
				{
					if (storage->multivectorDocContextsSkipped)
						continue;
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "paged document-node sidecar does not support context metadata yet");
				}
				if (tuple->contextCount > entry->tokenCount)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "document context tuple range is invalid");
				mv = storage->multivectorDocVectors[tuple->docId];
				flatSize =
					PgturbohybridMultiVectorSize(entry->tokenCount,
												 meta->dimensions);
				if (mv == NULL)
				{
					mv = MemoryContextAllocZero(storage->ctx, flatSize);
					SET_VARSIZE(mv, flatSize);
					mv->dim = meta->dimensions;
					mv->count = entry->tokenCount;
				}
				else if (PgturbohybridMultiVectorHasContexts(mv))
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "duplicate document context tuple");

				extendedSize =
					PgturbohybridMultiVectorExtendedSize(entry->tokenCount,
														 meta->dimensions,
														 tuple->contextCount,
														 hasFields);
				extended = MemoryContextAllocZero(storage->ctx, extendedSize);
				memcpy(extended, mv, flatSize);
				SET_VARSIZE(extended, extendedSize);
				extended->flags = tuple->flags;
				*((int32 *) ((char *) extended + flatSize)) =
					(int32) tuple->contextCount;
				memcpy((int32 *) ((char *) extended + flatSize) + 1,
					   tuple->values,
					   sizeof(int32) * (Size) tuple->contextCount);
				if (hasFields)
					memcpy((int32 *) ((char *) extended + flatSize) + 1 +
						   tuple->contextCount,
						   tuple->values + tuple->contextCount,
						   sizeof(int32) * (Size) tuple->contextCount);
				PgturbohybridCheckMultiVector(extended);
				storage->multivectorDocVectors[tuple->docId] = extended;
				if (mv != extended)
					pfree(mv);
			}
			else
				PgturbohybridGraphMultiVectorDocMapError(index,
														 "unknown docmap tuple type");
		}

		nextblkno = opaque->nextblkno;
		UnlockReleaseBuffer(buf);
		blkno = nextblkno;
	}

	if (nodesSeen != meta->tqNodeCount ||
		docsSeen != meta->tqMultivectorDocCount)
		PgturbohybridGraphMultiVectorDocMapError(index,
												 "docmap sidecar does not cover every node and document");
	for (uint32 nodeId = 0; nodeId < meta->tqNodeCount; nodeId++)
	{
		TqMultiVectorNodeMapEntry *nodeEntry =
			&storage->multivectorNodeMap[nodeId];
		TqMultiVectorDocMapEntry *docEntry =
			&storage->multivectorDocMap[nodeEntry->docId];

		CHECK_FOR_INTERRUPTS();
		if (nodeEntry->tokenOrdinal >= docEntry->tokenCount ||
			(!documentNodes &&
			 (nodeId < docEntry->firstNodeId ||
			  nodeId >= docEntry->firstNodeId + docEntry->tokenCount)) ||
			(documentNodes && nodeId != docEntry->firstNodeId))
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "node docmap entry does not match its document range");
	}
	if (documentNodes)
	{
		if (!hasDocVectors && !proxyOnlyDocMap && !hasCentroids &&
			!hasQuantizedPostings)
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "document-node docmap is missing document vector storage flag");
		if (hasDocVectors)
		{
			for (uint32 docId = 0; docId < meta->tqMultivectorDocCount; docId++)
			{
				TqMultiVectorDocMapEntry *entry =
					&storage->multivectorDocMap[docId];
				Size		totalFloats =
					PgturbohybridMultiVectorFloatCount(entry->tokenCount,
													   meta->dimensions);

				CHECK_FOR_INTERRUPTS();
				if (storage->multivectorDocVectors[docId] == NULL ||
					vectorFloatCounts[docId] != totalFloats)
				{
					if (pagedDocVectors &&
						storage->multivectorDocVectorFirstChunk[docId] != PG_UINT32_MAX &&
						storage->multivectorDocVectorChunkCounts[docId] > 0 &&
						vectorFloatCounts[docId] == totalFloats)
						continue;
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "document-node docmap does not cover every document vector");
				}
			}
			storage->multivectorDocVectorsLoaded = true;
		}
		if (hasQuantizedPostings)
		{
			uint32	   *listOffsets;
			uint32	   *listWrite;
			uint32	   *docPostingCounts;
			uint32		expectedPostings = 0;

			if (hasQuantizedCodebook && !quantizedCodebookSeen)
				PgturbohybridGraphMultiVectorDocMapError(index,
														 "quantized posting sidecar is missing codebook metadata");
			if (!hasQuantizedCodebook)
			{
				storage->multivectorQuantizedInvertedCodebookSource =
					PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_DETERMINISTIC;
				storage->multivectorQuantizedInvertedCodebookDim =
					(uint32) meta->dimensions;
				storage->multivectorQuantizedInvertedCodebookTopM = 1;
				strlcpy(storage->multivectorQuantizedInvertedCodebookChecksum,
						"deterministic",
						sizeof(storage->multivectorQuantizedInvertedCodebookChecksum));
			}
			for (uint32 docId = 0; docId < meta->tqMultivectorDocCount; docId++)
			{
				TqMultiVectorDocMapEntry *entry =
					&storage->multivectorDocMap[docId];

				CHECK_FOR_INTERRUPTS();
				if ((uint64) expectedPostings +
					(uint64) entry->tokenCount *
					(uint64) Max(storage->multivectorQuantizedInvertedCodebookTopM,
								  1U) >
					(uint64) PG_UINT32_MAX)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "quantized posting sidecar is too large");
				expectedPostings +=
					(uint32) entry->tokenCount *
					Max(storage->multivectorQuantizedInvertedCodebookTopM, 1U);
			}
			if (quantizedPostingScratchCount != expectedPostings)
				PgturbohybridGraphMultiVectorDocMapError(index,
														 "quantized posting sidecar does not cover every document token");

			oldCtx = MemoryContextSwitchTo(storage->ctx);
			listOffsets =
				palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
													   (Size) quantizedPostingCodebookSize + 1,
													   "pgturbohybrid multivector quantized posting offsets"));
			listWrite =
				palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
													   quantizedPostingCodebookSize,
													   "pgturbohybrid multivector quantized posting write cursors"));
			docPostingCounts =
				palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
													   meta->tqMultivectorDocCount,
													   "pgturbohybrid multivector quantized posting coverage"));
			storage->multivectorQuantizedInvertedPostings =
				palloc0(PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridGraphMultiVectorQuantizedPostingEntry),
													   Max(quantizedPostingScratchCount, 1U),
													   "pgturbohybrid multivector quantized postings"));
			MemoryContextSwitchTo(oldCtx);

			listOffsets[0] = 0;
			for (uint32 codeword = 0; codeword < quantizedPostingCodebookSize;
				 codeword++)
			{
				CHECK_FOR_INTERRUPTS();
				listOffsets[codeword + 1] =
					listOffsets[codeword] + quantizedPostingListCounts[codeword];
				listWrite[codeword] = listOffsets[codeword];
			}
			for (uint32 postingIndex = 0;
				 postingIndex < quantizedPostingScratchCount;
				 postingIndex++)
			{
				PgturbohybridGraphMultiVectorQuantizedPostingEntry *entry =
					&quantizedPostingScratch[postingIndex];
				uint32		codeword =
					quantizedPostingScratchCodewords[postingIndex];
				TqMultiVectorDocMapEntry *docEntry;
				uint32		offset;

				CHECK_FOR_INTERRUPTS();
				if (codeword >= quantizedPostingCodebookSize)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "quantized posting entry has invalid codeword");
				docEntry = &storage->multivectorDocMap[entry->docId];
				if (entry->tokenOrdinal >= (uint32) docEntry->tokenCount)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "quantized posting entry is invalid");
				docPostingCounts[entry->docId]++;
				offset = listWrite[codeword]++;
				storage->multivectorQuantizedInvertedPostings[offset] =
					*entry;
			}
			for (uint32 docId = 0; docId < meta->tqMultivectorDocCount; docId++)
			{
				TqMultiVectorDocMapEntry *entry =
					&storage->multivectorDocMap[docId];

				CHECK_FOR_INTERRUPTS();
				if (docPostingCounts[docId] !=
					(uint32) entry->tokenCount *
					Max(storage->multivectorQuantizedInvertedCodebookTopM, 1U))
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "quantized posting sidecar does not cover every document token");
			}
			storage->multivectorQuantizedInvertedListOffsets = listOffsets;
			storage->multivectorQuantizedInvertedCodebookSize =
				quantizedPostingCodebookSize;
			storage->multivectorQuantizedInvertedPostingCount =
				quantizedPostingScratchCount;
			storage->multivectorQuantizedInvertedPostingsLoaded = true;
			PgturbohybridGraphBuildQuantizedInvertedDocCodeMap(index,
															   meta,
															   storage,
															   quantizedPostingScratch,
															   quantizedPostingScratchCodewords,
															   quantizedPostingScratchCount,
															   quantizedPostingCodebookSize);
			pfree(listWrite);
			pfree(docPostingCounts);
		}
		if (hasCentroids && !skipCentroidVectors)
		{
			for (uint32 docId = 0; docId < meta->tqMultivectorDocCount; docId++)
			{
				PgturbohybridMultiVector *centroids =
					storage->multivectorDocCentroids[docId];
				Size		totalFloats;

				CHECK_FOR_INTERRUPTS();
				if (centroids == NULL)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "document-node docmap does not cover every document centroid");
				totalFloats =
					PgturbohybridMultiVectorFloatCount(centroids->count,
													   meta->dimensions);
				if (centroidFloatCounts[docId] != totalFloats)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "document-node docmap does not cover every document centroid");
			}
			storage->multivectorDocCentroidsLoaded = true;
			if (hasCentroidPostings)
			{
				uint32	   *listOffsets;
				uint32	   *listWrite;
				uint32	   *docPostingCounts;
				uint32		expectedPostings = 0;

				for (uint32 docId = 0; docId < meta->tqMultivectorDocCount; docId++)
				{
					PgturbohybridMultiVector *centroids =
						storage->multivectorDocCentroids[docId];

					CHECK_FOR_INTERRUPTS();
					if ((uint64) expectedPostings + (uint64) centroids->count >
						(uint64) PG_UINT32_MAX)
						PgturbohybridGraphMultiVectorDocMapError(index,
																 "centroid posting sidecar is too large");
					expectedPostings += (uint32) centroids->count;
				}
				if (centroidPostingScratchCount != expectedPostings)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "centroid posting sidecar does not cover every document centroid");

				oldCtx = MemoryContextSwitchTo(storage->ctx);
				listOffsets =
					palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
														   (Size) centroidPostingCodebookSize + 1,
														   "pgturbohybrid multivector centroid posting offsets"));
				listWrite =
					palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
														   centroidPostingCodebookSize,
														   "pgturbohybrid multivector centroid posting write cursors"));
				docPostingCounts =
					palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
														   meta->tqMultivectorDocCount,
														   "pgturbohybrid multivector centroid posting coverage"));
				storage->multivectorCentroidPostings =
					palloc0(PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridGraphMultiVectorCentroidPostingEntry),
														   Max(centroidPostingScratchCount, 1U),
														   "pgturbohybrid multivector centroid postings"));
				MemoryContextSwitchTo(oldCtx);

				listOffsets[0] = 0;
				for (uint32 codeword = 0; codeword < centroidPostingCodebookSize;
					 codeword++)
				{
					CHECK_FOR_INTERRUPTS();
					listOffsets[codeword + 1] =
						listOffsets[codeword] + centroidPostingListCounts[codeword];
					listWrite[codeword] = listOffsets[codeword];
				}
				for (uint32 postingIndex = 0;
					 postingIndex < centroidPostingScratchCount;
					 postingIndex++)
				{
					PgturbohybridGraphMultiVectorCentroidPostingEntry *entry =
						&centroidPostingScratch[postingIndex];
					uint32		codeword =
						centroidPostingScratchCodewords[postingIndex];
					PgturbohybridMultiVector *centroids;
					uint32		offset;

					CHECK_FOR_INTERRUPTS();
					if (codeword >= centroidPostingCodebookSize)
						PgturbohybridGraphMultiVectorDocMapError(index,
																 "centroid posting entry has invalid codeword");
					centroids = storage->multivectorDocCentroids[entry->docId];
					if (centroids == NULL ||
						entry->centroidOrdinal >= (uint32) centroids->count)
						PgturbohybridGraphMultiVectorDocMapError(index,
																 "centroid posting entry is invalid");
					docPostingCounts[entry->docId]++;
					offset = listWrite[codeword]++;
					storage->multivectorCentroidPostings[offset] = *entry;
				}
				for (uint32 docId = 0; docId < meta->tqMultivectorDocCount; docId++)
				{
					PgturbohybridMultiVector *centroids =
						storage->multivectorDocCentroids[docId];

					CHECK_FOR_INTERRUPTS();
					if (docPostingCounts[docId] != (uint32) centroids->count)
						PgturbohybridGraphMultiVectorDocMapError(index,
																 "centroid posting sidecar does not cover every document centroid");
				}
				storage->multivectorCentroidPostingListOffsets = listOffsets;
					storage->multivectorCentroidPostingCodebookSize =
						centroidPostingCodebookSize;
					storage->multivectorCentroidPostingCount =
						centroidPostingScratchCount;
					storage->multivectorCentroidPostingsLoaded = true;
					if (hasCentroidDocCodes)
						PgturbohybridGraphMaterializeCentroidDocCodeMap(index,
																		 meta,
																		 storage,
																		 centroidDocCodeCounts,
																		 centroidDocCodeScratchDocIds,
																		 centroidDocCodeScratchCodes,
																		 centroidDocCodeScratchCount,
																		 centroidPostingCodebookSize);
					else
						PgturbohybridGraphBuildCentroidDocCodeMap(index,
																  meta,
																  storage,
																  centroidPostingScratch,
																  centroidPostingScratchCodewords,
																  centroidPostingScratchCount,
																  centroidPostingCodebookSize);
				pfree(listWrite);
				pfree(docPostingCounts);
			}
		}
		else if (hasCentroidPostings)
		{
			uint32	   *listOffsets;
			uint32	   *listWrite;

			oldCtx = MemoryContextSwitchTo(storage->ctx);
			listOffsets =
				palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
													   (Size) centroidPostingCodebookSize + 1,
													   "pgturbohybrid multivector centroid posting offsets"));
			listWrite =
				palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
													   centroidPostingCodebookSize,
													   "pgturbohybrid multivector centroid posting write cursors"));
			storage->multivectorCentroidPostings =
				palloc0(PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridGraphMultiVectorCentroidPostingEntry),
													   Max(centroidPostingScratchCount, 1U),
													   "pgturbohybrid multivector centroid postings"));
			MemoryContextSwitchTo(oldCtx);

			listOffsets[0] = 0;
			for (uint32 codeword = 0; codeword < centroidPostingCodebookSize;
				 codeword++)
			{
				CHECK_FOR_INTERRUPTS();
				listOffsets[codeword + 1] =
					listOffsets[codeword] + centroidPostingListCounts[codeword];
				listWrite[codeword] = listOffsets[codeword];
			}
			for (uint32 postingIndex = 0;
				 postingIndex < centroidPostingScratchCount;
				 postingIndex++)
			{
				PgturbohybridGraphMultiVectorCentroidPostingEntry *entry =
					&centroidPostingScratch[postingIndex];
				uint32		codeword =
					centroidPostingScratchCodewords[postingIndex];
				uint32		offset;

				CHECK_FOR_INTERRUPTS();
				if (codeword >= centroidPostingCodebookSize)
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "centroid posting entry has invalid codeword");
				offset = listWrite[codeword]++;
				storage->multivectorCentroidPostings[offset] = *entry;
			}
			storage->multivectorCentroidPostingListOffsets = listOffsets;
				storage->multivectorCentroidPostingCodebookSize =
					centroidPostingCodebookSize;
				storage->multivectorCentroidPostingCount =
					centroidPostingScratchCount;
				storage->multivectorCentroidPostingsLoaded = true;
				if (hasCentroidDocCodes)
					PgturbohybridGraphMaterializeCentroidDocCodeMap(index,
																	 meta,
																	 storage,
																	 centroidDocCodeCounts,
																	 centroidDocCodeScratchDocIds,
																	 centroidDocCodeScratchCodes,
																	 centroidDocCodeScratchCount,
																	 centroidPostingCodebookSize);
				else
					PgturbohybridGraphBuildCentroidDocCodeMap(index,
															  meta,
															  storage,
															  centroidPostingScratch,
															  centroidPostingScratchCodewords,
															  centroidPostingScratchCount,
															  centroidPostingCodebookSize);
			pfree(listWrite);
		}
	}

	storage->multivectorDocCount = meta->tqMultivectorDocCount;
	storage->multivectorDocMapBytes = meta->tqMultivectorDocMapBytes;
	storage->multivectorDocMapLoaded = true;
	pfree(nodeSeen);
	pfree(docSeen);
	if (vectorFloatCounts != NULL)
		pfree(vectorFloatCounts);
	if (centroidFloatCounts != NULL)
		pfree(centroidFloatCounts);
	if (centroidPostingListCounts != NULL)
		pfree(centroidPostingListCounts);
	if (centroidPostingScratch != NULL)
		pfree(centroidPostingScratch);
	if (centroidPostingScratchCodewords != NULL)
		pfree(centroidPostingScratchCodewords);
	if (centroidDocCodeCounts != NULL)
		pfree(centroidDocCodeCounts);
	if (centroidDocCodeScratchDocIds != NULL)
		pfree(centroidDocCodeScratchDocIds);
	if (centroidDocCodeScratchCodes != NULL)
		pfree(centroidDocCodeScratchCodes);
	if (quantizedPostingListCounts != NULL)
		pfree(quantizedPostingListCounts);
	if (quantizedPostingScratch != NULL)
		pfree(quantizedPostingScratch);
	if (quantizedPostingScratchCodewords != NULL)
		pfree(quantizedPostingScratchCodewords);
	return true;
}

bool
PgturbohybridGraphLoadCentroidLiteCompactDocMapWithStats(Relation index,
										 PgturbohybridGraphMetaPageData *meta,
										 PgturbohybridGraphScanStorage *storage,
										 const bool *selectedCodewords,
										 uint32 codebookSize,
										 bool requireFullDocCodes,
										 PgturbohybridMultiVectorDocSidecarAccessStats *stats)
{
	MemoryContext oldCtx;
	bool	   *docSeen;
	uint32	   *listCounts;
	uint32	   *listWrite;
	PgturbohybridGraphMultiVectorCentroidPostingEntry *postingScratch = NULL;
	uint32	   *postingScratchCodewords = NULL;
	uint32		postingScratchCount = 0;
	uint32		postingScratchCapacity = 0;
	uint32	   *docCodeCounts = NULL;
	uint32	   *docCodeScratchDocIds = NULL;
	uint32	   *docCodeScratchCodes = NULL;
	uint32		docCodeScratchCount = 0;
	uint32		docCodeScratchCapacity = 0;
	uint32		docsSeen = 0;
	uint32		selectedCount = 0;
	uint32		maxSelectedCodeword = 0;
	BlockNumber blkno;
	bool		documentNodes =
		meta->tqMultivectorGraphMode ==
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES;
	bool		hasCentroidPostings =
		documentNodes &&
		(meta->tqMultivectorDocMapFlags &
		 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CENTROIDS) != 0 &&
		(meta->tqMultivectorDocMapFlags &
		 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CENTROID_POSTINGS) != 0;
	bool		hasCentroidDocCodes =
		hasCentroidPostings &&
		(meta->tqMultivectorDocMapFlags &
		 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CENTROID_DOC_CODES) != 0;

	if (!documentNodes || !hasCentroidPostings ||
		codebookSize != (uint32) meta->dimensions * 2U ||
		selectedCodewords == NULL)
		PgturbohybridGraphMultiVectorDocMapError(index,
												 "centroid_lite compact sidecar request is invalid");
	if (requireFullDocCodes && !hasCentroidDocCodes)
		PgturbohybridGraphMultiVectorDocMapError(index,
												 "centroid_lite codeword MaxSim requires a persisted centroid doc-code sidecar");
	if (!BlockNumberIsValid(meta->tqMultivectorDocMapStartBlkno))
		PgturbohybridGraphMultiVectorDocMapError(index,
												 "multivector docmap sidecar is not available");
	if (meta->tqMultivectorDocMapVersion !=
		PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION)
		PgturbohybridGraphMultiVectorDocMapError(index,
												 "unsupported docmap sidecar version");
	if (storage->multivectorDocMapLoaded)
	{
		if (stats != NULL)
			stats->cacheHits++;
		return storage->multivectorCentroidPostingsLoaded &&
			(!requireFullDocCodes ||
			 (storage->multivectorCentroidDocCodesLoaded &&
			  storage->multivectorCentroidDocCodeOffsets != NULL &&
			  storage->multivectorCentroidDocCodes != NULL));
	}

	for (uint32 codeword = 0; codeword < codebookSize; codeword++)
	{
		if (selectedCodewords[codeword])
		{
			selectedCount++;
			maxSelectedCodeword = codeword;
		}
	}

	oldCtx = MemoryContextSwitchTo(storage->ctx);
	storage->multivectorDocMap =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(TqMultiVectorDocMapEntry),
											   meta->tqMultivectorDocCount,
											   "pgturbohybrid multivector docmap"));
	docSeen =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(bool),
											   meta->tqMultivectorDocCount,
											   "pgturbohybrid multivector docmap seen"));
	listCounts =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
											   codebookSize,
											   "pgturbohybrid multivector centroid posting counts"));
	if (hasCentroidDocCodes)
		docCodeCounts =
			palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
												   meta->tqMultivectorDocCount,
												   "pgturbohybrid multivector centroid doc-code counts"));
	MemoryContextSwitchTo(oldCtx);

	blkno = meta->tqMultivectorDocMapStartBlkno;
	for (uint32 pageNo = 0; pageNo < meta->tqMultivectorDocMapPageCount; pageNo++)
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;
		BlockNumber nextblkno;
		bool		stopAfterPage = false;
		instr_time	pageReadStart;

		CHECK_FOR_INTERRUPTS();
		if (!BlockNumberIsValid(blkno))
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "docmap page chain ended early");
		if (stats != NULL)
			INSTR_TIME_SET_CURRENT(pageReadStart);
		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = PgturbohybridGraphPageGetOpaque(page);
		if (stats != NULL)
		{
			stats->pagesRead++;
			stats->docMapPagesRead++;
			stats->cacheMisses++;
			stats->pageReadUs +=
				(uint64) PgturbohybridGraphElapsedUsSince(pageReadStart);
		}
		if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) !=
			PGTURBOHYBRID_GRAPH_PAGE_KIND_MULTIVECTOR_DOCMAP)
		{
			UnlockReleaseBuffer(buf);
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "docmap page kind mismatch");
		}

		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff;
			 offno = OffsetNumberNext(offno))
		{
			ItemId		iid = PageGetItemId(page, offno);
			Item		item = PageGetItem(page, iid);
			Size		itemSize = ItemIdGetLength(iid);
			uint8		type = *((uint8 *) item);

			CHECK_FOR_INTERRUPTS();
			if (stats != NULL)
			{
				stats->bytesTouched += itemSize;
				stats->docMapBytesTouched += itemSize;
			}

				if (type ==
					PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_DOC_TUPLE_TYPE)
				{
					PgturbohybridGraphMultiVectorDocMapDocTuple tuple =
						(PgturbohybridGraphMultiVectorDocMapDocTuple) item;
				Size		tupleSize;

				if (itemSize < offsetof(PgturbohybridGraphMultiVectorDocMapDocTupleData,
										entries) ||
					tuple->magic != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC ||
					tuple->version != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION ||
					tuple->count == 0)
				{
					UnlockReleaseBuffer(buf);
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "malformed document docmap tuple");
				}
				tupleSize =
					PgturbohybridGraphMultiVectorDocMapDocTupleSize(tuple->count);
				if (itemSize < tupleSize ||
					(uint64) tuple->firstDocId + tuple->count >
					meta->tqMultivectorDocCount)
				{
					UnlockReleaseBuffer(buf);
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "document docmap tuple range is invalid");
				}
				for (uint16 i = 0; i < tuple->count; i++)
				{
					uint32		docId = tuple->firstDocId + i;
					TqMultiVectorDocMapEntry *entry = &tuple->entries[i];

					CHECK_FOR_INTERRUPTS();
					if (docSeen[docId] ||
						entry->tokenCount == 0 ||
						entry->firstNodeId >= meta->tqNodeCount)
					{
						UnlockReleaseBuffer(buf);
						PgturbohybridGraphMultiVectorDocMapError(index,
																 "document docmap entry is invalid");
					}
					docSeen[docId] = true;
						storage->multivectorDocMap[docId] = *entry;
						docsSeen++;
					}
				}
				else if (type ==
						 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_CENTROID_DOC_CODE_TUPLE_TYPE)
				{
					PgturbohybridGraphMultiVectorDocMapCentroidDocCodeTuple tuple =
						(PgturbohybridGraphMultiVectorDocMapCentroidDocCodeTuple) item;
					Size		tupleSize;

					if (!hasCentroidDocCodes)
					{
						UnlockReleaseBuffer(buf);
						PgturbohybridGraphMultiVectorDocMapError(index,
																 "centroid doc-code tuple found without doc-code storage flag");
					}
					if (itemSize < offsetof(PgturbohybridGraphMultiVectorDocMapCentroidDocCodeTupleData,
											codes) ||
						tuple->magic != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC ||
						tuple->version != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION ||
						tuple->count == 0 ||
						tuple->docId >= meta->tqMultivectorDocCount ||
						tuple->codebookSize != codebookSize ||
						tuple->startCode != docCodeCounts[tuple->docId])
					{
						UnlockReleaseBuffer(buf);
						PgturbohybridGraphMultiVectorDocMapError(index,
																 "malformed centroid doc-code tuple");
					}
					tupleSize =
						PgturbohybridGraphMultiVectorDocMapCentroidDocCodeTupleSize(tuple->count);
					if (itemSize < tupleSize)
					{
						UnlockReleaseBuffer(buf);
						PgturbohybridGraphMultiVectorDocMapError(index,
																 "centroid doc-code tuple range is invalid");
					}
					for (uint16 i = 0; i < tuple->count; i++)
					{
						CHECK_FOR_INTERRUPTS();
						if (tuple->codes[i] >= codebookSize)
						{
							UnlockReleaseBuffer(buf);
							PgturbohybridGraphMultiVectorDocMapError(index,
																	 "centroid doc-code tuple codeword is invalid");
						}
					}
					PgturbohybridGraphAppendCentroidDocCodeScratch(index,
																   storage->ctx,
																   tuple->docId,
																   tuple->codes,
																   tuple->count,
																   &docCodeScratchDocIds,
																   &docCodeScratchCodes,
																   &docCodeScratchCount,
																   &docCodeScratchCapacity);
					docCodeCounts[tuple->docId] += tuple->count;
				}
				else if (type ==
						 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_CENTROID_POSTING_TUPLE_TYPE)
				{
				PgturbohybridGraphMultiVectorDocMapCentroidPostingTuple tuple =
					(PgturbohybridGraphMultiVectorDocMapCentroidPostingTuple) item;
				Size		tupleSize;

				if (itemSize < offsetof(PgturbohybridGraphMultiVectorDocMapCentroidPostingTupleData,
										entries) ||
					tuple->magic != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC ||
					tuple->version != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION ||
					tuple->count == 0 ||
					tuple->codeword >= codebookSize)
				{
					UnlockReleaseBuffer(buf);
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "malformed centroid posting tuple");
				}
				tupleSize =
					PgturbohybridGraphMultiVectorDocMapCentroidPostingTupleSize(tuple->count);
				if (itemSize < tupleSize)
				{
					UnlockReleaseBuffer(buf);
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "centroid posting tuple range is invalid");
				}
					if (!requireFullDocCodes && selectedCount == 0)
					{
						if (docsSeen == meta->tqMultivectorDocCount)
						{
						stopAfterPage = true;
						break;
					}
					continue;
				}
					if (!requireFullDocCodes && selectedCount > 0 &&
						tuple->codeword > maxSelectedCodeword)
					{
						if (docsSeen == meta->tqMultivectorDocCount)
					{
						stopAfterPage = true;
						break;
					}
					continue;
				}
				if (!selectedCodewords[tuple->codeword])
					continue;
				if ((uint64) postingScratchCount + tuple->count >
					(uint64) PG_UINT32_MAX)
				{
					UnlockReleaseBuffer(buf);
					PgturbohybridGraphMultiVectorDocMapError(index,
															 "centroid posting sidecar is too large");
				}
				if (postingScratchCount + tuple->count >
					postingScratchCapacity)
				{
					uint32		oldCapacity = postingScratchCapacity;
					uint32		newCapacity =
						postingScratchCapacity == 0 ?
						1024 : postingScratchCapacity;
					Size		oldPostingBytes;
					Size		newPostingBytes;
					Size		oldCodewordBytes;
					Size		newCodewordBytes;

					while (newCapacity < postingScratchCount + tuple->count)
					{
						if (newCapacity > PG_UINT32_MAX / 2)
						{
							UnlockReleaseBuffer(buf);
							PgturbohybridGraphMultiVectorDocMapError(index,
																	 "centroid posting sidecar is too large");
						}
						newCapacity += newCapacity;
					}
					oldPostingBytes =
						PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridGraphMultiVectorCentroidPostingEntry),
													   oldCapacity,
													   "pgturbohybrid multivector centroid posting scratch");
					newPostingBytes =
						PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridGraphMultiVectorCentroidPostingEntry),
													   newCapacity,
													   "pgturbohybrid multivector centroid posting scratch");
					oldCodewordBytes =
						PgturbohybridCheckedArrayBytes(sizeof(uint32),
													   oldCapacity,
													   "pgturbohybrid multivector centroid posting codewords");
					newCodewordBytes =
						PgturbohybridCheckedArrayBytes(sizeof(uint32),
													   newCapacity,
													   "pgturbohybrid multivector centroid posting codewords");
					if (postingScratch == NULL)
					{
						postingScratch =
							MemoryContextAllocZero(storage->ctx,
												   newPostingBytes);
						postingScratchCodewords =
							MemoryContextAllocZero(storage->ctx,
												   newCodewordBytes);
					}
					else
					{
						postingScratch =
							repalloc(postingScratch, newPostingBytes);
						memset((char *) postingScratch + oldPostingBytes, 0,
							   newPostingBytes - oldPostingBytes);
						postingScratchCodewords =
							repalloc(postingScratchCodewords,
									 newCodewordBytes);
						memset((char *) postingScratchCodewords +
							   oldCodewordBytes, 0,
							   newCodewordBytes - oldCodewordBytes);
					}
					postingScratchCapacity = newCapacity;
				}
				for (uint16 i = 0; i < tuple->count; i++)
				{
					PgturbohybridGraphMultiVectorCentroidPostingEntry *entry =
						&tuple->entries[i];

					CHECK_FOR_INTERRUPTS();
					if (entry->docId >= meta->tqMultivectorDocCount)
					{
						UnlockReleaseBuffer(buf);
						PgturbohybridGraphMultiVectorDocMapError(index,
																 "centroid posting entry is invalid");
					}
					postingScratch[postingScratchCount] = *entry;
					postingScratchCodewords[postingScratchCount] =
						tuple->codeword;
					listCounts[tuple->codeword]++;
					postingScratchCount++;
				}
			}
		}

		nextblkno = opaque->nextblkno;
		UnlockReleaseBuffer(buf);
		blkno = nextblkno;
		if (stopAfterPage)
			break;
	}

	if (docsSeen != meta->tqMultivectorDocCount)
		PgturbohybridGraphMultiVectorDocMapError(index,
												 "docmap sidecar does not cover every document");

	oldCtx = MemoryContextSwitchTo(storage->ctx);
	storage->multivectorCentroidPostingListOffsets =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
											   (Size) codebookSize + 1,
											   "pgturbohybrid multivector centroid posting offsets"));
	listWrite =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
											   codebookSize,
											   "pgturbohybrid multivector centroid posting write cursors"));
	storage->multivectorCentroidPostings =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridGraphMultiVectorCentroidPostingEntry),
											   Max(postingScratchCount, 1U),
											   "pgturbohybrid multivector centroid postings"));
	MemoryContextSwitchTo(oldCtx);

	storage->multivectorCentroidPostingListOffsets[0] = 0;
	for (uint32 codeword = 0; codeword < codebookSize; codeword++)
	{
		storage->multivectorCentroidPostingListOffsets[codeword + 1] =
			storage->multivectorCentroidPostingListOffsets[codeword] +
			listCounts[codeword];
		listWrite[codeword] =
			storage->multivectorCentroidPostingListOffsets[codeword];
	}
	for (uint32 postingIndex = 0; postingIndex < postingScratchCount;
		 postingIndex++)
	{
		uint32		codeword = postingScratchCodewords[postingIndex];
		uint32		offset;

		CHECK_FOR_INTERRUPTS();
		if (codeword >= codebookSize || !selectedCodewords[codeword])
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "centroid posting entry has invalid codeword");
		offset = listWrite[codeword]++;
		storage->multivectorCentroidPostings[offset] =
			postingScratch[postingIndex];
	}

	storage->multivectorCentroidPostingCodebookSize = codebookSize;
	storage->multivectorCentroidPostingCount = postingScratchCount;
	storage->multivectorCentroidPostingsLoaded = true;
	if (hasCentroidDocCodes)
		PgturbohybridGraphMaterializeCentroidDocCodeMap(index,
														 meta,
														 storage,
														 docCodeCounts,
														 docCodeScratchDocIds,
														 docCodeScratchCodes,
														 docCodeScratchCount,
														 codebookSize);
	else if (requireFullDocCodes)
		PgturbohybridGraphMultiVectorDocMapError(index,
												 "centroid_lite codeword MaxSim requires a persisted centroid doc-code sidecar");
	storage->multivectorDocCount = meta->tqMultivectorDocCount;
	storage->multivectorDocMapBytes = meta->tqMultivectorDocMapBytes;
	storage->multivectorDocMapLoaded = true;
	storage->multivectorDocCentroidVectorsSkipped = true;
	pfree(docSeen);
	pfree(listCounts);
	pfree(listWrite);
	if (docCodeCounts != NULL)
		pfree(docCodeCounts);
	if (docCodeScratchDocIds != NULL)
		pfree(docCodeScratchDocIds);
	if (docCodeScratchCodes != NULL)
		pfree(docCodeScratchCodes);
	if (postingScratch != NULL)
		pfree(postingScratch);
	if (postingScratchCodewords != NULL)
		pfree(postingScratchCodewords);
	return true;
}

static bool
PgturbohybridGraphDocSidecarCacheMatches(PgturbohybridGraphDocSidecarCache *cache,
										 Relation index,
										 PgturbohybridGraphMetaPageData *meta,
										 bool pagedDocVectors,
										 bool contextsSkipped)
{
	return cache->relid == RelationGetRelid(index) &&
		cache->relfilenumber == PgturbohybridGraphRelFileNumber(index) &&
		cache->dimensions == meta->dimensions &&
		cache->tqNodeCount == meta->tqNodeCount &&
		cache->tqMultivectorDocMapStartBlkno == meta->tqMultivectorDocMapStartBlkno &&
		cache->tqMultivectorDocMapPageCount == meta->tqMultivectorDocMapPageCount &&
		cache->tqMultivectorDocCount == meta->tqMultivectorDocCount &&
		cache->tqMultivectorDocMapBytes == meta->tqMultivectorDocMapBytes &&
		cache->tqMultivectorDocMapVersion == meta->tqMultivectorDocMapVersion &&
		cache->tqMultivectorDocMapFlags == meta->tqMultivectorDocMapFlags &&
		cache->tqMultivectorGraphMode == meta->tqMultivectorGraphMode &&
		cache->pagedDocVectors == pagedDocVectors &&
		cache->contextsSkipped == contextsSkipped;
}

static void
PgturbohybridGraphCopyDocSidecarStorage(PgturbohybridGraphScanStorage *dest,
										PgturbohybridGraphScanStorage *src)
{
	dest->multivectorNodeMap = src->multivectorNodeMap;
	dest->multivectorDocMap = src->multivectorDocMap;
	dest->multivectorDocVectors = src->multivectorDocVectors;
	dest->multivectorDocVectorChunks = src->multivectorDocVectorChunks;
	dest->multivectorDocVectorFirstChunk = src->multivectorDocVectorFirstChunk;
	dest->multivectorDocVectorChunkCounts = src->multivectorDocVectorChunkCounts;
	dest->multivectorDocVectorChunkCount = src->multivectorDocVectorChunkCount;
	dest->multivectorDocVectorChunkCapacity = src->multivectorDocVectorChunkCapacity;
	dest->multivectorDocCentroids = src->multivectorDocCentroids;
	dest->multivectorDocCentroidResiduals = src->multivectorDocCentroidResiduals;
	dest->multivectorCentroidPostings = src->multivectorCentroidPostings;
	dest->multivectorCentroidPostingListOffsets = src->multivectorCentroidPostingListOffsets;
	dest->multivectorCentroidDocCodeOffsets = src->multivectorCentroidDocCodeOffsets;
	dest->multivectorCentroidDocCodes = src->multivectorCentroidDocCodes;
	dest->multivectorCentroidPostingCodebookSize = src->multivectorCentroidPostingCodebookSize;
	dest->multivectorCentroidPostingCount = src->multivectorCentroidPostingCount;
	dest->multivectorCentroidDocCodeCount = src->multivectorCentroidDocCodeCount;
	dest->multivectorQuantizedInvertedPostings = src->multivectorQuantizedInvertedPostings;
	dest->multivectorQuantizedInvertedListOffsets = src->multivectorQuantizedInvertedListOffsets;
	dest->multivectorQuantizedInvertedDocCodeOffsets =
		src->multivectorQuantizedInvertedDocCodeOffsets;
	dest->multivectorQuantizedInvertedDocCodes =
		src->multivectorQuantizedInvertedDocCodes;
	dest->multivectorQuantizedInvertedCodebookSize = src->multivectorQuantizedInvertedCodebookSize;
	dest->multivectorQuantizedInvertedCodebookDim = src->multivectorQuantizedInvertedCodebookDim;
	dest->multivectorQuantizedInvertedCodebookTopM = src->multivectorQuantizedInvertedCodebookTopM;
	dest->multivectorQuantizedInvertedCodebookSource = src->multivectorQuantizedInvertedCodebookSource;
	strlcpy(dest->multivectorQuantizedInvertedCodebookChecksum,
			src->multivectorQuantizedInvertedCodebookChecksum,
			sizeof(dest->multivectorQuantizedInvertedCodebookChecksum));
	dest->multivectorQuantizedInvertedPostingCount = src->multivectorQuantizedInvertedPostingCount;
	dest->multivectorQuantizedInvertedDocCodeCount =
		src->multivectorQuantizedInvertedDocCodeCount;
	dest->multivectorDocCount = src->multivectorDocCount;
	dest->multivectorDocMapBytes = src->multivectorDocMapBytes;
	dest->multivectorDocVectorsLoaded = src->multivectorDocVectorsLoaded;
	dest->multivectorDocVectorsPaged = src->multivectorDocVectorsPaged;
	dest->multivectorDocContextsSkipped = src->multivectorDocContextsSkipped;
	dest->multivectorDocCentroidVectorsSkipped =
		src->multivectorDocCentroidVectorsSkipped;
	dest->multivectorDocCentroidsLoaded = src->multivectorDocCentroidsLoaded;
	dest->multivectorCentroidPostingsLoaded = src->multivectorCentroidPostingsLoaded;
	dest->multivectorCentroidDocCodesLoaded = src->multivectorCentroidDocCodesLoaded;
	dest->multivectorQuantizedInvertedPostingsLoaded =
		src->multivectorQuantizedInvertedPostingsLoaded;
	dest->multivectorQuantizedInvertedDocCodesLoaded =
		src->multivectorQuantizedInvertedDocCodesLoaded;
	dest->multivectorDocMapLoaded = src->multivectorDocMapLoaded;
}

static PgturbohybridGraphDocSidecarCache *
PgturbohybridGraphFindDocSidecarCache(Relation index,
									  PgturbohybridGraphMetaPageData *meta,
									  bool pagedDocVectors,
									  bool contextsSkipped)
{
	PgturbohybridGraphDocSidecarCache **link =
		&pgturbohybridGraphDocSidecarCacheList;

	while (*link != NULL)
	{
		PgturbohybridGraphDocSidecarCache *cache = *link;

		if (cache->relid == RelationGetRelid(index) &&
			!PgturbohybridGraphDocSidecarCacheMatches(cache, index, meta,
													  cache->pagedDocVectors,
													  cache->contextsSkipped))
		{
			*link = cache->next;
			MemoryContextDelete(cache->ctx);
			continue;
		}
		if (PgturbohybridGraphDocSidecarCacheMatches(cache, index, meta,
													 pagedDocVectors,
													 contextsSkipped))
			return cache;

		link = &cache->next;
	}

	return NULL;
}

static PgturbohybridGraphDocSidecarCache *
PgturbohybridGraphBuildDocSidecarCache(Relation index,
									   PgturbohybridGraphMetaPageData *meta,
									   bool pagedDocVectors,
									   bool contextsSkipped)
{
	MemoryContext cacheCtx;
	MemoryContext oldCtx;
	PgturbohybridGraphDocSidecarCache *cache;
	PgturbohybridMultiVectorDocSidecarAccessStats stats;
	instr_time	buildStart;
	instr_time	buildElapsed;

	INSTR_TIME_SET_CURRENT(buildStart);
	memset(&stats, 0, sizeof(stats));

	cacheCtx = AllocSetContextCreate(CacheMemoryContext,
									 "pgturbohybrid doc sidecar cache",
									 ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(cacheCtx);
	cache = palloc0(sizeof(PgturbohybridGraphDocSidecarCache));
	cache->relid = RelationGetRelid(index);
	cache->relfilenumber = PgturbohybridGraphRelFileNumber(index);
	cache->dimensions = meta->dimensions;
	cache->tqNodeCount = meta->tqNodeCount;
	cache->tqMultivectorDocMapStartBlkno =
		meta->tqMultivectorDocMapStartBlkno;
	cache->tqMultivectorDocMapPageCount =
		meta->tqMultivectorDocMapPageCount;
	cache->tqMultivectorDocCount = meta->tqMultivectorDocCount;
	cache->tqMultivectorDocMapBytes = meta->tqMultivectorDocMapBytes;
	cache->tqMultivectorDocMapVersion =
		meta->tqMultivectorDocMapVersion;
	cache->tqMultivectorDocMapFlags = meta->tqMultivectorDocMapFlags;
	cache->tqMultivectorGraphMode = meta->tqMultivectorGraphMode;
	cache->pagedDocVectors = pagedDocVectors;
	cache->contextsSkipped = contextsSkipped;
	cache->ctx = cacheCtx;
	cache->storage.ctx = cacheCtx;
	cache->storage.multivectorDocVectorsPaged = pagedDocVectors;
	cache->storage.multivectorDocContextsSkipped = contextsSkipped;

	if (!PgturbohybridGraphLoadMultiVectorDocMapWithStats(index, meta,
														  &cache->storage,
														  true,
														  &stats))
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
				 errmsg("document-node multivector sidecar cache could not be built"),
				 errhint("REINDEX the index to rebuild the document-node multivector sidecar.")));
	cache->buildBytesTouched = stats.bytesTouched;
	cache->buildPagesRead = stats.pagesRead;
	cache->next = pgturbohybridGraphDocSidecarCacheList;
	pgturbohybridGraphDocSidecarCacheList = cache;
	MemoryContextSwitchTo(oldCtx);

	INSTR_TIME_SET_CURRENT(buildElapsed);
	INSTR_TIME_SUBTRACT(buildElapsed, buildStart);
	cache->buildUs = (int64) INSTR_TIME_GET_MICROSEC(buildElapsed);
	return cache;
}

bool
PgturbohybridGraphAttachMultiVectorDocSidecarCache(Relation index,
									  PgturbohybridGraphMetaPageData *meta,
									  PgturbohybridGraphScanStorage *storage,
									  bool pagedDocVectors,
									  bool contextsSkipped,
									  PgturbohybridGraphCacheInitInfo *info)
{
	PgturbohybridGraphDocSidecarCache *cache;

	if (index == NULL || meta == NULL || storage == NULL ||
		!BlockNumberIsValid(meta->tqMultivectorDocMapStartBlkno) ||
		meta->tqMultivectorDocMapVersion !=
		PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION)
		return false;
	if (pgturbohybrid_native_cache_policy ==
		PGTURBOHYBRID_NATIVE_CACHE_POLICY_OFF)
		return false;

	cache = PgturbohybridGraphFindDocSidecarCache(index, meta,
												  pagedDocVectors,
												  contextsSkipped);
	if (cache == NULL)
	{
		cache = PgturbohybridGraphBuildDocSidecarCache(index, meta,
													   pagedDocVectors,
													   contextsSkipped);
		if (info != NULL)
		{
			info->docSidecarCacheBuiltThisScan = true;
			info->docSidecarCacheBuildUs = cache->buildUs;
			info->docSidecarCacheBytesTouched = cache->buildBytesTouched;
			info->docSidecarCachePagesRead = cache->buildPagesRead;
		}
	}
	else if (info != NULL)
		info->docSidecarCacheReused = true;

	PgturbohybridGraphCopyDocSidecarStorage(storage, &cache->storage);
	if (info != NULL)
	{
		info->docSidecarCacheUsed = true;
		info->docMapBytes = (int64) cache->storage.multivectorDocMapBytes;
	}
	return true;
}

bool
PgturbohybridGraphLoadCodePage(Relation index, PgturbohybridGraphScanOpaque so, PgturbohybridGraphMetaPageData *meta,
					PgturbohybridGraphScanStorage *storage, uint32 nodeId)
{
	int			pageNo;
	BlockNumber blkno;

	if (nodeId >= meta->tqNodeCount || !BlockNumberIsValid(meta->tqCodeStartBlkno))
		return false;

	/* Cache hit: code already resident (cross-scan native cache or this scan). */
	if (storage->cached && storage->nodes[nodeId].loaded)
		return storage->nodes[nodeId].loaded;

	pageNo = nodeId / storage->codeTuplesPerPage;
	if (pageNo < 0 || pageNo >= storage->codePageCount)
		return false;

	/* Cache hit: this candidate's code page was already loaded this scan. */
	if (storage->codePagesLoaded[pageNo])
		return storage->nodes[nodeId].loaded;

	blkno = PgturbohybridGraphGetMappedBlockNumber(meta->tqCodeStartBlkno, pageNo,
										 storage->codeBlknos);

	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;
		bool		fallbackTried = false;
		instr_time	lockStart;

retry:

		CHECK_FOR_INTERRUPTS();
		buf = ReadBuffer(index, blkno);
		INSTR_TIME_SET_CURRENT(lockStart);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		if (so != NULL)
			so->graphCodeBufferLockWaitUs +=
				PgturbohybridGraphElapsedUsSince(lockStart);
		page = BufferGetPage(buf);
		opaque = PgturbohybridGraphPageGetOpaque(page);
		if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) != PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_CODE)
		{
			UnlockReleaseBuffer(buf);
			if (!fallbackTried)
			{
				fallbackTried = true;
				if (!PgturbohybridGraphResolveChainBlockNumber(index, meta->tqCodeStartBlkno,
													 pageNo, storage->codePageCount,
													 PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_CODE,
													 storage->codeBlknos, &blkno))
					return false;
				goto retry;
			}
			return false;
		}

		if (so != NULL)
			so->graphCodePagesRead++;
		maxoff = PageGetMaxOffsetNumber(page);
		{
			bool		tqWeighted = (meta->tqFlags & PGTURBOHYBRID_GRAPH_TQ_WEIGHTED) != 0;

			for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
			{
				ItemId		iid = PageGetItemId(page, offno);
				PgturbohybridGraphCodeTuple tuple = (PgturbohybridGraphCodeTuple) PageGetItem(page, iid);
				PgturbohybridGraphScanNode *node;

				if (tuple->type != PGTURBOHYBRID_GRAPH_CODE_TUPLE_TYPE ||
					tuple->nodeId >= meta->tqNodeCount)
					continue;

				node = &storage->nodes[tuple->nodeId];
				node->heaptid = tuple->heaptid;
				node->level = tuple->level;
				node->exactBlkno = tuple->exactBlkno;
				node->exactOffno = tuple->exactOffno;
				node->payloadMask = tuple->payloadMask;
				node->scale = tuple->scale;
				node->norm = tuple->norm;
				node->codeNorm = tuple->correction;
				node->ecCorrection = PgturbohybridGraphTupleEcCorrection(tuple, tqWeighted);
				node->flags = tuple->flags;
				if (meta->tqPayloadBytes > 0)
				{
					if (storage->payloadArena != NULL)
						node->payloads = (int32 *) (storage->payloadArena +
													((Size) tuple->nodeId * meta->tqPayloadBytes));
					else if (node->payloads == NULL)
						node->payloads = (int32 *) MemoryContextAlloc(storage->ctx,
																	  meta->tqPayloadBytes);
					memcpy(node->payloads, PgturbohybridGraphTuplePayloads(tuple, tqWeighted),
						   meta->tqPayloadBytes);
				}
				if (meta->tqCodeBytes > 0)
				{
					if (storage->codeArena != NULL)
						node->code = storage->codeArena + ((Size) tuple->nodeId * meta->tqCodeBytes);
					else if (node->code == NULL)
						node->code = (uint8 *) MemoryContextAlloc(storage->ctx,
																  meta->tqCodeBytes);
					memcpy(node->code, PgturbohybridGraphTupleCode(tuple, meta->tqPayloadBytes,
																   meta->tqResidualRerankBytes,
																   tqWeighted),
						   meta->tqCodeBytes);
				}
				if (meta->tqResidualRerankBytes > 0)
				{
					if (storage->residualArena != NULL)
						node->residualSketch = storage->residualArena +
							((Size) tuple->nodeId * meta->tqResidualRerankBytes);
					else if (node->residualSketch == NULL)
						node->residualSketch = (uint8 *) MemoryContextAlloc(storage->ctx,
																			meta->tqResidualRerankBytes);
					memcpy(node->residualSketch,
						   PgturbohybridGraphTupleResidual(tuple, meta->tqPayloadBytes,
														   meta->tqResidualRerankBytes,
														   tqWeighted),
						   meta->tqResidualRerankBytes);
				}
				node->loaded = true;
			}
		}

		storage->codePagesLoaded[pageNo] = true;
		UnlockReleaseBuffer(buf);
	}

	return storage->nodes[nodeId].loaded;
}

static void
PgturbohybridGraphLoadAllAdjPages(Relation index, PgturbohybridGraphScanOpaque so, PgturbohybridGraphMetaPageData *meta,
					   PgturbohybridGraphScanStorage *storage)
{
	BlockNumber blkno = meta->tqAdjStartBlkno;
	BlockNumber nblocks;

	if (!BlockNumberIsValid(blkno) || storage->adjPageCount <= 0 ||
		storage->adjPagesLoaded == NULL || storage->adjPagesLoaded[0])
		return;

	nblocks = RelationGetNumberOfBlocks(index);
	while (BlockNumberIsValid(blkno) && blkno < nblocks)
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;
		BlockNumber nextblkno;
		instr_time	lockStart;

		CHECK_FOR_INTERRUPTS();
		buf = ReadBuffer(index, blkno);
		INSTR_TIME_SET_CURRENT(lockStart);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		if (so != NULL)
			so->graphAdjBufferLockWaitUs +=
				PgturbohybridGraphElapsedUsSince(lockStart);
		page = BufferGetPage(buf);
		opaque = PgturbohybridGraphPageGetOpaque(page);
		if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) != PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_ADJ)
		{
			UnlockReleaseBuffer(buf);
			break;
		}

		if (so != NULL)
			so->graphAdjPagesRead++;
		nextblkno = opaque->nextblkno;
		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
		{
			ItemId		iid = PageGetItemId(page, offno);
			PgturbohybridGraphAdjTuple tuple;
			int			tupleSlot;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (PgturbohybridGraphAdjTuple) PageGetItem(page, iid);
			if (tuple->type != PGTURBOHYBRID_GRAPH_ADJ_TUPLE_TYPE ||
				tuple->nodeId >= meta->tqNodeCount ||
				tuple->level > meta->graphMaxLevel ||
				tuple->level >= storage->levelCount ||
				tuple->count > PgturbohybridGraphLevelM(meta->m, tuple->level))
				continue;

			tupleSlot = PgturbohybridGraphAdjSlot(meta, tuple->nodeId, tuple->level);
			storage->adjBlknos[tupleSlot] = blkno;
			storage->adjOffnos[tupleSlot] = offno;
			storage->neighborCounts[tupleSlot] = tuple->count;
			if (storage->neighbors[tupleSlot] != NULL)
			{
				pfree(storage->neighbors[tupleSlot]);
				storage->neighbors[tupleSlot] = NULL;
			}
			if (tuple->count > 0)
			{
				storage->neighbors[tupleSlot] =
					MemoryContextAlloc(storage->ctx, sizeof(uint32) * tuple->count);
				memcpy(storage->neighbors[tupleSlot], tuple->neighbors, sizeof(uint32) * tuple->count);
			}
		}

		UnlockReleaseBuffer(buf);
		if (nextblkno == blkno)
			break;
		blkno = nextblkno;
	}

	storage->adjPagesLoaded[0] = true;
}

bool
PgturbohybridGraphLoadAdjPage(Relation index, PgturbohybridGraphScanOpaque so, PgturbohybridGraphMetaPageData *meta,
				   PgturbohybridGraphScanStorage *storage, uint32 nodeId, int level)
{
	if (nodeId >= meta->tqNodeCount || level < 0 || level > meta->graphMaxLevel ||
		!BlockNumberIsValid(meta->tqAdjStartBlkno))
		return false;

	if (storage->cached)
		return true;

	PgturbohybridGraphLoadAllAdjPages(index, so, meta, storage);
	return true;
}

static bool
PgturbohybridGraphLoadExactVectors(Relation index, PgturbohybridGraphMetaPageData *meta,
						PgturbohybridGraphScanStorage *storage)
{
	if (storage->exactArena == NULL ||
		!BlockNumberIsValid(meta->tqExactStartBlkno))
		return false;

	for (uint32 nodeId = 0; nodeId < meta->tqNodeCount; nodeId++)
	{
		PgturbohybridGraphScanNode *node = &storage->nodes[nodeId];
		char	   *exactVector;

		CHECK_FOR_INTERRUPTS();
		if (!node->loaded)
			continue;

		exactVector = storage->exactArena +
			((Size) nodeId * storage->exactBytes);
		if (!PgturbohybridGraphReadExactVectorInto(index, node, meta->dimensions,
										exactVector, NULL))
			return false;
		node->exactVector = exactVector;
	}

	return true;
}

static void
PgturbohybridGraphBuildPayloadRefs(PgturbohybridGraphMetaPageData *meta, PgturbohybridGraphScanStorage *storage)
{
	uint32		refCount = 0;
	uint32		refIndex = 0;

	if (meta->tqPayloadCount == 0 || meta->tqPayloadBytes == 0)
		return;

	for (uint32 nodeId = 0; nodeId < meta->tqNodeCount; nodeId++)
	{
		PgturbohybridGraphScanNode *node = &storage->nodes[nodeId];

		if (!node->loaded || node->payloads == NULL ||
			node->flags & PGTURBOHYBRID_GRAPH_NODE_DEAD)
			continue;

		for (int slot = 0; slot < meta->tqPayloadCount; slot++)
		{
			if (node->payloadMask & (uint16) (1U << slot))
				refCount++;
		}
	}

	if (refCount == 0)
		return;

	storage->payloadRefs = palloc0(sizeof(PgturbohybridGraphPayloadRef) * refCount);
	storage->payloadRefCount = refCount;

	for (uint32 nodeId = 0; nodeId < meta->tqNodeCount; nodeId++)
	{
		PgturbohybridGraphScanNode *node = &storage->nodes[nodeId];

		if (!node->loaded || node->payloads == NULL ||
			node->flags & PGTURBOHYBRID_GRAPH_NODE_DEAD)
			continue;

		for (int slot = 0; slot < meta->tqPayloadCount; slot++)
		{
			if ((node->payloadMask & (uint16) (1U << slot)) == 0)
				continue;

			storage->payloadRefs[refIndex].payloadSlot = (int16) slot;
			storage->payloadRefs[refIndex].payloadValue = node->payloads[slot];
			storage->payloadRefs[refIndex].nodeId = nodeId;
			refIndex++;
		}
	}

	qsort(storage->payloadRefs, storage->payloadRefCount,
		  sizeof(PgturbohybridGraphPayloadRef), PgturbohybridGraphPayloadRefCompare);
}

bool
PgturbohybridGraphPayloadRefRange(PgturbohybridGraphScanStorage *storage, int payloadSlot,
					   int32 payloadValue, uint32 *firstIndex,
					   uint32 *refCount)
{
	uint32		lo = 0;
	uint32		hi = storage->payloadRefCount;
	uint32		first;

	if (payloadSlot < 0 || storage->payloadRefs == NULL ||
		storage->payloadRefCount == 0)
		return false;

	while (lo < hi)
	{
		uint32		mid = lo + (hi - lo) / 2;
		PgturbohybridGraphPayloadRef *ref = &storage->payloadRefs[mid];

		if (ref->payloadSlot < payloadSlot ||
			(ref->payloadSlot == payloadSlot &&
			 ref->payloadValue < payloadValue))
			lo = mid + 1;
		else
			hi = mid;
	}

	first = lo;
	hi = storage->payloadRefCount;
	while (lo < hi)
	{
		uint32		mid = lo + (hi - lo) / 2;
		PgturbohybridGraphPayloadRef *ref = &storage->payloadRefs[mid];

		if (ref->payloadSlot > payloadSlot ||
			(ref->payloadSlot == payloadSlot &&
			 ref->payloadValue > payloadValue))
			hi = mid;
		else
			lo = mid + 1;
	}

	if (first >= lo)
		return false;

	*firstIndex = first;
	*refCount = lo - first;
	return true;
}

PgturbohybridMultiVector *
PgturbohybridGraphLoadMultiVectorDocVector(Relation index,
									  PgturbohybridGraphMetaPageData *meta,
									  PgturbohybridGraphScanStorage *storage,
									  TqDocId docId,
									  MemoryContext ctx,
									  PgturbohybridMultiVectorDocSidecarAccessStats *stats)
{
	TqMultiVectorDocMapEntry *entry;
	PgturbohybridMultiVector *mv;
	Size		totalFloats;
	Size		mvSize;
	uint32		firstChunk;
	uint32		chunkCount;
	uint32		copiedFloats = 0;
	MemoryContext oldCtx;
	instr_time	vectorReconstructStart;

	if (storage == NULL || meta == NULL || docId >= meta->tqMultivectorDocCount)
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
				 errmsg("document-node multivector sidecar has invalid document id"),
				 errhint("REINDEX the index to rebuild the document-node multivector sidecar.")));

	if (!storage->multivectorDocVectorsPaged)
	{
		if (storage->multivectorDocVectors == NULL ||
			storage->multivectorDocVectors[docId] == NULL)
			ereport(ERROR,
					(errcode(ERRCODE_INDEX_CORRUPTED),
					 errmsg("document-node multivector sidecar is missing a document vector"),
					 errhint("REINDEX the index to rebuild the document-node multivector sidecar.")));
		return storage->multivectorDocVectors[docId];
	}

	if (storage->multivectorDocVectorFirstChunk == NULL ||
		storage->multivectorDocVectorChunkCounts == NULL ||
		storage->multivectorDocVectorChunks == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
				 errmsg("paged document-node multivector sidecar metadata is missing"),
				 errhint("REINDEX the index to rebuild the document-node multivector sidecar.")));

	entry = &storage->multivectorDocMap[docId];
	totalFloats =
		PgturbohybridMultiVectorFloatCount(entry->tokenCount,
										   meta->dimensions);
	mvSize = PgturbohybridMultiVectorSize(entry->tokenCount, meta->dimensions);
	if (stats != NULL)
		INSTR_TIME_SET_CURRENT(vectorReconstructStart);
	oldCtx = MemoryContextSwitchTo(ctx);
	mv = MemoryContextAllocZero(ctx, mvSize);
	MemoryContextSwitchTo(oldCtx);
	SET_VARSIZE(mv, mvSize);
	mv->dim = meta->dimensions;
	mv->count = entry->tokenCount;
	if (stats != NULL)
		stats->vectorReconstructUs +=
			(uint64) PgturbohybridGraphElapsedUsSince(vectorReconstructStart);

	firstChunk = storage->multivectorDocVectorFirstChunk[docId];
	chunkCount = storage->multivectorDocVectorChunkCounts[docId];
	if (firstChunk == PG_UINT32_MAX || chunkCount == 0 ||
		(uint64) firstChunk + (uint64) chunkCount >
		(uint64) storage->multivectorDocVectorChunkCount)
		PgturbohybridGraphMultiVectorDocMapError(index,
												 "paged document vector sidecar chunk range is invalid");

	for (uint32 chunkOffset = 0; chunkOffset < chunkCount; chunkOffset++)
	{
		PgturbohybridGraphMultiVectorDocVectorChunkRef *ref =
			&storage->multivectorDocVectorChunks[firstChunk + chunkOffset];
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
			ItemId		iid;
			Item		item;
			Size		itemSize;
			PgturbohybridGraphDecodedMultiVectorDocVectorTuple tuple;
			instr_time	pageReadStart;

		CHECK_FOR_INTERRUPTS();
		if (stats != NULL)
			INSTR_TIME_SET_CURRENT(pageReadStart);
		buf = ReadBuffer(index, ref->blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = PgturbohybridGraphPageGetOpaque(page);
		if (stats != NULL)
			stats->pageReadUs +=
				(uint64) PgturbohybridGraphElapsedUsSince(pageReadStart);
		if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) !=
			PGTURBOHYBRID_GRAPH_PAGE_KIND_MULTIVECTOR_DOCMAP ||
			ref->offno < FirstOffsetNumber ||
			ref->offno > PageGetMaxOffsetNumber(page))
		{
			UnlockReleaseBuffer(buf);
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "paged document vector sidecar chunk reference is invalid");
		}
			iid = PageGetItemId(page, ref->offno);
			item = PageGetItem(page, iid);
			itemSize = ItemIdGetLength(iid);
			if (!PgturbohybridGraphDecodeMultiVectorDocVectorTuple(item,
																   itemSize,
																   &tuple))
			{
				UnlockReleaseBuffer(buf);
				PgturbohybridGraphMultiVectorDocMapError(index,
														 "paged document vector sidecar chunk is malformed");
			}
			if (tuple.docId != docId ||
				tuple.startFloat != ref->startFloat ||
				tuple.count != ref->count ||
				tuple.storageKind != ref->storageKind ||
				tuple.startFloat != copiedFloats ||
				(uint64) tuple.startFloat + tuple.count > totalFloats)
			{
				UnlockReleaseBuffer(buf);
				PgturbohybridGraphMultiVectorDocMapError(index,
														 "paged document vector sidecar chunk is invalid");
			}
			if (stats != NULL)
				INSTR_TIME_SET_CURRENT(vectorReconstructStart);
			PgturbohybridGraphDecodeMultiVectorDocValues(mv->values,
														 tuple.startFloat,
														 tuple.count,
														 tuple.storageKind,
														 tuple.values,
														 tuple.scale);
			copiedFloats += tuple.count;
		if (stats != NULL)
			stats->vectorReconstructUs +=
				(uint64) PgturbohybridGraphElapsedUsSince(vectorReconstructStart);
		if (stats != NULL)
		{
			stats->pagesRead++;
			stats->pagedVectorPagesRead++;
			stats->cacheMisses++;
			stats->bytesTouched += itemSize;
			stats->pagedVectorBytesTouched += itemSize;
		}
		UnlockReleaseBuffer(buf);
	}
	if (copiedFloats != totalFloats)
		PgturbohybridGraphMultiVectorDocMapError(index,
												 "paged document vector sidecar chunks are incomplete");
	if (stats != NULL)
		stats->vectorsLoaded++;
	return mv;
}

bool
PgturbohybridGraphVisitMultiVectorDocVectorChunks(Relation index,
									  PgturbohybridGraphMetaPageData *meta,
									  PgturbohybridGraphScanStorage *storage,
									  TqDocId docId,
									  PgturbohybridMultiVectorDocChunkCallback callback,
									  void *arg,
									  PgturbohybridMultiVectorDocSidecarAccessStats *stats)
{
	TqMultiVectorDocMapEntry *entry;
	Size		totalFloats;
	uint32		firstChunk;
	uint32		chunkCount;
	uint32		visitedFloats = 0;

	if (storage == NULL || meta == NULL || callback == NULL ||
		docId >= meta->tqMultivectorDocCount)
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
				 errmsg("document-node multivector sidecar has invalid document id"),
				 errhint("REINDEX the index to rebuild the document-node multivector sidecar.")));

	if (!storage->multivectorDocVectorsPaged)
		return false;

	if (storage->multivectorDocVectorFirstChunk == NULL ||
		storage->multivectorDocVectorChunkCounts == NULL ||
		storage->multivectorDocVectorChunks == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
				 errmsg("paged document-node multivector sidecar metadata is missing"),
				 errhint("REINDEX the index to rebuild the document-node multivector sidecar.")));

	entry = &storage->multivectorDocMap[docId];
	totalFloats =
		PgturbohybridMultiVectorFloatCount(entry->tokenCount,
										   meta->dimensions);
	firstChunk = storage->multivectorDocVectorFirstChunk[docId];
	chunkCount = storage->multivectorDocVectorChunkCounts[docId];
	if (firstChunk == PG_UINT32_MAX || chunkCount == 0 ||
		(uint64) firstChunk + (uint64) chunkCount >
		(uint64) storage->multivectorDocVectorChunkCount)
		PgturbohybridGraphMultiVectorDocMapError(index,
												 "paged document vector sidecar chunk range is invalid");

	for (uint32 chunkOffset = 0; chunkOffset < chunkCount; chunkOffset++)
	{
		PgturbohybridGraphMultiVectorDocVectorChunkRef *ref =
			&storage->multivectorDocVectorChunks[firstChunk + chunkOffset];
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		ItemId		iid;
		Item		item;
		Size		itemSize;
		PgturbohybridGraphDecodedMultiVectorDocVectorTuple tuple;
		float	   *decodedValues = NULL;
		const float *callbackValues = NULL;
		instr_time	pageReadStart;

		CHECK_FOR_INTERRUPTS();
		if (stats != NULL)
			INSTR_TIME_SET_CURRENT(pageReadStart);
		buf = ReadBuffer(index, ref->blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = PgturbohybridGraphPageGetOpaque(page);
		if (stats != NULL)
			stats->pageReadUs +=
				(uint64) PgturbohybridGraphElapsedUsSince(pageReadStart);
		if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) !=
			PGTURBOHYBRID_GRAPH_PAGE_KIND_MULTIVECTOR_DOCMAP ||
			ref->offno < FirstOffsetNumber ||
			ref->offno > PageGetMaxOffsetNumber(page))
		{
			UnlockReleaseBuffer(buf);
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "paged document vector sidecar chunk reference is invalid");
		}
		iid = PageGetItemId(page, ref->offno);
		item = PageGetItem(page, iid);
		itemSize = ItemIdGetLength(iid);
		if (!PgturbohybridGraphDecodeMultiVectorDocVectorTuple(item,
															   itemSize,
															   &tuple))
		{
			UnlockReleaseBuffer(buf);
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "paged document vector sidecar chunk is malformed");
		}
		if (tuple.docId != docId ||
			tuple.startFloat != ref->startFloat ||
			tuple.count != ref->count ||
			tuple.storageKind != ref->storageKind ||
			tuple.startFloat != visitedFloats ||
			(uint64) tuple.startFloat + tuple.count > totalFloats)
		{
			UnlockReleaseBuffer(buf);
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "paged document vector sidecar chunk is invalid");
		}

		if (tuple.storageKind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32)
			callbackValues = (const float *) tuple.values;
		else
		{
			decodedValues = palloc(sizeof(float) * tuple.count);
			PgturbohybridGraphDecodeMultiVectorDocValues(decodedValues, 0,
														 tuple.count,
														 tuple.storageKind,
														 tuple.values,
														 tuple.scale);
			callbackValues = decodedValues;
		}

		if (!callback(callbackValues, tuple.count, tuple.startFloat, arg))
		{
			if (decodedValues != NULL)
				pfree(decodedValues);
			if (stats != NULL)
			{
				stats->pagesRead++;
				stats->pagedVectorPagesRead++;
				stats->cacheMisses++;
				stats->bytesTouched += itemSize;
				stats->pagedVectorBytesTouched += itemSize;
			}
			UnlockReleaseBuffer(buf);
			return true;
		}
		if (decodedValues != NULL)
			pfree(decodedValues);
		visitedFloats += tuple.count;
		if (stats != NULL)
		{
			stats->pagesRead++;
			stats->pagedVectorPagesRead++;
			stats->cacheMisses++;
			stats->bytesTouched += itemSize;
			stats->pagedVectorBytesTouched += itemSize;
		}
		UnlockReleaseBuffer(buf);
	}
	if (visitedFloats != totalFloats)
		PgturbohybridGraphMultiVectorDocMapError(index,
												 "paged document vector sidecar chunks are incomplete");
	return true;
}

bool
PgturbohybridGraphVisitMultiVectorDocCompactChunks(Relation index,
										  PgturbohybridGraphMetaPageData *meta,
										  PgturbohybridGraphScanStorage *storage,
										  TqDocId docId,
										  PgturbohybridMultiVectorDocCompactChunkCallback callback,
										  void *arg,
										  PgturbohybridMultiVectorDocSidecarAccessStats *stats)
{
	TqMultiVectorDocMapEntry *entry;
	Size		totalFloats;
	uint32		firstChunk;
	uint32		chunkCount;
	uint32		visitedFloats = 0;

	if (storage == NULL || meta == NULL || callback == NULL ||
		docId >= meta->tqMultivectorDocCount)
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
				 errmsg("document-node multivector sidecar has invalid document id"),
				 errhint("REINDEX the index to rebuild the document-node multivector sidecar.")));

	if (!storage->multivectorDocVectorsPaged)
		return false;

	if (storage->multivectorDocVectorFirstChunk == NULL ||
		storage->multivectorDocVectorChunkCounts == NULL ||
		storage->multivectorDocVectorChunks == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
				 errmsg("paged document-node multivector sidecar metadata is missing"),
				 errhint("REINDEX the index to rebuild the document-node multivector sidecar.")));

	entry = &storage->multivectorDocMap[docId];
	totalFloats =
		PgturbohybridMultiVectorFloatCount(entry->tokenCount,
										   meta->dimensions);
	firstChunk = storage->multivectorDocVectorFirstChunk[docId];
	chunkCount = storage->multivectorDocVectorChunkCounts[docId];
	if (firstChunk == PG_UINT32_MAX || chunkCount == 0 ||
		(uint64) firstChunk + (uint64) chunkCount >
		(uint64) storage->multivectorDocVectorChunkCount)
		PgturbohybridGraphMultiVectorDocMapError(index,
												 "paged document vector sidecar chunk range is invalid");

	for (uint32 chunkOffset = 0; chunkOffset < chunkCount; chunkOffset++)
	{
		PgturbohybridGraphMultiVectorDocVectorChunkRef *ref =
			&storage->multivectorDocVectorChunks[firstChunk + chunkOffset];
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		ItemId		iid;
		Item		item;
		Size		itemSize;
		PgturbohybridGraphDecodedMultiVectorDocVectorTuple tuple;
		instr_time	pageReadStart;

		CHECK_FOR_INTERRUPTS();
		if (stats != NULL)
			INSTR_TIME_SET_CURRENT(pageReadStart);
		buf = ReadBuffer(index, ref->blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = PgturbohybridGraphPageGetOpaque(page);
		if (stats != NULL)
			stats->pageReadUs +=
				(uint64) PgturbohybridGraphElapsedUsSince(pageReadStart);
		if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) !=
			PGTURBOHYBRID_GRAPH_PAGE_KIND_MULTIVECTOR_DOCMAP ||
			ref->offno < FirstOffsetNumber ||
			ref->offno > PageGetMaxOffsetNumber(page))
		{
			UnlockReleaseBuffer(buf);
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "paged document vector sidecar chunk reference is invalid");
		}
		iid = PageGetItemId(page, ref->offno);
		item = PageGetItem(page, iid);
		itemSize = ItemIdGetLength(iid);
		if (!PgturbohybridGraphDecodeMultiVectorDocVectorTuple(item,
															   itemSize,
															   &tuple))
		{
			UnlockReleaseBuffer(buf);
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "paged document vector sidecar chunk is malformed");
		}
		if (tuple.docId != docId ||
			tuple.startFloat != ref->startFloat ||
			tuple.count != ref->count ||
			tuple.storageKind != ref->storageKind ||
			tuple.startFloat != visitedFloats ||
			(uint64) tuple.startFloat + tuple.count > totalFloats)
		{
			UnlockReleaseBuffer(buf);
			PgturbohybridGraphMultiVectorDocMapError(index,
													 "paged document vector sidecar chunk is invalid");
		}

		if (!callback(tuple.storageKind, tuple.values, tuple.scale,
					  tuple.count, tuple.startFloat, arg))
		{
			if (stats != NULL)
			{
				stats->pagesRead++;
				stats->pagedVectorPagesRead++;
				stats->cacheMisses++;
				stats->bytesTouched += itemSize;
				stats->pagedVectorBytesTouched += itemSize;
			}
			UnlockReleaseBuffer(buf);
			return true;
		}
		visitedFloats += tuple.count;
		if (stats != NULL)
		{
			stats->pagesRead++;
			stats->pagedVectorPagesRead++;
			stats->cacheMisses++;
			stats->bytesTouched += itemSize;
			stats->pagedVectorBytesTouched += itemSize;
		}
		UnlockReleaseBuffer(buf);
	}
	if (visitedFloats != totalFloats)
		PgturbohybridGraphMultiVectorDocMapError(index,
												 "paged document vector sidecar chunks are incomplete");
	return true;
}

PgturbohybridMultiVector *
PgturbohybridGraphReadMultiVectorDocFromSidecar(Relation index, TqDocId docId,
											MemoryContext ctx)
{
	PgturbohybridGraphMetaPageData meta;
	PgturbohybridGraphScanStorage storage;
	PgturbohybridMultiVectorDocSidecarAccessStats stats;
	PgturbohybridMultiVector *doc;
	PgturbohybridMultiVector *copy;
	Size		docSize;
	MemoryContext allocCtx;
	MemoryContext loadCtx;
	MemoryContext oldCtx;

	if (index == NULL)
		return NULL;
	if (!PgturbohybridGraphReadMeta(index, &meta) ||
		meta.tqMultivectorDocCount == 0 ||
			docId >= meta.tqMultivectorDocCount)
		return NULL;

	allocCtx = ctx != NULL ? ctx : CurrentMemoryContext;
	loadCtx = AllocSetContextCreate(allocCtx,
									"pgturbohybrid sidecar doc read",
									ALLOCSET_DEFAULT_SIZES);
	memset(&storage, 0, sizeof(storage));
	memset(&stats, 0, sizeof(stats));
	PgturbohybridGraphInitScanStorage(index, &meta, &storage, NULL);
	storage.ctx = loadCtx;
	if (!PgturbohybridGraphLoadMultiVectorDocMapWithStats(index, &meta,
														  &storage,
														  false,
														  &stats) ||
		!storage.multivectorDocMapLoaded ||
		!storage.multivectorDocVectorsLoaded)
	{
		MemoryContextDelete(loadCtx);
		return NULL;
	}

	doc = PgturbohybridGraphLoadMultiVectorDocVector(index, &meta, &storage,
													 docId, loadCtx, &stats);
	if (doc == NULL)
	{
		MemoryContextDelete(loadCtx);
		return NULL;
	}

	docSize = VARSIZE_ANY(doc);
	oldCtx = MemoryContextSwitchTo(allocCtx);
	copy = palloc(docSize);
	memcpy(copy, doc, docSize);
	MemoryContextSwitchTo(oldCtx);
	MemoryContextDelete(loadCtx);
	return copy;
}

static bool
PgturbohybridGraphCacheMatches(PgturbohybridGraphNativeCache *cache, Relation index,
							   PgturbohybridGraphMetaPageData *meta)
{
	return cache->relid == RelationGetRelid(index) &&
		cache->relfilenumber == PgturbohybridGraphRelFileNumber(index) &&
		cache->dimensions == meta->dimensions &&
		cache->m == meta->m &&
		cache->graphMaxLevel == meta->graphMaxLevel &&
		cache->graphFlags == meta->graphFlags &&
		cache->graphGeneration == meta->graphGeneration &&
		cache->tqNodeCount == meta->tqNodeCount &&
		cache->tqEntryNodeId == meta->tqEntryNodeId &&
		cache->tqSegmentCount == meta->tqSegmentCount &&
		cache->tqCodeBytes == meta->tqCodeBytes &&
		cache->tqBits == meta->tqBits &&
		cache->tqPayloadCount == meta->tqPayloadCount &&
		cache->tqPayloadBytes == meta->tqPayloadBytes &&
		cache->tqResidualRerankBytes == meta->tqResidualRerankBytes &&
		cache->tqCodeStartBlkno == meta->tqCodeStartBlkno &&
		cache->tqAdjStartBlkno == meta->tqAdjStartBlkno &&
		cache->tqExactStartBlkno == meta->tqExactStartBlkno &&
		cache->tqCorrectionStartBlkno == meta->tqCorrectionStartBlkno &&
		cache->tqMultivectorDocMapStartBlkno == meta->tqMultivectorDocMapStartBlkno &&
		cache->tqMultivectorDocMapPageCount == meta->tqMultivectorDocMapPageCount &&
		cache->tqMultivectorDocCount == meta->tqMultivectorDocCount &&
		cache->tqMultivectorDocMapBytes == meta->tqMultivectorDocMapBytes &&
		cache->tqMultivectorDocMapVersion == meta->tqMultivectorDocMapVersion &&
		cache->tqMultivectorDocMapFlags == meta->tqMultivectorDocMapFlags &&
		cache->tqMultivectorGraphMode == meta->tqMultivectorGraphMode &&
		cache->multivectorDocSidecarResident ==
		PgturbohybridGraphShouldCacheMultiVectorDocSidecar(meta) &&
		memcmp(cache->tqSegments, meta->tqSegments,
			   sizeof(PgturbohybridGraphSegmentMetaData) * meta->tqSegmentCount) == 0;
}

static void
PgturbohybridGraphDropStaleCaches(Relation index, PgturbohybridGraphMetaPageData *meta)
{
	PgturbohybridGraphNativeCache **link = &pgturbohybridGraphCacheList;

	while (*link != NULL)
	{
		PgturbohybridGraphNativeCache *cache = *link;

		if (cache->relid == RelationGetRelid(index) &&
			!PgturbohybridGraphCacheMatches(cache, index, meta))
		{
			*link = cache->next;
			MemoryContextDelete(cache->ctx);
			continue;
		}

		link = &cache->next;
	}
}

/*
 * Detach every session-shared mmap'd node map for this relation.  Only this
 * backend's mapping is released (munmap is per-process), so it is safe while
 * other backends hold their own mappings.  Called from InvalidateCaches so a
 * REINDEX/insert cannot leave a stale mapping (plus its palloc'd view context)
 * alive for the backend's lifetime -- the list was previously never swept.
 */
static void
PgturbohybridGraphInvalidateSharedMaps(Relation index)
{
	PgturbohybridGraphSharedMap **link = &pgturbohybridGraphSharedMapList;
	Oid			relid = RelationGetRelid(index);

	while (*link != NULL)
	{
		PgturbohybridGraphSharedMap *map = *link;

		if (map->relid == relid)
		{
			*link = map->next;
#ifndef WIN32
			munmap(map->base, map->size);
#endif
			MemoryContextDelete(map->ctx);
			continue;
		}

		link = &map->next;
	}
}

void
PgturbohybridGraphInvalidateCaches(Relation index)
{
	PgturbohybridGraphNativeCache **link = &pgturbohybridGraphCacheList;
	PgturbohybridGraphDocSidecarCache **docLink =
		&pgturbohybridGraphDocSidecarCacheList;
	Oid			relid = RelationGetRelid(index);

	while (*link != NULL)
	{
		PgturbohybridGraphNativeCache *cache = *link;

		if (cache->relid == relid)
		{
			*link = cache->next;
			MemoryContextDelete(cache->ctx);
			continue;
		}

		link = &cache->next;
	}

	while (*docLink != NULL)
	{
		PgturbohybridGraphDocSidecarCache *cache = *docLink;

		if (cache->relid == relid)
		{
			*docLink = cache->next;
			MemoryContextDelete(cache->ctx);
			continue;
		}

		docLink = &cache->next;
	}

	/* Correction-sidecar cache lives in pgturbohybrid_quant_cache.c. */
	PgturbohybridGraphInvalidateCorrectionCache(index);

	/* Session-shared mmap'd node maps. */
	PgturbohybridGraphInvalidateSharedMaps(index);
}

static PgturbohybridGraphNativeCache *
PgturbohybridGraphFindCache(Relation index, PgturbohybridGraphMetaPageData *meta)
{
	PgturbohybridGraphDropStaleCaches(index, meta);

	for (PgturbohybridGraphNativeCache *cache = pgturbohybridGraphCacheList; cache != NULL; cache = cache->next)
	{
		if (PgturbohybridGraphCacheMatches(cache, index, meta))
			return cache;
	}

	return NULL;
}

/*
 * Sum the resident footprint a built per-backend cache holds: the code, exact
 * and adjacency arenas (the terms that dominate and are duplicated once per
 * backend), plus the node array and per-scan metadata.
 */
static void
PgturbohybridGraphCacheComputeResidentBytes(PgturbohybridGraphNativeCache *cache,
											PgturbohybridGraphMetaPageData *meta)
{
	PgturbohybridGraphScanStorage *storage = &cache->storage;
	int			adjRecordCount = PgturbohybridGraphAdjRecordCount(meta);
	Size		codeBytes = 0;
	Size		adjBytes = 0;
	Size		exactBytes = 0;
	Size		otherBytes = 0;

	if (storage->codeArena != NULL)
		codeBytes = (Size) meta->tqNodeCount * (Size) meta->tqCodeBytes;
	if (storage->exactArena != NULL)
		exactBytes = (Size) meta->tqNodeCount * storage->exactBytes;

	/* Adjacency: the per-slot metadata arrays plus the actual neighbor lists. */
	adjBytes += (Size) adjRecordCount *
		(sizeof(uint32 *) + sizeof(uint16) + sizeof(BlockNumber) + sizeof(OffsetNumber));
	if (storage->neighborCounts != NULL)
	{
		for (int i = 0; i < adjRecordCount; i++)
			adjBytes += (Size) storage->neighborCounts[i] * sizeof(uint32);
	}

	/* Node array, residual/payload arenas, visited generation, code-page map. */
	otherBytes += (Size) meta->tqNodeCount * sizeof(PgturbohybridGraphScanNode);
	if (storage->residualArena != NULL)
		otherBytes += (Size) meta->tqNodeCount * (Size) meta->tqResidualRerankBytes;
	if (storage->payloadArena != NULL)
		otherBytes += (Size) meta->tqNodeCount * (Size) meta->tqPayloadBytes;
	if (storage->visitedGeneration != NULL)
		otherBytes += (Size) meta->tqNodeCount * sizeof(uint32);
	if (storage->multivectorDocMapLoaded)
		otherBytes +=
			(Size) meta->tqNodeCount * sizeof(TqMultiVectorNodeMapEntry) +
			(Size) meta->tqMultivectorDocCount *
			sizeof(TqMultiVectorDocMapEntry);
	if (storage->multivectorDocVectorsLoaded &&
		storage->multivectorDocVectors != NULL)
	{
		otherBytes += (Size) meta->tqMultivectorDocCount *
			sizeof(PgturbohybridMultiVector *);
		if (storage->multivectorDocVectorsPaged)
		{
			otherBytes += (Size) meta->tqMultivectorDocCount *
				(sizeof(uint32) + sizeof(uint32));
			otherBytes += (Size) storage->multivectorDocVectorChunkCapacity *
				sizeof(PgturbohybridGraphMultiVectorDocVectorChunkRef);
		}
		for (uint32 docId = 0; docId < meta->tqMultivectorDocCount; docId++)
		{
			PgturbohybridMultiVector *mv =
				storage->multivectorDocVectors[docId];

			if (mv != NULL)
				otherBytes += (Size) VARSIZE_ANY(mv);
		}
	}
	if (storage->multivectorDocCentroidsLoaded &&
		storage->multivectorDocCentroids != NULL)
	{
		otherBytes += (Size) meta->tqMultivectorDocCount *
			(sizeof(PgturbohybridMultiVector *) + sizeof(float));
		for (uint32 docId = 0; docId < meta->tqMultivectorDocCount; docId++)
		{
			PgturbohybridMultiVector *mv =
				storage->multivectorDocCentroids[docId];

			if (mv != NULL)
				otherBytes += (Size) VARSIZE_ANY(mv);
		}
	}
	if (storage->multivectorCentroidPostingsLoaded &&
		storage->multivectorCentroidPostings != NULL &&
		storage->multivectorCentroidPostingListOffsets != NULL)
	{
		otherBytes += ((Size) storage->multivectorCentroidPostingCodebookSize + 1) *
			sizeof(uint32);
		otherBytes += (Size) storage->multivectorCentroidPostingCount *
			sizeof(PgturbohybridGraphMultiVectorCentroidPostingEntry);
		if (storage->multivectorCentroidDocCodesLoaded &&
			storage->multivectorCentroidDocCodeOffsets != NULL &&
			storage->multivectorCentroidDocCodes != NULL)
		{
			otherBytes += ((Size) meta->tqMultivectorDocCount + 1) *
				sizeof(uint32);
			otherBytes += (Size) storage->multivectorCentroidDocCodeCount *
				sizeof(uint32);
		}
	}
	if (storage->multivectorQuantizedInvertedPostingsLoaded &&
		storage->multivectorQuantizedInvertedPostings != NULL &&
		storage->multivectorQuantizedInvertedListOffsets != NULL)
	{
		otherBytes += ((Size) storage->multivectorQuantizedInvertedCodebookSize + 1) *
			sizeof(uint32);
		otherBytes += (Size) storage->multivectorQuantizedInvertedPostingCount *
			sizeof(PgturbohybridGraphMultiVectorQuantizedPostingEntry);
		if (storage->multivectorQuantizedInvertedDocCodesLoaded &&
			storage->multivectorQuantizedInvertedDocCodeOffsets != NULL &&
			storage->multivectorQuantizedInvertedDocCodes != NULL)
		{
			otherBytes += ((Size) meta->tqMultivectorDocCount + 1) *
				sizeof(uint32);
			otherBytes += (Size) storage->multivectorQuantizedInvertedDocCodeCount *
				sizeof(uint32);
		}
	}
	otherBytes += (Size) storage->codePageCount * (sizeof(bool) + sizeof(BlockNumber));

	cache->residentCodeBytes = codeBytes;
	cache->residentAdjBytes = adjBytes;
	cache->residentExactBytes = exactBytes;
	cache->residentTotalBytes = codeBytes + adjBytes + exactBytes + otherBytes;
}

static PgturbohybridGraphNativeCache *
PgturbohybridGraphBuildCache(Relation index, PgturbohybridGraphMetaPageData *meta)
{
	MemoryContext cacheCtx;
	MemoryContext oldCtx;
	PgturbohybridGraphNativeCache *cache;
	PgturbohybridGraphScanOpaqueData loadStats;
	instr_time	buildStart;
	instr_time	buildElapsed;

	INSTR_TIME_SET_CURRENT(buildStart);
	memset(&loadStats, 0, sizeof(loadStats));
	loadStats.pgturbohybridGraphScan = true;

	cacheCtx = AllocSetContextCreate(CacheMemoryContext,
									 "pgturbohybrid native graph cache",
									 ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(cacheCtx);

	cache = palloc0(sizeof(PgturbohybridGraphNativeCache));
	cache->relid = RelationGetRelid(index);
	cache->relfilenumber = PgturbohybridGraphRelFileNumber(index);
	cache->dimensions = meta->dimensions;
	cache->m = meta->m;
	cache->graphMaxLevel = meta->graphMaxLevel;
	cache->graphFlags = meta->graphFlags;
	cache->graphGeneration = meta->graphGeneration;
	cache->tqNodeCount = meta->tqNodeCount;
	cache->tqEntryNodeId = meta->tqEntryNodeId;
	cache->tqSegmentCount = meta->tqSegmentCount;
	cache->tqCodeBytes = meta->tqCodeBytes;
	cache->tqBits = meta->tqBits;
	cache->tqPayloadCount = meta->tqPayloadCount;
	cache->tqPayloadBytes = meta->tqPayloadBytes;
	cache->tqResidualRerankBytes = meta->tqResidualRerankBytes;
	cache->tqCodeStartBlkno = meta->tqCodeStartBlkno;
	cache->tqAdjStartBlkno = meta->tqAdjStartBlkno;
	cache->tqExactStartBlkno = meta->tqExactStartBlkno;
	cache->tqCorrectionStartBlkno = meta->tqCorrectionStartBlkno;
	cache->tqMultivectorDocMapStartBlkno =
		meta->tqMultivectorDocMapStartBlkno;
	cache->tqMultivectorDocMapPageCount =
		meta->tqMultivectorDocMapPageCount;
	cache->tqMultivectorDocCount = meta->tqMultivectorDocCount;
	cache->tqMultivectorDocMapBytes =
		meta->tqMultivectorDocMapBytes;
	cache->tqMultivectorDocMapVersion =
		meta->tqMultivectorDocMapVersion;
	cache->tqMultivectorDocMapFlags =
		meta->tqMultivectorDocMapFlags;
	cache->tqMultivectorGraphMode = meta->tqMultivectorGraphMode;
	cache->multivectorDocSidecarResident =
		PgturbohybridGraphShouldCacheMultiVectorDocSidecar(meta);
	memcpy(cache->tqSegments, meta->tqSegments,
		   sizeof(PgturbohybridGraphSegmentMetaData) * meta->tqSegmentCount);
	cache->ctx = cacheCtx;

	PgturbohybridGraphInitScanStorageUncached(meta, &cache->storage,
								   PgturbohybridGraphShouldCacheExactVectors(index, meta));
	cache->storage.ctx = cacheCtx;
	if (meta->tqNodeCount > 0)
	{
		cache->storage.visitedGeneration = palloc0(sizeof(uint32) * meta->tqNodeCount);
		cache->storage.visitGeneration = palloc0(sizeof(uint32));
	}

	if (cache->storage.codeArena != NULL)
	{
		for (uint32 nodeId = 0; nodeId < meta->tqNodeCount;
			 nodeId += cache->storage.codeTuplesPerPage)
			(void) PgturbohybridGraphLoadCodePage(index, &loadStats, meta,
												  &cache->storage, nodeId);
	}

	PgturbohybridGraphBuildPayloadRefs(meta, &cache->storage);

	if (meta->tqNodeCount > 0)
		PgturbohybridGraphLoadAllAdjPages(index, &loadStats, meta,
										  &cache->storage);

	if (cache->storage.exactArena != NULL)
		(void) PgturbohybridGraphLoadExactVectors(index, meta, &cache->storage);
	if (BlockNumberIsValid(meta->tqMultivectorDocMapStartBlkno) &&
		meta->tqMultivectorDocMapVersion ==
		PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION &&
		cache->multivectorDocSidecarResident)
		(void) PgturbohybridGraphLoadMultiVectorDocMap(index, meta,
													   &cache->storage,
													   false);
	cache->storage.cached = true;

	PgturbohybridGraphCacheComputeResidentBytes(cache, meta);
	if (PgturbohybridGraphNativeCacheWarns(cache))
		ereport(DEBUG1,
				(errmsg("pgturbohybrid native graph per-backend cache exceeds warning threshold"),
				 errdetail("index=%s resident_bytes=%llu warn_mb=%d cache_policy=%s",
						   RelationGetRelationName(index),
						   (unsigned long long) cache->residentTotalBytes,
						   pgturbohybrid_native_cache_warn_mb,
						   PgturbohybridGraphNativeCachePolicyNameForLog(pgturbohybrid_native_cache_policy)),
				 errhint("per_backend native cache memory is allocated per active PostgreSQL backend; use turbohybrid_estimate_memory(index), native_cache_scope=shared, or native_cache_scope=off when concurrency would multiply this footprint too far.")));
	cache->buildCodeBufferLockWaitUs = loadStats.graphCodeBufferLockWaitUs;
	cache->buildAdjBufferLockWaitUs = loadStats.graphAdjBufferLockWaitUs;

	cache->next = pgturbohybridGraphCacheList;
	pgturbohybridGraphCacheList = cache;

	MemoryContextSwitchTo(oldCtx);

	INSTR_TIME_SET_CURRENT(buildElapsed);
	INSTR_TIME_SUBTRACT(buildElapsed, buildStart);
	cache->buildUs = (int64) INSTR_TIME_GET_MICROSEC(buildElapsed);

	return cache;
}

static void
PgturbohybridGraphSharedInitStorageScratch(PgturbohybridGraphMetaPageData *meta,
										   PgturbohybridGraphScanStorage *storage)
{
	storage->ctx = CurrentMemoryContext;
	if (meta->tqNodeCount > 0)
	{
		storage->visitedGeneration = palloc0(sizeof(uint32) * meta->tqNodeCount);
		storage->visitGeneration = palloc0(sizeof(uint32));
	}
}

static bool
PgturbohybridGraphSharedBuildView(PgturbohybridGraphSharedMap *map,
								  PgturbohybridGraphMetaPageData *meta)
{
	PgturbohybridGraphSharedCacheHeader *hdr =
		(PgturbohybridGraphSharedCacheHeader *) map->base;
	PgturbohybridGraphSharedNode *sharedNodes =
		(PgturbohybridGraphSharedNode *) PgturbohybridGraphSharedCheckedPtr(map->base, map->size,
														 hdr->nodes.offset, hdr->nodes.length);
	uint64	   *neighborOffsets =
		(uint64 *) PgturbohybridGraphSharedCheckedPtr(map->base, map->size,
											 hdr->neighborOffsets.offset,
											 hdr->neighborOffsets.length);
	uint32	   *neighborData =
		(uint32 *) PgturbohybridGraphSharedCheckedPtr(map->base, map->size,
											 hdr->neighborData.offset,
											 hdr->neighborData.length);
	PgturbohybridGraphScanStorage *storage = &map->view;
	MemoryContext oldCtx;

	oldCtx = MemoryContextSwitchTo(map->ctx);
	memset(storage, 0, sizeof(*storage));
	storage->ctx = map->ctx;
	storage->nodes =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridGraphScanNode),
											   meta->tqNodeCount,
											   "pgturbohybrid shared graph node view"));
	storage->codeArena =
		(uint8 *) PgturbohybridGraphSharedCheckedPtr(map->base, map->size,
											 hdr->codeArena.offset, hdr->codeArena.length);
	storage->payloadArena =
		(uint8 *) PgturbohybridGraphSharedCheckedPtr(map->base, map->size,
											 hdr->payloadArena.offset, hdr->payloadArena.length);
	storage->residualArena =
		(uint8 *) PgturbohybridGraphSharedCheckedPtr(map->base, map->size,
											 hdr->residualArena.offset, hdr->residualArena.length);
	storage->exactArena =
		(char *) PgturbohybridGraphSharedCheckedPtr(map->base, map->size,
											 hdr->exactArena.offset, hdr->exactArena.length);
	storage->exactBytes = hdr->exactBytes;
	storage->neighborCounts =
		(uint16 *) PgturbohybridGraphSharedCheckedPtr(map->base, map->size,
											  hdr->neighborCounts.offset,
											  hdr->neighborCounts.length);
	storage->neighbors =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32 *),
											   hdr->adjRecordCount,
											   "pgturbohybrid shared graph neighbor view"));
	storage->payloadRefs =
		(PgturbohybridGraphPayloadRef *) PgturbohybridGraphSharedCheckedPtr(map->base, map->size,
														 hdr->payloadRefs.offset,
														 hdr->payloadRefs.length);
	storage->payloadRefCount = hdr->payloadRefCount;
	storage->levelCount = PgturbohybridGraphLevelCapacity(meta->m);
	storage->codeTuplesPerPage =
		PgturbohybridGraphTuplesPerPage(PgturbohybridGraphCodeTupleSize(meta->dimensions,
												  meta->tqPayloadCount,
												  meta->tqBits,
												  (meta->tqFlags & PGTURBOHYBRID_GRAPH_TQ_WEIGHTED) != 0,
												  meta->tqResidualRerankBytes));
	storage->codePageCount = PgturbohybridGraphPageCount(meta->tqNodeCount,
														 storage->codeTuplesPerPage);
	storage->adjPageCount = BlockNumberIsValid(meta->tqAdjStartBlkno) ? 1 : 0;
	storage->cached = true;

	for (uint32 nodeId = 0; nodeId < meta->tqNodeCount; nodeId++)
	{
		PgturbohybridGraphScanNode *node = &storage->nodes[nodeId];
		PgturbohybridGraphSharedNode *sharedNode = &sharedNodes[nodeId];

		node->heaptid = sharedNode->heaptid;
		node->payloadMask = sharedNode->payloadMask;
		node->level = sharedNode->level;
		node->exactBlkno = sharedNode->exactBlkno;
		node->exactOffno = sharedNode->exactOffno;
		node->scale = sharedNode->scale;
		node->norm = sharedNode->norm;
		node->codeNorm = sharedNode->codeNorm;
		node->ecCorrection = sharedNode->ecCorrection;
		node->flags = sharedNode->flags;
		node->loaded = sharedNode->loaded;
		if (node->loaded && storage->codeArena != NULL && meta->tqCodeBytes > 0)
			node->code = storage->codeArena + ((Size) nodeId * meta->tqCodeBytes);
		if (node->loaded && storage->payloadArena != NULL && meta->tqPayloadBytes > 0)
			node->payloads = (int32 *) (storage->payloadArena +
										((Size) nodeId * meta->tqPayloadBytes));
		if (node->loaded && storage->residualArena != NULL && meta->tqResidualRerankBytes > 0)
			node->residualSketch = storage->residualArena +
				((Size) nodeId * meta->tqResidualRerankBytes);
		if (node->loaded && storage->exactArena != NULL && storage->exactBytes > 0)
			node->exactVector = storage->exactArena + ((Size) nodeId * storage->exactBytes);
	}

	for (uint32 slot = 0; slot < hdr->adjRecordCount; slot++)
	{
		if (storage->neighborCounts[slot] > 0)
			storage->neighbors[slot] = neighborData + neighborOffsets[slot];
	}

	MemoryContextSwitchTo(oldCtx);
	return true;
}

static bool
PgturbohybridGraphMapSharedCacheFile(Relation index,
									 PgturbohybridGraphMetaPageData *meta,
									 const char *path, uint64 key,
									 PgturbohybridGraphSharedMap **mapOut,
									 int64 *attachUs, bool *invalidOut)
{
#ifdef WIN32
	(void) index;
	(void) meta;
	(void) path;
	(void) key;
	(void) mapOut;
	(void) invalidOut;
	if (attachUs != NULL)
		*attachUs = 0;
	if (invalidOut != NULL)
		*invalidOut = false;
	return false;
#else
	int			fd;
	struct stat st;
	void	   *base;
	PgturbohybridGraphSharedCacheHeader *hdr;
	MemoryContext ctx;
	MemoryContext oldCtx;
	PgturbohybridGraphSharedMap *map;
	instr_time	start;

	if (attachUs != NULL)
		*attachUs = 0;
	INSTR_TIME_SET_CURRENT(start);
	fd = open(path, O_RDONLY);
	if (fd < 0)
		return false;
	if (fstat(fd, &st) != 0 || st.st_size < (off_t) sizeof(PgturbohybridGraphSharedCacheHeader))
	{
		if (invalidOut != NULL)
			*invalidOut = true;
		close(fd);
		return false;
	}
	base = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
	close(fd);
	if (base == MAP_FAILED)
		return false;

	hdr = (PgturbohybridGraphSharedCacheHeader *) base;
	if (!PgturbohybridGraphSharedHeaderMatches(hdr, index, meta, key) ||
		!PgturbohybridGraphValidateSharedLayout(base, hdr, meta,
											(uint64) st.st_size))
	{
		if (invalidOut != NULL)
			*invalidOut = true;
		munmap(base, st.st_size);
		return false;
	}

	ctx = AllocSetContextCreate(CacheMemoryContext,
								"pgturbohybrid shared graph cache view",
								ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(ctx);
	map = palloc0(sizeof(PgturbohybridGraphSharedMap));
	map->key = key;
	map->relid = RelationGetRelid(index);
	map->relfilenumber = PgturbohybridGraphRelFileNumber(index);
	PgturbohybridGraphCacheIdentityInit(index, meta, &map->identity);
	map->base = base;
	map->size = (Size) st.st_size;
	map->ctx = ctx;
	MemoryContextSwitchTo(oldCtx);

	if (!PgturbohybridGraphSharedBuildView(map, meta))
	{
		MemoryContextDelete(ctx);
		munmap(base, st.st_size);
		return false;
	}

	map->next = pgturbohybridGraphSharedMapList;
	pgturbohybridGraphSharedMapList = map;
	map->attachUs = PgturbohybridGraphElapsedUsSince(start);
	if (attachUs != NULL)
		*attachUs = map->attachUs;
	*mapOut = map;
	return true;
#endif
}

static PgturbohybridGraphSharedMap *
PgturbohybridGraphFindSharedMap(Relation index, PgturbohybridGraphMetaPageData *meta,
								uint64 key)
{
	Oid			relid = RelationGetRelid(index);
	Oid			relfilenumber = PgturbohybridGraphRelFileNumber(index);
	PgturbohybridGraphSharedMap **link = &pgturbohybridGraphSharedMapList;

	while (*link != NULL)
	{
		PgturbohybridGraphSharedMap *map = *link;

		if (map->key == key &&
			map->relid == relid &&
			map->relfilenumber == relfilenumber)
		{
			PgturbohybridGraphCacheIdentity identity;

			PgturbohybridGraphCacheIdentityInit(index, meta, &identity);
			if (PgturbohybridGraphCacheIdentityEqual(&map->identity, &identity))
			return map;
		}

		/*
		 * Same relation, superseded generation (another backend REINDEXed or
		 * rewrote the shared file): this mapping can never match again, so
		 * detach it here rather than leaking it -- a read-only backend never
		 * reaches InvalidateCaches().
		 */
		if (map->relid == relid)
		{
			*link = map->next;
#ifndef WIN32
			munmap(map->base, map->size);
#endif
			MemoryContextDelete(map->ctx);
			continue;
		}

		link = &map->next;
	}
	return NULL;
}

static bool
PgturbohybridGraphWriteSharedCacheFile(Relation index,
									   PgturbohybridGraphMetaPageData *meta,
									   const char *path, const char *tmpPath,
									   uint64 key, int64 *buildUs,
									   int64 *codeLockWaitUs,
									   int64 *adjLockWaitUs)
{
#ifdef WIN32
	(void) index;
	(void) meta;
	(void) path;
	(void) tmpPath;
	(void) key;
	if (buildUs != NULL)
		*buildUs = 0;
	if (codeLockWaitUs != NULL)
		*codeLockWaitUs = 0;
	if (adjLockWaitUs != NULL)
		*adjLockWaitUs = 0;
	return false;
#else
	PgturbohybridGraphScanStorage storage;
	PgturbohybridGraphScanOpaqueData loadStats;
	PgturbohybridGraphSharedCacheHeader hdr;
	PgturbohybridGraphSharedNode *sharedNodes;
	uint16	   *neighborCounts;
	uint64	   *neighborOffsets;
	uint32	   *neighborData;
	uint8	   *codeArena;
	uint8	   *payloadArena;
	uint8	   *residualArena;
	char	   *exactArena;
	PgturbohybridGraphPayloadRef *payloadRefs;
	uint64		neighborValueCount = 0;
	uint64		codeBytes = 0;
	uint64		payloadBytes = 0;
	uint64		residualBytes = 0;
	uint64		exactBytes = 0;
	uint64		adjRecordCount = PgturbohybridGraphAdjRecordCount(meta);
	int			fd;
	void	   *base;
	instr_time	start;

	if (buildUs != NULL)
		*buildUs = 0;
	if (codeLockWaitUs != NULL)
		*codeLockWaitUs = 0;
	if (adjLockWaitUs != NULL)
		*adjLockWaitUs = 0;

	INSTR_TIME_SET_CURRENT(start);
	memset(&loadStats, 0, sizeof(loadStats));
	loadStats.pgturbohybridGraphScan = true;
	PgturbohybridGraphInitScanStorageUncached(meta, &storage,
								   PgturbohybridGraphShouldCacheExactVectors(index, meta));
	if (meta->tqNodeCount > 0)
	{
		for (uint32 nodeId = 0; nodeId < meta->tqNodeCount;
			 nodeId += storage.codeTuplesPerPage)
			(void) PgturbohybridGraphLoadCodePage(index, &loadStats, meta,
												  &storage, nodeId);
	}
	PgturbohybridGraphBuildPayloadRefs(meta, &storage);
	if (meta->tqNodeCount > 0)
		PgturbohybridGraphLoadAllAdjPages(index, &loadStats, meta, &storage);
	if (storage.exactArena != NULL)
		(void) PgturbohybridGraphLoadExactVectors(index, meta, &storage);

	for (uint32 slot = 0; slot < adjRecordCount; slot++)
		neighborValueCount += storage.neighborCounts[slot];

	codeBytes = storage.codeArena != NULL ?
		(uint64) meta->tqNodeCount * (uint64) meta->tqCodeBytes : 0;
	payloadBytes = storage.payloadArena != NULL ?
		(uint64) meta->tqNodeCount * (uint64) meta->tqPayloadBytes : 0;
	residualBytes = storage.residualArena != NULL ?
		(uint64) meta->tqNodeCount * (uint64) meta->tqResidualRerankBytes : 0;
	exactBytes = storage.exactArena != NULL ?
		(uint64) meta->tqNodeCount * (uint64) storage.exactBytes : 0;

	memset(&hdr, 0, sizeof(hdr));
	hdr.magic = PGTURBOHYBRID_GRAPH_SHARED_CACHE_MAGIC;
	hdr.version = PGTURBOHYBRID_GRAPH_SHARED_CACHE_VERSION;
	hdr.key = key;
	PgturbohybridGraphCacheIdentityInit(index, meta, &hdr.identity);
	hdr.dimensions = meta->dimensions;
	hdr.m = meta->m;
	hdr.graphMaxLevel = meta->graphMaxLevel;
	hdr.graphFlags = meta->graphFlags;
	hdr.tqBits = meta->tqBits;
	hdr.tqNodeCount = meta->tqNodeCount;
	hdr.tqEntryNodeId = meta->tqEntryNodeId;
	hdr.tqCodeBytes = meta->tqCodeBytes;
	hdr.tqPayloadCount = meta->tqPayloadCount;
	hdr.tqPayloadBytes = meta->tqPayloadBytes;
	hdr.tqResidualRerankBytes = meta->tqResidualRerankBytes;
	hdr.tqCodeStartBlkno = meta->tqCodeStartBlkno;
	hdr.tqAdjStartBlkno = meta->tqAdjStartBlkno;
	hdr.tqExactStartBlkno = meta->tqExactStartBlkno;
	hdr.tqCorrectionStartBlkno = meta->tqCorrectionStartBlkno;
	hdr.adjRecordCount = (uint32) adjRecordCount;
	hdr.neighborValueCount = neighborValueCount;
	hdr.payloadRefCount = storage.payloadRefCount;
	hdr.exactBytes = (uint32) storage.exactBytes;

	if (!PgturbohybridGraphBuildSharedLayout(meta, neighborValueCount,
										 storage.payloadRefCount,
										 storage.exactBytes, true, &hdr))
		return false;
	hdr.residentCodeBytes = codeBytes;
	hdr.residentAdjBytes =
		adjRecordCount * (sizeof(uint16) + sizeof(uint64)) +
		neighborValueCount * sizeof(uint32);
	hdr.residentExactBytes = exactBytes;
	hdr.residentTotalBytes = hdr.fileSize;

	fd = open(tmpPath, O_CREAT | O_EXCL | O_RDWR, 0600);
	if (fd < 0)
		return false;
	if (ftruncate(fd, (off_t) hdr.fileSize) != 0)
	{
		close(fd);
		unlink(tmpPath);
		return false;
	}
	PgturbohybridGraphSharedTestPause("after_ftruncate");
	base = mmap(NULL, (Size) hdr.fileSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (base == MAP_FAILED)
	{
		close(fd);
		unlink(tmpPath);
		return false;
	}
	PgturbohybridGraphSharedTestPause("after_mmap");
	memset(base, 0, (Size) hdr.fileSize);

	memcpy(base, &hdr, sizeof(hdr));
	sharedNodes = (PgturbohybridGraphSharedNode *) PgturbohybridGraphSharedCheckedPtr(base, hdr.fileSize, hdr.nodes.offset, hdr.nodes.length);
	codeArena = (uint8 *) PgturbohybridGraphSharedCheckedPtr(base, hdr.fileSize, hdr.codeArena.offset, hdr.codeArena.length);
	payloadArena = (uint8 *) PgturbohybridGraphSharedCheckedPtr(base, hdr.fileSize, hdr.payloadArena.offset, hdr.payloadArena.length);
	residualArena = (uint8 *) PgturbohybridGraphSharedCheckedPtr(base, hdr.fileSize, hdr.residualArena.offset, hdr.residualArena.length);
	exactArena = (char *) PgturbohybridGraphSharedCheckedPtr(base, hdr.fileSize, hdr.exactArena.offset, hdr.exactArena.length);
	neighborCounts = (uint16 *) PgturbohybridGraphSharedCheckedPtr(base, hdr.fileSize, hdr.neighborCounts.offset, hdr.neighborCounts.length);
	neighborOffsets = (uint64 *) PgturbohybridGraphSharedCheckedPtr(base, hdr.fileSize, hdr.neighborOffsets.offset, hdr.neighborOffsets.length);
	neighborData = (uint32 *) PgturbohybridGraphSharedCheckedPtr(base, hdr.fileSize, hdr.neighborData.offset, hdr.neighborData.length);
	payloadRefs = (PgturbohybridGraphPayloadRef *) PgturbohybridGraphSharedCheckedPtr(base, hdr.fileSize, hdr.payloadRefs.offset, hdr.payloadRefs.length);

	for (uint32 nodeId = 0; nodeId < meta->tqNodeCount; nodeId++)
	{
		PgturbohybridGraphScanNode *node = &storage.nodes[nodeId];

		sharedNodes[nodeId].heaptid = node->heaptid;
		sharedNodes[nodeId].payloadMask = node->payloadMask;
		sharedNodes[nodeId].level = node->level;
		sharedNodes[nodeId].exactBlkno = node->exactBlkno;
		sharedNodes[nodeId].exactOffno = node->exactOffno;
		sharedNodes[nodeId].scale = node->scale;
		sharedNodes[nodeId].norm = node->norm;
		sharedNodes[nodeId].codeNorm = node->codeNorm;
		sharedNodes[nodeId].ecCorrection = node->ecCorrection;
		sharedNodes[nodeId].flags = node->flags;
		sharedNodes[nodeId].loaded = node->loaded;
	}
	if (codeBytes > 0)
		memcpy(codeArena, storage.codeArena, codeBytes);
	if (payloadBytes > 0)
		memcpy(payloadArena, storage.payloadArena, payloadBytes);
	if (residualBytes > 0)
		memcpy(residualArena, storage.residualArena, residualBytes);
	if (exactBytes > 0)
		memcpy(exactArena, storage.exactArena, exactBytes);
	memcpy(neighborCounts, storage.neighborCounts, adjRecordCount * sizeof(uint16));
	{
		uint64		cursor = 0;

		for (uint32 slot = 0; slot < adjRecordCount; slot++)
		{
			neighborOffsets[slot] = cursor;
			if (storage.neighborCounts[slot] > 0 && storage.neighbors[slot] != NULL)
			{
				memcpy(&neighborData[cursor], storage.neighbors[slot],
					   storage.neighborCounts[slot] * sizeof(uint32));
				cursor += storage.neighborCounts[slot];
			}
		}
	}
	if (storage.payloadRefCount > 0)
	{
		for (uint32 refIndex = 0; refIndex < storage.payloadRefCount; refIndex++)
		{
			payloadRefs[refIndex].payloadSlot =
				storage.payloadRefs[refIndex].payloadSlot;
			payloadRefs[refIndex].payloadValue =
				storage.payloadRefs[refIndex].payloadValue;
			payloadRefs[refIndex].nodeId =
				storage.payloadRefs[refIndex].nodeId;
		}
	}
	if (!PgturbohybridGraphValidateSharedLayout(base, &hdr, meta, hdr.fileSize))
	{
		munmap(base, (Size) hdr.fileSize);
		close(fd);
		unlink(tmpPath);
		return false;
	}
	PgturbohybridGraphSharedTestPause("before_fsync");

	{
		/*
		 * Always munmap even if msync failed -- a short-circuiting || would
		 * skip the munmap on msync error and leak the writable mapping for the
		 * backend's lifetime (raw mmap is outside PG's resource owner).
		 */
		int			syncRc = msync(base, (Size) hdr.fileSize, MS_SYNC);
		int			unmapRc = munmap(base, (Size) hdr.fileSize);

		if (syncRc != 0 || unmapRc != 0)
		{
			close(fd);
			unlink(tmpPath);
			return false;
		}
	}
	if (fsync(fd) != 0)
	{
		close(fd);
		unlink(tmpPath);
		return false;
	}
	close(fd);
	PgturbohybridGraphSharedTestPause("before_rename");
	if (rename(tmpPath, path) != 0)
	{
		unlink(tmpPath);
		return false;
	}
	{
		char		parent[MAXPGPATH];
		char	   *slash;
		int			dirFd;

		strlcpy(parent, path, sizeof(parent));
		slash = strrchr(parent, '/');
		if (slash == NULL)
			return false;
		*slash = '\0';
		dirFd = open(parent, O_RDONLY);
		if (dirFd < 0 || fsync(dirFd) != 0)
		{
			if (dirFd >= 0)
				close(dirFd);
			unlink(path);
			return false;
		}
		close(dirFd);
	}

	if (buildUs != NULL)
		*buildUs = PgturbohybridGraphElapsedUsSince(start);
	if (codeLockWaitUs != NULL)
		*codeLockWaitUs = loadStats.graphCodeBufferLockWaitUs;
	if (adjLockWaitUs != NULL)
		*adjLockWaitUs = loadStats.graphAdjBufferLockWaitUs;
	return true;
#endif
}

static PgturbohybridGraphSharedMap *
PgturbohybridGraphGetSharedMap(Relation index, PgturbohybridGraphMetaPageData *meta,
							   int64 *attachUs, int64 *buildUs, int64 *waitUs,
								   int64 *codeLockWaitUs, int64 *adjLockWaitUs,
								   bool *builtThisScan,
								   bool *invalidated,
								   int64 *gcRemovedFiles,
								   PgturbohybridGraphNativeCacheReason *reason)
{
#ifdef WIN32
	(void) index;
	(void) meta;
	(void) invalidated;
	(void) gcRemovedFiles;
	if (attachUs != NULL)
		*attachUs = 0;
	if (buildUs != NULL)
		*buildUs = 0;
	if (waitUs != NULL)
		*waitUs = 0;
	if (codeLockWaitUs != NULL)
		*codeLockWaitUs = 0;
	if (adjLockWaitUs != NULL)
		*adjLockWaitUs = 0;
	if (builtThisScan != NULL)
		*builtThisScan = false;
	if (invalidated != NULL)
		*invalidated = false;
	if (reason != NULL)
		*reason = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_SHARED_ATTACH_FAILED;
	return NULL;
#else
	uint64		key = PgturbohybridGraphSharedCacheKey(index, meta);
	PgturbohybridGraphSharedMap *map;
	char		dir[MAXPGPATH];
	char		path[MAXPGPATH];
	char		lockPath[MAXPGPATH];
	char		tmpPath[MAXPGPATH];
	int			lockFd;
	bool		builder = false;
	bool		invalidFile = false;
	bool		writeOk = false;
	instr_time	waitStart;
	int64		removedFiles = 0;

	if (attachUs != NULL)
		*attachUs = 0;
	if (buildUs != NULL)
		*buildUs = 0;
	if (waitUs != NULL)
		*waitUs = 0;
	if (codeLockWaitUs != NULL)
		*codeLockWaitUs = 0;
	if (adjLockWaitUs != NULL)
		*adjLockWaitUs = 0;
	if (builtThisScan != NULL)
		*builtThisScan = false;
	if (invalidated != NULL)
		*invalidated = false;
	if (gcRemovedFiles != NULL)
		*gcRemovedFiles = 0;

	map = PgturbohybridGraphFindSharedMap(index, meta, key);
	if (map != NULL)
		return map;

	if (!PgturbohybridGraphEnsureSharedCacheDir(index, dir, sizeof(dir)))
	{
		if (reason != NULL)
			*reason = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_SHARED_ATTACH_FAILED;
		return NULL;
	}
	PgturbohybridGraphSharedCachePath(index, meta, key, path, sizeof(path));
	PgturbohybridGraphSharedCacheLockPath(index, meta, key, lockPath, sizeof(lockPath));
	if (snprintf(tmpPath, sizeof(tmpPath), "%s.tmp.%d.%lld", path, MyProcPid,
				  (long long) GetCurrentTimestamp()) >= (int) sizeof(tmpPath))
	{
		if (reason != NULL)
			*reason = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_SHARED_ATTACH_FAILED;
		return NULL;
	}

	if (PgturbohybridGraphMapSharedCacheFile(index, meta, path, key, &map,
										 attachUs, &invalidFile))
	{
		removedFiles += PgturbohybridGraphSharedRunGc();
		if (gcRemovedFiles != NULL)
			*gcRemovedFiles = removedFiles;
		return map;
	}
	if (invalidFile && invalidated != NULL)
		*invalidated = true;
	removedFiles += PgturbohybridGraphSharedRunGc();

	lockFd = OpenTransientFilePerm(lockPath, O_CREAT | O_RDWR, 0600);
	if (lockFd < 0)
	{
		if (reason != NULL)
			*reason = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_SHARED_ATTACH_FAILED;
		return NULL;
	}
	builder = PgturbohybridGraphSharedTryLock(lockFd);

	INSTR_TIME_SET_CURRENT(waitStart);
	while (!builder)
	{
		CHECK_FOR_INTERRUPTS();
		if (PgturbohybridGraphMapSharedCacheFile(index, meta, path, key, &map,
											 attachUs, NULL))
		{
			if (waitUs != NULL)
				*waitUs = PgturbohybridGraphElapsedUsSince(waitStart);
			CloseTransientFile(lockFd);
			if (gcRemovedFiles != NULL)
				*gcRemovedFiles = removedFiles;
			return map;
		}
		builder = PgturbohybridGraphSharedTryLock(lockFd);
		if (builder)
			break;
		if (PgturbohybridGraphElapsedUsSince(waitStart) >= PGTURBOHYBRID_GRAPH_SHARED_CACHE_WAIT_US)
			break;
		pg_usleep(PGTURBOHYBRID_GRAPH_SHARED_CACHE_POLL_US);
	}

	if (waitUs != NULL)
		*waitUs = PgturbohybridGraphElapsedUsSince(waitStart);
	if (!builder)
	{
		CloseTransientFile(lockFd);
		if (reason != NULL)
			*reason = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_SHARED_BUILD_TIMEOUT;
		return NULL;
	}
	PgturbohybridGraphSharedTestPause("after_lock");

	/* The previous owner may have published between our last map and lock. */
	if (PgturbohybridGraphMapSharedCacheFile(index, meta, path, key, &map,
										   attachUs, NULL))
	{
		removedFiles += PgturbohybridGraphSharedRemoveOldGenerations(index, meta, path);
		PgturbohybridGraphSharedUnlock(lockFd);
		CloseTransientFile(lockFd);
		if (gcRemovedFiles != NULL)
			*gcRemovedFiles = removedFiles;
		return map;
	}

	PG_TRY();
	{
		writeOk = PgturbohybridGraphWriteSharedCacheFile(index, meta, path, tmpPath, key,
												buildUs, codeLockWaitUs,
												adjLockWaitUs);
	}
	PG_CATCH();
	{
		unlink(tmpPath);
		PgturbohybridGraphSharedUnlock(lockFd);
		CloseTransientFile(lockFd);
		PG_RE_THROW();
	}
	PG_END_TRY();
	if (!writeOk)
	{
		PgturbohybridGraphSharedUnlock(lockFd);
		CloseTransientFile(lockFd);
		if (reason != NULL)
			*reason = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_SHARED_ATTACH_FAILED;
		return NULL;
	}

	if (builtThisScan != NULL)
		*builtThisScan = true;
	removedFiles += PgturbohybridGraphSharedRemoveOldGenerations(index, meta, path);
	PgturbohybridGraphSharedUnlock(lockFd);
	CloseTransientFile(lockFd);
	if (gcRemovedFiles != NULL)
		*gcRemovedFiles = removedFiles;
	if (PgturbohybridGraphMapSharedCacheFile(index, meta, path, key, &map,
										   attachUs, NULL))
	{
		if (invalidFile && reason != NULL)
			*reason = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_SHARED_INVALIDATED;
		map->builtThisBackend = true;
		map->buildUs = buildUs != NULL ? *buildUs : 0;
		return map;
	}
	if (reason != NULL)
		*reason = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_SHARED_ATTACH_FAILED;
	return NULL;
#endif
}

static const char *
PgturbohybridGraphNativeCacheReasonNameForPrewarm(PgturbohybridGraphNativeCacheReason reason)
{
	switch (reason)
	{
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_SHARED_FITS_MAX_MB:
			return "shared_fits_max_mb";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_AUTO_FITS_MAX_MB:
			return "auto_fits_max_mb";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_PER_BACKEND_FITS_MAX_MB:
			return "per_backend_fits_max_mb";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_POLICY_OFF:
			return "policy_off";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_EXCEEDS_MAX_MB:
			return "exceeds_max_mb";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_SHARED_BUILD_TIMEOUT:
			return "shared_build_timeout";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_SHARED_ATTACH_FAILED:
			return "shared_attach_failed";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_SHARED_INVALIDATED:
			return "shared_invalidated";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_NONE:
		default:
			return "none";
	}
}

static int64
PgturbohybridGraphSharedBackendMapCount(void)
{
	PgturbohybridGraphSharedMap *map;
	int64		count = 0;

	for (map = pgturbohybridGraphSharedMapList; map != NULL; map = map->next)
		count++;
	return count;
}

static void
PgturbohybridPrewarmJsonbAddKey(PgturbohybridJsonbState *state, const char *key)
{
	JsonbValue	value;

	value.type = jbvString;
	value.val.string.val = (char *) key;
	value.val.string.len = strlen(key);
	PgturbohybridJsonbPush(state, WJB_KEY, &value);
}

static void
PgturbohybridPrewarmJsonbAddString(PgturbohybridJsonbState *state,
								   const char *key, const char *val)
{
	JsonbValue	value;

	PgturbohybridPrewarmJsonbAddKey(state, key);
	value.type = jbvString;
	value.val.string.val = (char *) val;
	value.val.string.len = strlen(val);
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

static void
PgturbohybridPrewarmJsonbAddBool(PgturbohybridJsonbState *state,
								 const char *key, bool val)
{
	JsonbValue	value;

	PgturbohybridPrewarmJsonbAddKey(state, key);
	value.type = jbvBool;
	value.val.boolean = val;
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

static void
PgturbohybridPrewarmJsonbAddInt64(PgturbohybridJsonbState *state,
								  const char *key, int64 val)
{
	JsonbValue	value;

	PgturbohybridPrewarmJsonbAddKey(state, key);
	value.type = jbvNumeric;
	value.val.numeric = DatumGetNumeric(DirectFunctionCall1(int8_numeric,
															Int64GetDatum(val)));
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_prewarm);
FUNCTION_PREFIX Datum
pgturbohybrid_prewarm(PG_FUNCTION_ARGS)
{
	Oid			indexOid = PG_GETARG_OID(0);
	Relation	index;
	PgturbohybridGraphMetaPageData meta;
	PgturbohybridGraphSharedMap *sharedMap = NULL;
	PgturbohybridGraphNativeCacheReason reason =
		PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_NONE;
	PgturbohybridGraphSharedCacheHeader *hdr = NULL;
	bool		cacheExactVectors;
	bool		cacheable;
	bool		builtThisScan = false;
	bool		invalidated = false;
	int64		attachUs = 0;
	int64		buildUs = 0;
	int64		waitUs = 0;
	int64		codeLockWaitUs = 0;
	int64		adjLockWaitUs = 0;
	int64		gcRemovedFiles = 0;
	PgturbohybridJsonbState jsonState;
	Jsonb	   *jsonb;

	index = index_open(indexOid, AccessShareLock);
	if (!PgturbohybridGraphReadMeta(index, &meta) ||
		meta.storageKind != PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH_NATIVE)
	{
		index_close(index, AccessShareLock);
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turbohybrid_prewarm requires a native turbohybrid graph index")));
	}

	cacheExactVectors = PgturbohybridGraphShouldCacheExactVectors(index, &meta);
	cacheable = PgturbohybridGraphShouldUseNativeCacheWithPolicy(&meta,
																 cacheExactVectors,
																 PGTURBOHYBRID_NATIVE_CACHE_POLICY_SHARED,
																 &reason);
	if (cacheable)
	{
		sharedMap = PgturbohybridGraphGetSharedMap(index, &meta, &attachUs,
												   &buildUs, &waitUs,
												   &codeLockWaitUs,
												   &adjLockWaitUs,
												   &builtThisScan,
												   &invalidated,
												   &gcRemovedFiles,
												   &reason);
		if (sharedMap != NULL)
			hdr = (PgturbohybridGraphSharedCacheHeader *) sharedMap->base;
	}

	PgturbohybridJsonbStateInit(&jsonState);
	PgturbohybridJsonbBeginObject(&jsonState);
	PgturbohybridPrewarmJsonbAddString(&jsonState, "index",
									   RelationGetRelationName(index));
	PgturbohybridPrewarmJsonbAddInt64(&jsonState, "index_oid",
									  (int64) indexOid);
	PgturbohybridPrewarmJsonbAddString(&jsonState, "native_cache_scope",
									   "shared");
	PgturbohybridPrewarmJsonbAddString(&jsonState, "native_cache_reason",
									   PgturbohybridGraphNativeCacheReasonNameForPrewarm(reason));
	PgturbohybridPrewarmJsonbAddBool(&jsonState, "native_cache_used",
									 sharedMap != NULL);
	PgturbohybridPrewarmJsonbAddBool(&jsonState, "native_cache_built",
									 sharedMap != NULL && builtThisScan);
	PgturbohybridPrewarmJsonbAddBool(&jsonState, "native_cache_attached",
									 sharedMap != NULL);
	PgturbohybridPrewarmJsonbAddBool(&jsonState, "native_cache_reused",
									 sharedMap != NULL && !builtThisScan);
	PgturbohybridPrewarmJsonbAddBool(&jsonState, "native_cache_invalidated",
										 invalidated);
	PgturbohybridPrewarmJsonbAddBool(&jsonState, "built",
										 sharedMap != NULL && builtThisScan);
	PgturbohybridPrewarmJsonbAddBool(&jsonState, "attached", sharedMap != NULL);
	PgturbohybridPrewarmJsonbAddBool(&jsonState, "reused",
										 sharedMap != NULL && !builtThisScan);
	PgturbohybridPrewarmJsonbAddBool(&jsonState, "invalidated", invalidated);
	PgturbohybridPrewarmJsonbAddString(&jsonState, "failure_reason",
									   sharedMap != NULL ? "none" :
									   PgturbohybridGraphNativeCacheReasonNameForPrewarm(reason));
	PgturbohybridPrewarmJsonbAddInt64(&jsonState, "gc_removed_files",
									  gcRemovedFiles);
	PgturbohybridPrewarmJsonbAddInt64(&jsonState, "backend_mappings",
									  PgturbohybridGraphSharedBackendMapCount());
	PgturbohybridPrewarmJsonbAddInt64(&jsonState, "native_cache_attach_us",
									  attachUs);
	PgturbohybridPrewarmJsonbAddInt64(&jsonState, "native_cache_build_us",
									  buildUs);
	PgturbohybridPrewarmJsonbAddInt64(&jsonState, "native_cache_wait_us",
									  waitUs);
	PgturbohybridPrewarmJsonbAddInt64(&jsonState, "native_cache_bytes",
									  hdr != NULL ? (int64) hdr->residentTotalBytes : 0);
	PgturbohybridPrewarmJsonbAddInt64(&jsonState, "effective_bytes",
									  hdr != NULL ? (int64) hdr->residentTotalBytes : 0);
	PgturbohybridPrewarmJsonbAddInt64(&jsonState, "native_cache_code_bytes",
									  hdr != NULL ? (int64) hdr->residentCodeBytes : 0);
	PgturbohybridPrewarmJsonbAddInt64(&jsonState, "native_cache_adj_bytes",
									  hdr != NULL ? (int64) hdr->residentAdjBytes : 0);
	PgturbohybridPrewarmJsonbAddInt64(&jsonState, "native_cache_exact_bytes",
									  hdr != NULL ? (int64) hdr->residentExactBytes : 0);
	PgturbohybridPrewarmJsonbAddInt64(&jsonState, "code_buffer_lock_wait_us",
									  codeLockWaitUs);
	PgturbohybridPrewarmJsonbAddInt64(&jsonState, "adj_buffer_lock_wait_us",
									  adjLockWaitUs);
	jsonb = PgturbohybridJsonbEndObject(&jsonState);

	index_close(index, AccessShareLock);
	PG_RETURN_JSONB_P(jsonb);
}

void
PgturbohybridGraphInitScanStorage(Relation index, PgturbohybridGraphMetaPageData *meta,
					   PgturbohybridGraphScanStorage *storage,
					   PgturbohybridGraphCacheInitInfo *info)
{
	PgturbohybridGraphNativeCache *cache;
	bool		cacheExactVectors = PgturbohybridGraphShouldCacheExactVectors(index, meta);
	PgturbohybridGraphNativeCacheReason reason;
	int			effectivePolicy = PgturbohybridGraphEffectiveScanNativeCachePolicy();
	bool		autoPolicy =
		pgturbohybrid_native_cache_policy == PGTURBOHYBRID_NATIVE_CACHE_POLICY_AUTO;

	if (info != NULL)
	{
		memset(info, 0, sizeof(*info));
		info->policy = pgturbohybrid_native_cache_policy;
		info->refcount = -1;
	}

	if (!PgturbohybridGraphShouldUseNativeCacheWithPolicy(meta, cacheExactVectors,
												  effectivePolicy, &reason))
	{
		PgturbohybridGraphInitScanStorageUncached(meta, storage, cacheExactVectors);
		if (info != NULL)
		{
			info->mode = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_UNCACHED;
			info->reason = reason;
		}
		return;
	}

	if (effectivePolicy == PGTURBOHYBRID_NATIVE_CACHE_POLICY_SHARED)
	{
		PgturbohybridGraphSharedMap *sharedMap;
		int64		attachUs = 0;
		int64		buildUs = 0;
		int64		waitUs = 0;
		int64		codeLockWaitUs = 0;
		int64		adjLockWaitUs = 0;
		bool		builtThisScan = false;

		sharedMap = PgturbohybridGraphGetSharedMap(index, meta, &attachUs, &buildUs,
												   &waitUs, &codeLockWaitUs,
												   &adjLockWaitUs, &builtThisScan,
												   NULL,
												   NULL,
												   &reason);
		if (sharedMap != NULL)
		{
			memcpy(storage, &sharedMap->view, sizeof(PgturbohybridGraphScanStorage));
			PgturbohybridGraphSharedInitStorageScratch(meta, storage);
			if (info != NULL)
			{
				PgturbohybridGraphSharedCacheHeader *hdr =
					(PgturbohybridGraphSharedCacheHeader *) sharedMap->base;

				info->mode = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_SHARED;
				info->reason = reason;
				info->used = true;
				info->reused = !builtThisScan;
				info->builtThisScan = builtThisScan;
				info->attachUs = attachUs;
				info->buildUs = buildUs;
				info->waitUs = waitUs;
				info->refcount = -1;
				info->totalBytes = (int64) hdr->residentTotalBytes;
				info->codeBytes = (int64) hdr->residentCodeBytes;
				info->adjBytes = (int64) hdr->residentAdjBytes;
				info->exactBytes = (int64) hdr->residentExactBytes;
				info->codeBufferLockWaitUs = codeLockWaitUs;
				info->adjBufferLockWaitUs = adjLockWaitUs;
			}
			return;
		}

		if (!autoPolicy)
		{
			PgturbohybridGraphWarnSharedFailureOnce(index, reason);
			PgturbohybridGraphInitScanStorageUncached(meta, storage, cacheExactVectors);
			if (info != NULL)
			{
				info->mode = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_UNCACHED;
				info->reason = reason;
				info->attachUs = attachUs;
				info->waitUs = waitUs;
			}
			return;
		}

		/* auto: preserve the shared failure reason while using per-backend. */
	}

	cache = PgturbohybridGraphFindCache(index, meta);
	if (cache == NULL)
	{
		cache = PgturbohybridGraphBuildCache(index, meta);
		if (info != NULL)
		{
			info->builtThisScan = true;
			info->buildUs = cache->buildUs;
		}
	}
	else if (info != NULL)
		info->reused = true;

	memcpy(storage, &cache->storage, sizeof(PgturbohybridGraphScanStorage));
	storage->cached = true;

	if (info != NULL)
	{
		info->mode = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_PER_BACKEND;
		info->reason = reason;
		info->used = true;
		info->totalBytes = (int64) cache->residentTotalBytes;
		info->codeBytes = (int64) cache->residentCodeBytes;
		info->adjBytes = (int64) cache->residentAdjBytes;
		info->exactBytes = (int64) cache->residentExactBytes;
		info->warning = PgturbohybridGraphNativeCacheWarns(cache);
		info->warningReason = info->warning ?
			"per_backend_resident_bytes_exceed_warn_mb" : "none";
		if (info->builtThisScan)
		{
			info->codeBufferLockWaitUs = cache->buildCodeBufferLockWaitUs;
			info->adjBufferLockWaitUs = cache->buildAdjBufferLockWaitUs;
		}
	}
}

PgturbohybridGraphNativeCache *
PgturbohybridGraphInitInsertStorage(Relation index, PgturbohybridGraphMetaPageData *meta,
						 PgturbohybridGraphScanStorage *storage)
{
	PgturbohybridGraphNativeCache *cache;
	bool		cacheExactVectors = PgturbohybridGraphShouldCacheExactVectors(index, meta);

	if (PgturbohybridGraphEffectiveScanNativeCachePolicy() ==
		PGTURBOHYBRID_NATIVE_CACHE_POLICY_SHARED)
	{
		/* Inserts need mutable cache state; shared-cache views are scan-only. */
		PgturbohybridGraphInitScanStorageUncached(meta, storage, cacheExactVectors);
		return NULL;
	}

	if (BlockNumberIsValid(meta->tqMultivectorDocMapStartBlkno))
	{
		/*
		 * Multivector row inserts append graph nodes one subvector at a time,
		 * then append the docmap entry once the row is complete.  Avoid
		 * validating a transiently incomplete sidecar while building insert
		 * storage for the later subvectors.
		 */
		PgturbohybridGraphInitScanStorageUncached(meta, storage, cacheExactVectors);
		return NULL;
	}

	if (!PgturbohybridGraphShouldUseNativeCache(meta, cacheExactVectors, NULL))
	{
		PgturbohybridGraphInitScanStorageUncached(meta, storage, cacheExactVectors);
		return NULL;
	}

	cache = PgturbohybridGraphFindCache(index, meta);
	if (cache == NULL)
		cache = PgturbohybridGraphBuildCache(index, meta);

	memcpy(storage, &cache->storage, sizeof(PgturbohybridGraphScanStorage));
	/*
	 * The native cache loads adjacency tuple locations with the adjacency
	 * records.  Single-row reciprocal updates use these block/offset arrays to
	 * avoid adjacency page-chain scans whenever cached insert storage is valid.
	 */
	Assert(cache->storage.adjBlknos != NULL);
	Assert(cache->storage.adjOffnos != NULL);
	storage->cached = true;
	return cache;
}

static uint32 *
PgturbohybridGraphCacheCopyNeighbors(PgturbohybridGraphScanStorage *storage, uint32 *neighbors,
						  int count)
{
	uint32	   *copy;

	if (count <= 0)
		return NULL;

	copy = MemoryContextAlloc(storage->ctx, sizeof(uint32) * count);
	memcpy(copy, neighbors, sizeof(uint32) * count);
	return copy;
}

void
PgturbohybridGraphAppendInsertCacheNode(PgturbohybridGraphNativeCache *cache, PgturbohybridGraphMetaPageData *meta,
							 uint32 nodeId, ItemPointer heapTid,
							 int nodeLevel, Vector *vector, uint8 *code,
							 uint8 *residualSketch,
							 float scale, float norm, float codeNorm,
							 float ecCorrection, int32 *payloads,
							 uint16 payloadMask, BlockNumber exactBlkno,
							 OffsetNumber exactOffno, uint32 **selected,
							 int *selectedCounts, BlockNumber *adjBlknos,
							 OffsetNumber *adjOffnos, BlockNumber codeStart,
							 BlockNumber adjStart, BlockNumber exactStart,
							 uint32 entryNodeId, uint16 graphMaxLevel)
{
	PgturbohybridGraphScanStorage *storage;
	uint32		oldNodeCount;
	uint32		newNodeCount;
	int			oldAdjRecordCount;
	int			newAdjRecordCount;
	int			oldCodePageCount;
	int			newCodePageCount;
	Size		codeBytes;
	Size		payloadBytes;
	Size		residualBytes;
	MemoryContext oldCtx;

	if (cache == NULL || nodeId != cache->tqNodeCount)
		return;

	storage = &cache->storage;
	oldNodeCount = cache->tqNodeCount;
	newNodeCount = oldNodeCount + 1;
	oldAdjRecordCount = oldNodeCount * PgturbohybridGraphLevelCapacity(meta->m);
	newAdjRecordCount = newNodeCount * PgturbohybridGraphLevelCapacity(meta->m);
	oldCodePageCount = storage->codePageCount;
	newCodePageCount = PgturbohybridGraphPageCount(newNodeCount, storage->codeTuplesPerPage);
	codeBytes = meta->tqCodeBytes;
	payloadBytes = meta->tqPayloadBytes;
	residualBytes = meta->tqResidualRerankBytes;

	oldCtx = MemoryContextSwitchTo(cache->ctx);

	storage->nodes = repalloc(storage->nodes,
							  PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridGraphScanNode),
															 newNodeCount,
															 "pgturbohybrid graph scan node cache"));
	memset(&storage->nodes[nodeId], 0, sizeof(PgturbohybridGraphScanNode));
	if (codeBytes > 0)
	{
		/* repalloc_huge: the code arena may exceed 1 GB (see ArenaBytesHuge). */
		storage->codeArena = repalloc_huge(storage->codeArena,
										   (Size) codeBytes * (Size) newNodeCount);
		memcpy(storage->codeArena + ((Size) nodeId * codeBytes),
			   code, codeBytes);
	}
	if (payloadBytes > 0)
	{
		storage->payloadArena = repalloc_huge(storage->payloadArena,
											  (Size) payloadBytes * (Size) newNodeCount);
		if (payloads != NULL)
			memcpy(storage->payloadArena + ((Size) nodeId * payloadBytes),
				   payloads, payloadBytes);
		else
			memset(storage->payloadArena + ((Size) nodeId * payloadBytes),
				   0, payloadBytes);
	}
	if (residualBytes > 0)
	{
		storage->residualArena = repalloc_huge(storage->residualArena,
											   (Size) residualBytes * (Size) newNodeCount);
		if (residualSketch != NULL)
			memcpy(storage->residualArena + ((Size) nodeId * residualBytes),
				   residualSketch, residualBytes);
		else
			memset(storage->residualArena + ((Size) nodeId * residualBytes),
				   0, residualBytes);
	}
	if (storage->exactArena != NULL)
	{
		storage->exactArena = repalloc(storage->exactArena,
									   PgturbohybridCheckedArrayBytes(storage->exactBytes,
																	  newNodeCount,
																	  "pgturbohybrid graph exact-vector arena"));
		memcpy(storage->exactArena + ((Size) nodeId * storage->exactBytes),
			   vector, storage->exactBytes);
	}
	if (storage->visitedGeneration != NULL)
	{
		storage->visitedGeneration = repalloc(storage->visitedGeneration,
											  PgturbohybridCheckedArrayBytes(sizeof(uint32),
																			 newNodeCount,
																			 "pgturbohybrid graph visited generation cache"));
		storage->visitedGeneration[nodeId] = 0;
	}
	if (newCodePageCount != oldCodePageCount)
	{
		storage->codePagesLoaded = repalloc(storage->codePagesLoaded,
											PgturbohybridCheckedArrayBytes(sizeof(bool),
																		   newCodePageCount,
																		   "pgturbohybrid graph code page cache"));
		storage->codeBlknos = repalloc(storage->codeBlknos,
									   PgturbohybridCheckedArrayBytes(sizeof(BlockNumber),
																	  newCodePageCount,
																	  "pgturbohybrid graph code block map"));
		for (int i = oldCodePageCount; i < newCodePageCount; i++)
		{
			storage->codePagesLoaded[i] = true;
			storage->codeBlknos[i] = InvalidBlockNumber;
		}
		storage->codePageCount = newCodePageCount;
	}

	storage->neighbors = repalloc(storage->neighbors,
								  PgturbohybridCheckedArrayBytes(sizeof(uint32 *),
																 newAdjRecordCount,
																 "pgturbohybrid graph neighbor cache"));
	storage->neighborCounts = repalloc(storage->neighborCounts,
									   PgturbohybridCheckedArrayBytes(sizeof(uint16),
																	  newAdjRecordCount,
																	  "pgturbohybrid graph neighbor count cache"));
	storage->adjBlknos = repalloc(storage->adjBlknos,
								  PgturbohybridCheckedArrayBytes(sizeof(BlockNumber),
																 newAdjRecordCount,
																 "pgturbohybrid graph adjacency block map"));
	storage->adjOffnos = repalloc(storage->adjOffnos,
								  PgturbohybridCheckedArrayBytes(sizeof(OffsetNumber),
																 newAdjRecordCount,
																 "pgturbohybrid graph adjacency offset map"));
	memset(&storage->neighbors[oldAdjRecordCount], 0,
		   sizeof(uint32 *) * (newAdjRecordCount - oldAdjRecordCount));
	memset(&storage->neighborCounts[oldAdjRecordCount], 0,
		   sizeof(uint16) * (newAdjRecordCount - oldAdjRecordCount));
	for (int i = oldAdjRecordCount; i < newAdjRecordCount; i++)
	{
		storage->adjBlknos[i] = InvalidBlockNumber;
		storage->adjOffnos[i] = InvalidOffsetNumber;
	}

	for (int level = 0; level < PgturbohybridGraphLevelCapacity(meta->m); level++)
	{
		int			slot = PgturbohybridGraphAdjSlot(meta, nodeId, level);
		int			count = level <= nodeLevel ? selectedCounts[level] : 0;

		storage->neighborCounts[slot] = count;
		if (level <= nodeLevel)
		{
			storage->adjBlknos[slot] = adjBlknos[level];
			storage->adjOffnos[slot] = adjOffnos[level];
		}
		storage->neighbors[slot] =
			PgturbohybridGraphCacheCopyNeighbors(storage,
									  level <= nodeLevel ? selected[level] : NULL,
									  count);
	}

	storage->nodes[nodeId].heaptid = *heapTid;
	storage->nodes[nodeId].level = nodeLevel;
	storage->nodes[nodeId].exactBlkno = exactBlkno;
	storage->nodes[nodeId].exactOffno = exactOffno;
	storage->nodes[nodeId].payloadMask = payloadMask;
	storage->nodes[nodeId].scale = scale;
	storage->nodes[nodeId].norm = norm;
	storage->nodes[nodeId].codeNorm = codeNorm;
	storage->nodes[nodeId].ecCorrection = ecCorrection;
	storage->nodes[nodeId].flags = 0;
	storage->nodes[nodeId].loaded = true;
	if (codeBytes > 0)
		storage->nodes[nodeId].code = storage->codeArena + ((Size) nodeId * codeBytes);
	if (payloadBytes > 0)
		storage->nodes[nodeId].payloads =
			(int32 *) (storage->payloadArena + ((Size) nodeId * payloadBytes));
	if (residualBytes > 0)
		storage->nodes[nodeId].residualSketch =
			storage->residualArena + ((Size) nodeId * residualBytes);
	if (storage->exactArena != NULL)
		storage->nodes[nodeId].exactVector =
			storage->exactArena + ((Size) nodeId * storage->exactBytes);

	if (payloadBytes > 0 && payloads != NULL && payloadMask != 0)
	{
		uint32		addedRefs = 0;

		for (int slot = 0; slot < meta->tqPayloadCount; slot++)
		{
			if (payloadMask & (uint16) (1U << slot))
				addedRefs++;
		}
		if (addedRefs > 0)
		{
			uint32		oldRefCount = storage->payloadRefCount;
			uint32		refIndex = oldRefCount;

			if (storage->payloadRefs == NULL)
				storage->payloadRefs =
					palloc0(sizeof(PgturbohybridGraphPayloadRef) * addedRefs);
			else
			{
				storage->payloadRefs = repalloc(storage->payloadRefs,
												sizeof(PgturbohybridGraphPayloadRef) *
												(oldRefCount + addedRefs));
				memset(&storage->payloadRefs[oldRefCount], 0,
					   sizeof(PgturbohybridGraphPayloadRef) * addedRefs);
			}
			for (int slot = 0; slot < meta->tqPayloadCount; slot++)
			{
				if ((payloadMask & (uint16) (1U << slot)) == 0)
					continue;
				storage->payloadRefs[refIndex].payloadSlot = (int16) slot;
				storage->payloadRefs[refIndex].payloadValue = payloads[slot];
				storage->payloadRefs[refIndex].nodeId = nodeId;
				refIndex++;
			}
			storage->payloadRefCount = oldRefCount + addedRefs;
			qsort(storage->payloadRefs, storage->payloadRefCount,
				  sizeof(PgturbohybridGraphPayloadRef), PgturbohybridGraphPayloadRefCompare);
		}
	}

	cache->tqNodeCount = newNodeCount;
	cache->tqEntryNodeId = entryNodeId;
	cache->graphMaxLevel = graphMaxLevel;
	cache->tqResidualRerankBytes = meta->tqResidualRerankBytes;
	cache->tqCodeStartBlkno = codeStart;
	cache->tqAdjStartBlkno = adjStart;
	cache->tqExactStartBlkno = exactStart;

	MemoryContextSwitchTo(oldCtx);
}
