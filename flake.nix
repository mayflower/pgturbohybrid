{
  description = "Development environment for the pgturbohybrid PostgreSQL extension";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    pgvector-v082 = {
      url = "github:pgvector/pgvector/v0.8.2";
      flake = false;
    };
    pgvector-master = {
      url = "github:pgvector/pgvector/master";
      flake = false;
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      pgvector-v082,
      pgvector-master,
    }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      forAllSystems = nixpkgs.lib.genAttrs systems;

      mkForSystem =
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          lib = pkgs.lib;
          postgresql = pkgs.postgresql_17;
          llamaCppColbert = pkgs.llama-cpp.overrideAttrs (old: {
            patches = (old.patches or [ ]) ++ [
              ./nix/patches/llama-cpp-get-key-or-arr-kv-overrides.patch
            ];
            preConfigure = ''
              prependToVar cmakeFlags "-DLLAMA_BUILD_COMMIT:STRING=$(cat COMMIT)"
            '';
            postInstall = ''
              ln -sf $out/bin/llama-cli $out/bin/llama

              mkdir -p $out/include
              cp $src/include/llama.h $out/include/
            '';
          });
          tapPerl = pkgs.perl.withPackages (ps: [
            ps.IPCRun
            ps.IOTty
          ]);

          cleanSource = lib.cleanSourceWith {
            src = ./.;
            filter =
              name: type:
              let
                base = baseNameOf name;
              in
              !(lib.elem base [
                ".nix-dev"
                ".deps"
                "dist"
                "log"
                "tmp_check"
                "tmp_check_iso"
                "output_iso"
                "results"
              ]);
          };

          mkPgvector =
            {
              src,
              version,
              name,
            }:
            postgresql.pkgs.callPackage (
              {
                lib,
                postgresql,
                postgresqlBuildExtension,
              }:
              postgresqlBuildExtension {
                pname = name;
                inherit src version;
                enableUpdateScript = false;

                meta = {
                  description = "Open-source vector similarity search for PostgreSQL";
                  homepage = "https://github.com/pgvector/pgvector";
                  license = lib.licenses.postgresql;
                  platforms = postgresql.meta.platforms;
                };
              }
            ) { };

          pgvectorStable = mkPgvector {
            name = "pgvector";
            version = "0.8.2";
            src = pgvector-v082;
          };

          pgvectorMaster = mkPgvector {
            name = "pgvector";
            version = "master";
            src = pgvector-master;
          };

          mkPgturbohybrid =
            {
              pgvector,
              simdBuild ? "portable",
            }:
            postgresql.pkgs.callPackage (
              {
                lib,
                postgresql,
                postgresqlBuildExtension,
              }:
              postgresqlBuildExtension {
                pname = "pgturbohybrid";
                version = "0.1.0";
                src = cleanSource;
                enableUpdateScript = false;

                buildInputs = [ pgvector ];
                makeFlags = [
                  "VECTOR_INCLUDE=${pgvector}/include/server/extension/vector"
                  "PGTURBOHYBRID_REQUIRE_VECTOR_HEADER=1"
                  "SIMD_BUILD=${simdBuild}"
                  "MATH_MODE=strict"
                ];

                meta = {
                  description = "Hybrid vector and BM25 search for PostgreSQL on top of pgvector";
                  homepage = "https://github.com/agentxagi/pgturbohybrid";
                  license = lib.licenses.postgresql;
                  platforms = postgresql.meta.platforms;
                };
              }
            ) { };

          mkPgColbertLlama =
            {
              pgvector,
              pgturbohybrid,
              engine ? "stub",
            }:
            postgresql.pkgs.callPackage (
              {
                lib,
                postgresql,
                postgresqlBuildExtension,
              }:
              postgresqlBuildExtension {
                pname = "pg_colbert_llama";
                version = "0.1.0";
                src = cleanSource;
                sourceRoot = "source/extensions/pg_colbert_llama";
                enableUpdateScript = false;

                buildInputs =
                  [
                    pgvector
                    pgturbohybrid
                  ]
                  ++ lib.optionals (engine == "llama") [
                    llamaCppColbert
                  ];
                makeFlags =
                  [
                    "VECTOR_INCLUDE=${pgvector}/include/server/extension/vector"
                    "PG_COLBERT_LLAMA_ENGINE=${engine}"
                  ]
                  ++ lib.optionals (engine == "llama") [
                    "LLAMA_CPP_INCLUDE=${llamaCppColbert}/include"
                    "LLAMA_CPP_LIB=${llamaCppColbert}/lib"
                    "LLAMA_CPP_BACKEND_DIR=${llamaCppColbert}/bin"
                    "LLAMA_CPP_LDFLAGS=-Wl,-rpath,${llamaCppColbert}/lib"
                  ];

                meta = {
                  description = "ColBERT embedding companion extension for pgturbohybrid";
                  homepage = "https://github.com/agentxagi/pgturbohybrid";
                  license = lib.licenses.postgresql;
                  platforms = postgresql.meta.platforms;
                };
              }
            ) { };

          mkDevSet =
            {
              suffix,
              pgvector,
            }:
            let
              pgturbohybrid = mkPgturbohybrid { inherit pgvector; };
              pgColbertLlamaStub = mkPgColbertLlama {
                inherit pgvector pgturbohybrid;
                engine = "stub";
              };
              pgColbertLlamaLlama = mkPgColbertLlama {
                inherit pgvector pgturbohybrid;
                engine = "llama";
              };
              postgresWithExtensions = postgresql.withPackages (
                _: [
                  pgvector
                  pgturbohybrid
                  pgColbertLlamaStub
                ]
              );
              postgresWithLlamaExtensions = postgresql.withPackages (
                _: [
                  pgvector
                  pgturbohybrid
                  pgColbertLlamaLlama
                ]
              );

              commonEnv = {
                TH_ENV_NAME = "pg17-${suffix}";
                TH_PGPORT = "55432";
                PGPORT = "55432";
                PGDATABASE = "pgturbohybrid_dev";
                PGUSER = "postgres";
                PG_CONFIG = "${postgresql.pg_config}/bin/pg_config";
                VECTOR_INCLUDE = "${pgvector}/include/server/extension/vector";
              };

              mkScript =
                name: text: mkScriptWithInputs [ ] name text;

              mkScriptWithInputs =
                extraRuntimeInputs: name: text:
                pkgs.writeShellApplication {
                  inherit name;
                  runtimeInputs =
                    [
                      postgresWithExtensions
                      pkgs.coreutils
                      pkgs.gnused
                      pkgs.gnugrep
                      pkgs.gnumake
                    ]
                    ++ extraRuntimeInputs;
                  text = ''
                    set -euo pipefail

                    if [ -n "''${PGDATABASE+x}" ]; then
                      export TH_PGDATABASE_WAS_EXPLICIT=1
                    else
                      export TH_PGDATABASE_WAS_EXPLICIT=0
                    fi

                    export TH_ENV_NAME="''${TH_ENV_NAME:-${commonEnv.TH_ENV_NAME}}"
                    export TH_PGPORT="''${TH_PGPORT:-${commonEnv.TH_PGPORT}}"
                    export PGPORT="''${PGPORT:-$TH_PGPORT}"
                    export PGDATABASE="''${PGDATABASE:-${commonEnv.PGDATABASE}}"
                    export PGUSER="''${PGUSER:-${commonEnv.PGUSER}}"
                    export TH_ROOT="''${TH_ROOT:-$PWD}"
                    export TH_STATE_DIR="''${TH_STATE_DIR:-$TH_ROOT/.nix-dev/$TH_ENV_NAME}"
                    export PGDATA="''${PGDATA:-$TH_STATE_DIR/pgdata}"
                    export PGHOST="''${PGHOST:-$TH_STATE_DIR/run}"
                    export TH_LOG_DIR="''${TH_LOG_DIR:-$TH_STATE_DIR/log}"
                    export TH_LOG_FILE="''${TH_LOG_FILE:-$TH_LOG_DIR/postgres.log}"
                    export PG_CONFIG="${postgresql.pg_config}/bin/pg_config"
                    export VECTOR_INCLUDE="${pgvector}/include/server/extension/vector"
                    export TH_EMPTY_POSTGRES_LISTEN_ADDRESS=""

                    mkdir -p "$TH_STATE_DIR" "$PGHOST" "$TH_LOG_DIR"

                    ${text}
                  '';
                };

              pgInit = mkScript "th-pg-init" ''
                if [ ! -s "$PGDATA/PG_VERSION" ]; then
                  initdb -D "$PGDATA" --username="$PGUSER" --auth=trust --no-locale --encoding=UTF8
                  {
                    echo "unix_socket_directories = '$PGHOST'"
                    echo "port = $TH_PGPORT"
                    if [ "''${TH_POSTGRES_SOCKET_ONLY:-0}" = "1" ]; then
                      echo "listen_addresses = '$TH_EMPTY_POSTGRES_LISTEN_ADDRESS'"
                    else
                      echo "listen_addresses = 'localhost'"
                    fi
                    echo "log_min_messages = warning"
                    echo "client_min_messages = warning"
                  } >> "$PGDATA/postgresql.conf"
                fi

                if ! pg_ctl -D "$PGDATA" status >/dev/null 2>&1; then
                  if ! pg_ctl -D "$PGDATA" -l "$TH_LOG_FILE" -o "-k '$PGHOST' -p '$TH_PGPORT'" start; then
                    if [ -f "$TH_LOG_FILE" ]; then
                      cat "$TH_LOG_FILE" >&2
                    fi
                    exit 1
                  fi
                fi

                createdb --host="$PGHOST" --port="$TH_PGPORT" --username="$PGUSER" "$PGDATABASE" >/dev/null 2>&1 || true
                psql --host="$PGHOST" --port="$TH_PGPORT" --username="$PGUSER" --dbname="$PGDATABASE" -v ON_ERROR_STOP=1 <<'SQL'
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
                CREATE EXTENSION IF NOT EXISTS pg_colbert_llama;
                SQL

                printf 'ready: %s on socket %s port %s database %s\n' "$TH_ENV_NAME" "$PGHOST" "$TH_PGPORT" "$PGDATABASE"
              '';

              pgStart = mkScript "th-pg-start" ''
                if [ ! -s "$PGDATA/PG_VERSION" ]; then
                  ${pgInit}/bin/th-pg-init
                  exit 0
                fi

                if pg_ctl -D "$PGDATA" status >/dev/null 2>&1; then
                  printf 'already running: %s\n' "$PGDATA"
                  exit 0
                fi

                pg_ctl -D "$PGDATA" -l "$TH_LOG_FILE" -o "-k '$PGHOST' -p '$TH_PGPORT'" start
              '';

              pgStop = mkScript "th-pg-stop" ''
                if [ -s "$PGDATA/PG_VERSION" ] && pg_ctl -D "$PGDATA" status >/dev/null 2>&1; then
                  pg_ctl -D "$PGDATA" stop -m fast
                else
                  printf 'not running: %s\n' "$PGDATA"
                fi
              '';

              pgReset = mkScript "th-pg-reset" ''
                if [ -s "$PGDATA/PG_VERSION" ] && pg_ctl -D "$PGDATA" status >/dev/null 2>&1; then
                  pg_ctl -D "$PGDATA" stop -m fast
                fi
                rm -rf "$TH_STATE_DIR"
                ${pgInit}/bin/th-pg-init
              '';

              colbertLlamaEnv = ''
                export PATH="${postgresWithLlamaExtensions}/bin:$PATH"
                export TH_ENV_NAME="''${TH_COLBERT_LLAMA_ENV_NAME:-${commonEnv.TH_ENV_NAME}-colbert-llama}"
                export TH_PGPORT="''${TH_COLBERT_LLAMA_PGPORT:-55433}"
                export PGPORT="$TH_PGPORT"
                export TH_POSTGRES_SOCKET_ONLY="''${TH_COLBERT_LLAMA_SOCKET_ONLY:-1}"
                export TH_STATE_DIR="$TH_ROOT/.nix-dev/$TH_ENV_NAME"
                export PGDATA="$TH_STATE_DIR/pgdata"
                export PGHOST="$TH_STATE_DIR/run"
                export TH_LOG_DIR="$TH_STATE_DIR/log"
                export TH_LOG_FILE="$TH_LOG_DIR/postgres.log"
                mkdir -p "$TH_STATE_DIR" "$PGHOST" "$TH_LOG_DIR"
              '';

              pgInitColbertLlama = mkScript "th-pg-init-colbert-llama" ''
                ${colbertLlamaEnv}

                if [ ! -s "$PGDATA/PG_VERSION" ]; then
                  initdb -D "$PGDATA" --username="$PGUSER" --auth=trust --no-locale --encoding=UTF8
                  {
                    echo "unix_socket_directories = '$PGHOST'"
                    echo "port = $TH_PGPORT"
                    if [ "''${TH_POSTGRES_SOCKET_ONLY:-0}" = "1" ]; then
                      echo "listen_addresses = '$TH_EMPTY_POSTGRES_LISTEN_ADDRESS'"
                    else
                      echo "listen_addresses = 'localhost'"
                    fi
                    echo "log_min_messages = warning"
                    echo "client_min_messages = warning"
                  } >> "$PGDATA/postgresql.conf"
                fi

                if ! pg_ctl -D "$PGDATA" status >/dev/null 2>&1; then
                  if ! pg_ctl -D "$PGDATA" -l "$TH_LOG_FILE" -o "-k '$PGHOST' -p '$TH_PGPORT' -c listen_addresses=\"\"" start; then
                    if [ -f "$TH_LOG_FILE" ]; then
                      cat "$TH_LOG_FILE" >&2
                    fi
                    exit 1
                  fi
                fi

                createdb --host="$PGHOST" --port="$TH_PGPORT" --username="$PGUSER" "$PGDATABASE" >/dev/null 2>&1 || true
                psql --host="$PGHOST" --port="$TH_PGPORT" --username="$PGUSER" --dbname="$PGDATABASE" -v ON_ERROR_STOP=1 <<'SQL'
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
                CREATE EXTENSION IF NOT EXISTS pg_colbert_llama;
                SQL

                printf 'ready: %s on socket %s port %s database %s\n' "$TH_ENV_NAME" "$PGHOST" "$TH_PGPORT" "$PGDATABASE"
              '';

              psqlScript = mkScript "th-psql" ''
                exec psql --host="$PGHOST" --port="$TH_PGPORT" --username="$PGUSER" --dbname="$PGDATABASE" "$@"
              '';

              psqlFileScript =
                name: sqlFile:
                mkScript name ''
                  ${pgInit}/bin/th-pg-init >/dev/null
                  exec psql \
                    --host="$PGHOST" \
                    --port="$TH_PGPORT" \
                    --username="$PGUSER" \
                    --dbname="$PGDATABASE" \
                    -v ON_ERROR_STOP=1 \
                    "$@" \
                    -f "$TH_ROOT/${sqlFile}"
                '';

              smoke = mkScript "th-smoke" ''
                ${pgInit}/bin/th-pg-init >/dev/null
                psql --host="$PGHOST" --port="$TH_PGPORT" --username="$PGUSER" --dbname="$PGDATABASE" -v ON_ERROR_STOP=1 <<'SQL'
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
                SELECT '[1,0,0]'::vector <~-> turbohybrid_query(vector_query => '[1,0,0]'::vector) AS distance;
                SQL
              '';

              installcheck = mkScript "th-installcheck" ''
                ${pgInit}/bin/th-pg-init >/dev/null
                if [ -n "''${PGOPTIONS:-}" ]; then
                  export PGOPTIONS="$PGOPTIONS -c client_min_messages=notice"
                else
                  export PGOPTIONS="-c client_min_messages=notice"
                fi
                exec make PG_CONFIG="$PG_CONFIG" installcheck
              '';

              proveInstallcheck = mkScriptWithInputs [ tapPerl ] "th-prove-installcheck" ''
                ${pgInit}/bin/th-pg-init >/dev/null
                export PERL5LIB="${postgresql.src}/src/test/perl:${postgresql.src}/test/perl:''${PERL5LIB:-}"
                export PG_PROVE_FLAGS="-I ${postgresql.src}/src/test/perl ''${PG_PROVE_FLAGS:-}"
                exec make \
                  PG_CONFIG="$PG_CONFIG" \
                  bindir="${postgresWithExtensions}/bin" \
                  PG_PROVE_FLAGS="-I ${postgresql.src}/src/test/perl ''${PG_PROVE_FLAGS:-}" \
                  prove_installcheck
              '';

              test = mkScript "th-test" ''
                ${smoke}/bin/th-smoke
                ${installcheck}/bin/th-installcheck
              '';

              benchRetrievalQuality =
                psqlFileScript "th-bench-retrieval-quality" "benchmarks/dev/retrieval_quality_grid.sql";

              benchProfileGrid =
                psqlFileScript "th-bench-profile-grid" "benchmarks/dev/profile_recall_latency_grid.sql";

              benchResidualRerank =
                psqlFileScript "th-bench-residual-rerank" "benchmarks/dev/residual_rerank_grid.sql";

              benchBm25PhraseRerank =
                psqlFileScript "th-bench-bm25-phrase-rerank" "benchmarks/dev/bm25_phrase_rerank_grid.sql";

              benchDenseCandidateMiss =
                psqlFileScript "th-bench-dense-candidate-miss" "benchmarks/dev/dense_candidate_miss_grid.sql";

              benchNativeCache =
                psqlFileScript "th-bench-native-cache-memory" "benchmarks/dev/native_cache_memory_bench.sql";

              benchTuneProfile = mkScript "th-bench-tune-profile" ''
                ${pgInit}/bin/th-pg-init >/dev/null
                exec psql \
                  --host="$PGHOST" \
                  --port="$TH_PGPORT" \
                  --username="$PGUSER" \
                  --dbname="$PGDATABASE" \
                  -v ON_ERROR_STOP=1 \
                  -v INDEX_NAME="''${INDEX_NAME:-documents_turbohybrid_idx}" \
                  -v TABLE_NAME="''${TABLE_NAME:-documents}" \
                  -v ID_COLUMN="''${ID_COLUMN:-id}" \
                  -v LIMIT_K="''${LIMIT_K:-10}" \
                  -v MAX_TRIALS="''${MAX_TRIALS:-96}" \
                  -v EVAL_QUERY_TABLE="''${EVAL_QUERY_TABLE:-eval_queries}" \
                  -v LATENCY_BUDGET_MS="''${LATENCY_BUDGET_MS:-20}" \
                  "$@" \
                  -f "$TH_ROOT/benchmarks/dev/tune_retrieval_profile.sql"
              '';

              benchFiqaQuick = mkScript "th-bench-fiqa-quick" ''
                if [ "''${TH_PGDATABASE_WAS_EXPLICIT:-0}" != "1" ]; then
                  export PGDATABASE="''${FIQA_PGDATABASE:-pgturbohybrid_fiqa_quick}"
                fi
                ${pgInit}/bin/th-pg-init >/dev/null
                export PGHOST PGPORT PGUSER PGDATABASE PG_CONFIG VECTOR_INCLUDE
                exec "$TH_ROOT/benchmarks/dev/run_fiqa_quick.sh" "$@"
              '';

              benchConcurrentDense = mkScriptWithInputs [ pkgs.uv ] "th-bench-concurrent-dense" ''
                ${pgInit}/bin/th-pg-init >/dev/null
                export PGHOST PGPORT PGUSER PGDATABASE
                exec uv run "$TH_ROOT/benchmarks/concurrent_dense_bench.py" "$@"
              '';

              benchDbpediaColbert = mkScriptWithInputs [ pkgs.uv ] "th-bench-dbpedia-colbert" ''
                for arg in "$@"; do
                  case "$arg" in
                    -h|--help)
                      exec uv run "$TH_ROOT/benchmarks/dbpedia_colbert_multivector.py" "$@"
                      ;;
                  esac
                done
                if [ "''${TH_PGDATABASE_WAS_EXPLICIT:-0}" != "1" ]; then
                  export PGDATABASE="''${DBPEDIA_COLBERT_PGDATABASE:-pgturbohybrid_dbpedia_colbert}"
                fi
                ${colbertLlamaEnv}
                ${pgInitColbertLlama}/bin/th-pg-init-colbert-llama >/dev/null
                export PGHOST PGPORT PGUSER PGDATABASE
                exec uv run "$TH_ROOT/benchmarks/dbpedia_colbert_multivector.py" "$@"
              '';

              dbpediaColbertHfDataset = mkScriptWithInputs [ pkgs.uv ] "th-dbpedia-colbert-hf-dataset" ''
                if [ "''${TH_PGDATABASE_WAS_EXPLICIT:-0}" != "1" ]; then
                  export PGDATABASE="''${DBPEDIA_COLBERT_PGDATABASE:-pgturbohybrid_dbpedia_colbert}"
                fi
                ${colbertLlamaEnv}
                ${pgInitColbertLlama}/bin/th-pg-init-colbert-llama >/dev/null
                export PGHOST PGPORT PGUSER PGDATABASE
                exec uv run "$TH_ROOT/benchmarks/dbpedia_colbert_hf_dataset.py" "$@"
              '';

              dbpediaColbertGenerateExport = mkScriptWithInputs [ pkgs.uv ] "th-dbpedia-colbert-generate-export" ''
                for arg in "$@"; do
                  case "$arg" in
                    -h|--help)
                      exec uv run "$TH_ROOT/benchmarks/dbpedia_colbert_generate_export.py" "$@"
                      ;;
                  esac
                done
                if [ "''${TH_PGDATABASE_WAS_EXPLICIT:-0}" != "1" ]; then
                  export PGDATABASE="''${DBPEDIA_COLBERT_PGDATABASE:-pgturbohybrid_dbpedia_colbert}"
                fi
                ${colbertLlamaEnv}
                ${pgInitColbertLlama}/bin/th-pg-init-colbert-llama >/dev/null
                export PGHOST PGPORT PGUSER PGDATABASE
                exec uv run "$TH_ROOT/benchmarks/dbpedia_colbert_generate_export.py" "$@"
              '';

              colbertBuildStub = mkScript "th-colbert-build-stub" ''
                make -C "$TH_ROOT/extensions/pg_colbert_llama" \
                  PG_CONFIG="$PG_CONFIG" \
                  VECTOR_INCLUDE="$VECTOR_INCLUDE" \
                  PG_COLBERT_LLAMA_ENGINE=stub \
                  "$@"
              '';

              colbertTestStub = mkScriptWithInputs [
                tapPerl
                pkgs.python3
              ] "th-colbert-test-stub" ''
                ${pgInit}/bin/th-pg-init >/dev/null
                make -C "$TH_ROOT/extensions/pg_colbert_llama" \
                  PG_CONFIG="$PG_CONFIG" \
                  VECTOR_INCLUDE="$VECTOR_INCLUDE" \
                  PG_COLBERT_LLAMA_ENGINE=stub
                make -C "$TH_ROOT/extensions/pg_colbert_llama" \
                  PG_CONFIG="$PG_CONFIG" \
                  VECTOR_INCLUDE="$VECTOR_INCLUDE" \
                  PG_COLBERT_LLAMA_ENGINE=stub \
                  installcheck
                python3 -m unittest discover \
                  "$TH_ROOT/extensions/pg_colbert_llama/test" \
                  -p 'test_*.py'
                export PERL5LIB="${postgresql.src}/src/test/perl:${postgresql.src}/test/perl:''${PERL5LIB:-}"
                export PG_PROVE_FLAGS="-I ${postgresql.src}/src/test/perl ''${PG_PROVE_FLAGS:-}"
                exec make -C "$TH_ROOT/extensions/pg_colbert_llama" \
                  PG_CONFIG="$PG_CONFIG" \
                  bindir="${postgresWithExtensions}/bin" \
                  PG_PROVE_FLAGS="-I ${postgresql.src}/src/test/perl ''${PG_PROVE_FLAGS:-}" \
                  prove_stub_installcheck
              '';

              colbertBuildLlama = mkScript "th-colbert-build-llama" ''
                exec nix --extra-experimental-features 'nix-command flakes' \
                  build "$TH_ROOT#pg_colbert_llama-llama" \
                  --print-build-logs \
                  "$@"
              '';

              colbertLiveTest = mkScriptWithInputs [ tapPerl ] "th-colbert-live-test"
                ''
                  if [ -z "''${PG_COLBERT_LLAMA_TEST_MODEL:-}" ]; then
                    echo "PG_COLBERT_LLAMA_TEST_MODEL is not set; skipping live pg_colbert_llama TAP tests"
                    exit 0
                  fi

                  postgres_with_colbert="''${TH_COLBERT_LLAMA_POSTGRES:-}"
                  if [ -z "$postgres_with_colbert" ]; then
                    postgres_with_colbert="$(nix --extra-experimental-features 'nix-command flakes' \
                      build "$TH_ROOT#postgres-with-colbert-llama" \
                      --no-link \
                      --print-out-paths)"
                  fi
                  export PATH="$postgres_with_colbert/bin:$PATH"

                  export TH_ENV_NAME="''${TH_COLBERT_LLAMA_ENV_NAME:-${commonEnv.TH_ENV_NAME}-colbert-llama}"
                  export TH_STATE_DIR="$TH_ROOT/.nix-dev/$TH_ENV_NAME"
                  export PGDATA="$TH_STATE_DIR/pgdata"
                  export PGHOST="$TH_STATE_DIR/run"
                  export TH_LOG_DIR="$TH_STATE_DIR/log"
                  export TH_LOG_FILE="$TH_LOG_DIR/postgres.log"
                  mkdir -p "$TH_STATE_DIR" "$PGHOST" "$TH_LOG_DIR"

                  if [ ! -s "$PGDATA/PG_VERSION" ]; then
                    initdb -D "$PGDATA" --username="$PGUSER" --auth=trust --no-locale --encoding=UTF8
                    {
                      echo "unix_socket_directories = '$PGHOST'"
                      echo "port = $TH_PGPORT"
                      echo "listen_addresses = 'localhost'"
                      echo "log_min_messages = warning"
                      echo "client_min_messages = warning"
                    } >> "$PGDATA/postgresql.conf"
                  fi

                  if ! pg_ctl -D "$PGDATA" status >/dev/null 2>&1; then
                    pg_ctl -D "$PGDATA" -l "$TH_LOG_FILE" -o "-k '$PGHOST' -p '$TH_PGPORT'" start
                  fi

                  export PERL5LIB="${postgresql.src}/src/test/perl:${postgresql.src}/test/perl:''${PERL5LIB:-}"
                  export PG_PROVE_FLAGS="-I ${postgresql.src}/src/test/perl ''${PG_PROVE_FLAGS:-}"
                  exec make -C "$TH_ROOT/extensions/pg_colbert_llama" \
                    PG_CONFIG="$PG_CONFIG" \
                    bindir="$postgres_with_colbert/bin" \
                    PG_PROVE_FLAGS="-I ${postgresql.src}/src/test/perl ''${PG_PROVE_FLAGS:-}" \
                    prove_live_installcheck
                '';

              scripts = [
                pgInit
                pgStart
                pgStop
                pgReset
                pgInitColbertLlama
                psqlScript
                smoke
                installcheck
                proveInstallcheck
                test
                benchRetrievalQuality
                benchProfileGrid
                benchTuneProfile
                benchResidualRerank
                benchBm25PhraseRerank
                benchDenseCandidateMiss
                benchNativeCache
                benchFiqaQuick
                benchConcurrentDense
                benchDbpediaColbert
                dbpediaColbertHfDataset
                dbpediaColbertGenerateExport
                colbertBuildStub
                colbertTestStub
                colbertBuildLlama
                colbertLiveTest
              ];

              colbertStubInstallcheck =
                pkgs.runCommand "pg_colbert_llama-stub-installcheck"
                  {
                    nativeBuildInputs = [
                      colbertTestStub
                      pgStop
                      pkgs.clang
                      pkgs.coreutils
                    ];
                  }
                  ''
                    cp -R --no-preserve=mode,ownership ${cleanSource} source
                    chmod -R u+w source

                    export TH_ROOT="$PWD/source"
                    export TH_ENV_NAME="flake-colbert-stub-${suffix}"
                    export TH_STATE_DIR="$TMPDIR/$TH_ENV_NAME"
                    export TH_PGPORT=55433
                    export PGPORT="$TH_PGPORT"
                    export PGDATABASE=pgturbohybrid_colbert_check
                    export TH_POSTGRES_SOCKET_ONLY=1

                    trap 'th-pg-stop >/dev/null 2>&1 || true' EXIT
                    th-colbert-test-stub
                    touch "$out"
                  '';
            in
            {
              inherit
                pgvector
                pgturbohybrid
                pgColbertLlamaStub
                pgColbertLlamaLlama
                colbertStubInstallcheck
                postgresWithExtensions
                postgresWithLlamaExtensions
                scripts
                commonEnv
                ;
            };

          stable = mkDevSet {
            suffix = "pgvector-v0.8.2";
            pgvector = pgvectorStable;
          };

          master = mkDevSet {
            suffix = "pgvector-master";
            pgvector = pgvectorMaster;
          };

          mkShell =
            {
              devSet,
              includeBenchDeps ? false,
            }:
            pkgs.mkShell {
              packages =
                [
                  devSet.postgresWithExtensions
                  postgresql.pg_config
                  pkgs.clang-tools
                  pkgs.git
                  pkgs.gnumake
                  pkgs.perl
                  pkgs.perlPackages.IPCRun
                  pkgs.pkg-config
                  pkgs.python3
                  pkgs.uv
                ]
                ++ lib.optionals includeBenchDeps [
                  pkgs.python3Packages.numpy
                  pkgs.python3Packages.pyarrow
                  pkgs.python3Packages.psycopg
                ]
                ++ devSet.scripts;

              inherit (devSet.commonEnv)
                TH_ENV_NAME
                TH_PGPORT
                PGPORT
                PGDATABASE
                PGUSER
                PG_CONFIG
                VECTOR_INCLUDE
                ;

              shellHook = ''
                export TH_ROOT="$PWD"
                export TH_STATE_DIR="$TH_ROOT/.nix-dev/$TH_ENV_NAME"
                export PGHOST="$TH_STATE_DIR/run"
                export PGDATA="$TH_STATE_DIR/pgdata"
                export TH_LOG_DIR="$TH_STATE_DIR/log"
                export TH_LOG_FILE="$TH_LOG_DIR/postgres.log"
                export PATH="${postgresql.pg_config}/bin:${devSet.postgresWithExtensions}/bin:$PATH"

                cat <<EOF
                pgturbohybrid Nix shell
                  PostgreSQL: ${postgresql.version}
                  pgvector:   ${devSet.pgvector.version}
                  PG_CONFIG:  $PG_CONFIG
                  PGDATA:     $PGDATA

                Commands:
                  th-pg-init             initialize/start local PostgreSQL and install extensions
                  th-pg-init-colbert-llama initialize/start llama-backed ColBERT PostgreSQL
                  th-pg-start / th-pg-stop
                  th-pg-reset            recreate the local cluster
                  th-psql                connect to $PGDATABASE
                  th-smoke               run a minimal vector + pgturbohybrid query
                  th-test                run smoke + SQL regression tests
                  th-installcheck        run SQL regression tests
                  th-prove-installcheck  run TAP tests
                  th-colbert-build-stub  build pg_colbert_llama with the stub engine
                  th-colbert-test-stub   run pg_colbert_llama stub regression tests
                  th-colbert-build-llama build pg_colbert_llama against llama.cpp
                  th-colbert-live-test   run gated live ColBERT TAP tests

                Deterministic benchmark helpers:
                  th-bench-retrieval-quality
                  th-bench-profile-grid
                  th-bench-tune-profile
                  th-bench-residual-rerank
                  th-bench-bm25-phrase-rerank
                  th-bench-dense-candidate-miss
                  th-bench-native-cache-memory
                  th-bench-dbpedia-colbert
                  th-dbpedia-colbert-hf-dataset
                  th-dbpedia-colbert-generate-export
                EOF
              '';
            };

          pgturbohybridNoSimd = mkPgturbohybrid {
            pgvector = pgvectorStable;
            simdBuild = "none";
          };
        in
        {
          inherit stable master;

          packages = {
            default = stable.postgresWithExtensions;
            postgres-with-extensions = stable.postgresWithExtensions;
            postgres-with-colbert-llama = stable.postgresWithLlamaExtensions;
            pgturbohybrid = stable.pgturbohybrid;
            pgturbohybrid-nosimd = pgturbohybridNoSimd;
            pg_colbert_llama = stable.pgColbertLlamaStub;
            pg_colbert_llama-llama = stable.pgColbertLlamaLlama;
            pgvector = stable.pgvector;
            postgres-with-extensions-pgvector-master = master.postgresWithExtensions;
            pgturbohybrid-pgvector-master = master.pgturbohybrid;
            pg_colbert_llama-pgvector-master = master.pgColbertLlamaStub;
            pgvector-master = master.pgvector;
          };

          devShells = {
            default = mkShell { devSet = stable; };
            bench = mkShell {
              devSet = stable;
              includeBenchDeps = true;
            };
            pgvector-master = mkShell { devSet = master; };
          };

          checks = {
            pgturbohybrid = stable.pgturbohybrid;
            pg_colbert_llama = stable.pgColbertLlamaStub;
            pg_colbert_llama-stub-installcheck = stable.colbertStubInstallcheck;
            postgres-with-extensions = stable.postgresWithExtensions;
            pgturbohybrid-pgvector-master = master.pgturbohybrid;
            pg_colbert_llama-pgvector-master = master.pgColbertLlamaStub;
            pgturbohybrid-nosimd = pgturbohybridNoSimd;
          };

          apps =
            let
              mkApp = description: drv: {
                type = "app";
                program = "${drv}/bin/${drv.name}";
                meta.description = description;
              };
              scriptByName = devSet: name: lib.findFirst (script: script.name == name) null devSet.scripts;
              stableScript = name: scriptByName stable name;
            in
            {
              default = mkApp "Run the pgturbohybrid smoke test" (stableScript "th-smoke");
              smoke = mkApp "Run the pgturbohybrid smoke test" (stableScript "th-smoke");
              installcheck = mkApp "Run SQL regression tests" (stableScript "th-installcheck");
              test = mkApp "Run smoke and SQL regression tests" (stableScript "th-test");
              bench-retrieval-quality =
                mkApp "Run the deterministic retrieval quality grid" (stableScript "th-bench-retrieval-quality");
              bench-profile-grid =
                mkApp "Run the deterministic profile recall/latency grid" (stableScript "th-bench-profile-grid");
              bench-tune-profile =
                mkApp "Run the retrieval profile autotuning SQL harness" (stableScript "th-bench-tune-profile");
              bench-concurrent-dense =
                mkApp "Run the concurrent dense Python benchmark via uv" (stableScript "th-bench-concurrent-dense");
              bench-dbpedia-colbert =
                mkApp "Run the DBpedia ColBERT multivector benchmark via uv" (stableScript "th-bench-dbpedia-colbert");
              dbpedia-colbert-hf-dataset =
                mkApp "Export/import the DBpedia ColBERT multivector Hugging Face dataset" (stableScript "th-dbpedia-colbert-hf-dataset");
              dbpedia-colbert-generate-export =
                mkApp "Generate DBpedia ColBERT multivectors and export the Hugging Face dataset" (stableScript "th-dbpedia-colbert-generate-export");
              colbert-build-stub =
                mkApp "Build pg_colbert_llama with the stub engine" (stableScript "th-colbert-build-stub");
              colbert-test-stub =
                mkApp "Run pg_colbert_llama stub regression tests" (stableScript "th-colbert-test-stub");
              colbert-build-llama =
                mkApp "Build pg_colbert_llama against llama.cpp" (stableScript "th-colbert-build-llama");
              colbert-live-test =
                mkApp "Run gated live pg_colbert_llama TAP tests" (stableScript "th-colbert-live-test");
            };
        };
    in
    {
      packages = forAllSystems (system: (mkForSystem system).packages);
      devShells = forAllSystems (system: (mkForSystem system).devShells);
      checks = forAllSystems (system: (mkForSystem system).checks);
      apps = forAllSystems (system: (mkForSystem system).apps);
    };
}
