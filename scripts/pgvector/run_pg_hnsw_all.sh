#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin=${PYTHON:-"$HOME/.venvs/regexann-pgvector-system/bin/python"}
if [[ -z "${PYTHON:-}" && ! -x "$python_bin" ]]; then
    python_bin=python3
fi

setup_local() (
    set -euo pipefail
    local_home="$HOME/.local/regexann-pg16-bench"
    pg_prefix="$local_home/install"
    pg_bin="$pg_prefix/bin"
    pg_data="$local_home/data"
    pg_socket="$local_home/socket"
    pg_log="$local_home/postgres.log"
    pg_jobs=${PG_BUILD_JOBS:-8}

    if [[ ! "$pg_jobs" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s\n' 'PG_BUILD_JOBS must be a positive integer.' >&2
        exit 2
    fi
    install -d -m 700 "$local_home" "$pg_socket"

    if [[ ! -x "$pg_bin/postgres" || ! -x "$pg_bin/pg_ctl" ||
          ! -x "$pg_bin/pg_config" || ! -x "$pg_bin/initdb" ||
          ! -x "$pg_bin/psql" || ! -x "$pg_bin/createdb" ]]; then
        for command_name in cc make tar; do
            command -v "$command_name" >/dev/null || {
                printf 'PostgreSQL source build requires %s.\n' "$command_name" >&2
                exit 1
            }
        done
        pg_work="$(mktemp -d "$local_home/postgres-build.XXXXXXXX")"
        trap 'rm -rf -- "$pg_work"' EXIT
        printf '%s\n' 'Installing PostgreSQL 16.16 into the private benchmark directory...'
        "$python_bin" - "$pg_work/postgresql-16.16.tar.gz" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys
from urllib.request import urlretrieve

archive = Path(sys.argv[1])
urlretrieve("https://ftp.postgresql.org/pub/source/v16.16/postgresql-16.16.tar.gz", archive)
expected = "3e930f3366f675ab26e873d16201201f8e9b6d9bc13686e4465380271145ea2d"
if sha256(archive.read_bytes()).hexdigest() != expected:
    raise SystemExit("PostgreSQL source archive checksum mismatch")
PY
        tar -xzf "$pg_work/postgresql-16.16.tar.gz" -C "$pg_work"
        (
            cd "$pg_work/postgresql-16.16"
            ./configure --prefix="$pg_prefix" --without-icu --without-readline --without-zlib
            make -j "$pg_jobs"
            make install
        )
    fi

    case "$("$pg_bin/postgres" --version)" in
        'postgres (PostgreSQL) 16.'*) ;;
        *)
            printf 'Expected PostgreSQL 16 in %s.\n' "$pg_bin" >&2
            exit 1
            ;;
    esac
    pg_libdir="$("$pg_bin/pg_config" --pkglibdir)"
    pg_sharedir="$("$pg_bin/pg_config" --sharedir)"
    if [[ ! -f "$pg_libdir/vector.so" || ! -f "$pg_sharedir/extension/vector.control" ]]; then
        command -v git >/dev/null || {
            printf '%s\n' 'pgvector source build requires git.' >&2
            exit 1
        }
        vector_work="$(mktemp -d "$local_home/vector-build.XXXXXXXX")"
        (
            trap 'rm -rf -- "$vector_work"' EXIT
            git clone --quiet --depth 1 --branch v0.8.2 \
                https://github.com/pgvector/pgvector.git "$vector_work/src"
            make -C "$vector_work/src" -j "$pg_jobs" PG_CONFIG="$pg_bin/pg_config"
            make -C "$vector_work/src" PG_CONFIG="$pg_bin/pg_config" install
        )
    fi

    if [[ ! -f "$pg_data/PG_VERSION" ]]; then
        "$pg_bin/initdb" -D "$pg_data" --encoding=UTF8 --locale=C.UTF-8 \
            --auth-local=peer --auth-host=reject
    fi
    if [[ "$(cat "$pg_data/PG_VERSION")" != 16 ]]; then
        printf 'Unexpected cluster version in %s.\n' "$pg_data" >&2
        exit 1
    fi
    if ! "$pg_bin/pg_ctl" -D "$pg_data" status >/dev/null 2>&1; then
        if ! "$pg_bin/pg_ctl" -D "$pg_data" -l "$pg_log" \
            -o "-p 5433 -h '' -k '$pg_socket'" -w start; then
            tail -n 50 "$pg_log" >&2
            exit 1
        fi
    fi

    db_exists="$("$pg_bin/psql" -XAt -v ON_ERROR_STOP=1 -h "$pg_socket" -p 5433 \
        -d postgres -c "SELECT 1 FROM pg_database WHERE datname = 'regexann_pg'")"
    if [[ "$db_exists" != 1 ]]; then
        "$pg_bin/createdb" -h "$pg_socket" -p 5433 regexann_pg
    fi
    "$pg_bin/psql" -X -v ON_ERROR_STOP=1 -h "$pg_socket" -p 5433 -d regexann_pg \
        -c "CREATE EXTENSION IF NOT EXISTS vector" \
        -c "SELECT '[0]'::vector AS pgvector_probe"
    printf 'Pgvector is ready at unix socket %s, port 5433.\n' "$pg_socket"
)

case "${1:-}" in
    --setup-local)
        if [[ $# -ne 1 ]]; then
            printf '%s\n' 'Usage: bash run_pg_hnsw_all.sh --setup-local' >&2
            exit 2
        fi
        setup_local
        ;;
    --memory-only)
        shift
        exec "$python_bin" "$script_dir/measure_memory.py" "$@"
        ;;
    -h|--help)
        printf '%s\n' \
            'Usage: bash run_pg_hnsw_all.sh [--pg-host HOST] [--pg-port PORT] [--benchmark-outdir DIR]' \
            '       bash run_pg_hnsw_all.sh --memory-only [--resume] [--out CSV] [--ef-search N]' \
            '       bash run_pg_hnsw_all.sh --setup-local'
        ;;
    *)
        exec "$python_bin" "$script_dir/measure_memory.py" --benchmark "$@"
        ;;
esac
