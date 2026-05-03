"""Compare accessor-backed schema inference with the fast catalog schema path.

This benchmark is intentionally opt-in for remote sources. It exercises schema
metadata only: no table column arrays should be materialized by either measured
schema function.

Examples
--------
    LAZYNWB_SCHEMA_BENCH_ANON=1 uv run python benchmarks/schema_benchmark.py
    uv run python benchmarks/schema_benchmark.py --reference-accessor --table /units
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import tempfile
import time
from collections.abc import Callable, Sequence

import polars as pl

import lazynwb
import lazynwb.table_metadata
import lazynwb.tables

_DEFAULT_SOURCE = (
    "s3://aind-scratch-data/dynamic-routing/cache/nwb/v0.0.272/"
    "620263_2022-07-26.nwb"
)
_DEFAULT_TABLES = ("/intervals/trials", "/units")


@dataclasses.dataclass(frozen=True, slots=True)
class _SchemaRun:
    label: str
    table_path: str
    elapsed_seconds: float
    column_count: int
    schema: pl.Schema


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark lazynwb schema inference paths."
    )
    parser.add_argument(
        "--source",
        default=os.environ.get("LAZYNWB_SCHEMA_SOURCE", _DEFAULT_SOURCE),
    )
    parser.add_argument("--table", action="append", dest="tables")
    parser.add_argument(
        "--reference-accessor",
        action="store_true",
        help="also run the old accessor-backed metadata schema path",
    )
    args = parser.parse_args(argv)

    lazynwb.config.anon = _env_bool("LAZYNWB_SCHEMA_BENCH_ANON", True)
    os.environ.setdefault("AWS_REGION", "us-west-2")
    tables = tuple(args.tables or _DEFAULT_TABLES)

    with tempfile.TemporaryDirectory(prefix="lazynwb-schema-bench-") as cache_dir:
        os.environ["LAZYNWB_CATALOG_CACHE_PATH"] = str(
            pathlib.Path(cache_dir) / "catalog.sqlite"
        )
        print(f"source={args.source}")
        print(f"tables={','.join(tables)}")
        print(f"anonymous={lazynwb.config.anon}")
        for table_path in tables:
            cold = _time_schema(
                "fast-cold",
                table_path,
                lambda table_path=table_path: _fast_public_schema(
                    args.source,
                    table_path,
                ),
            )
            warm = _time_schema(
                "fast-warm",
                table_path,
                lambda table_path=table_path: _fast_public_schema(
                    args.source,
                    table_path,
                ),
            )
            _print_run(cold)
            _print_run(warm)
            if args.reference_accessor:
                reference = _time_schema(
                    "accessor-reference",
                    table_path,
                    lambda table_path=table_path: _accessor_reference_schema(
                        args.source,
                        table_path,
                    ),
                )
                _print_run(reference)
                if reference.schema != cold.schema:
                    raise AssertionError(
                        f"schema mismatch for {table_path}: "
                        f"fast={cold.schema} reference={reference.schema}"
                    )


def _fast_public_schema(source_url: str, table_path: str) -> pl.Schema:
    return lazynwb.tables.get_table_schema(
        source_url,
        table_path,
        exclude_internal_columns=True,
    )


def _accessor_reference_schema(source_url: str, table_path: str) -> pl.Schema:
    columns = lazynwb.table_metadata.get_table_column_metadata(source_url, table_path)
    return lazynwb.tables.get_table_schema_from_metadata(columns)


def _time_schema(
    label: str,
    table_path: str,
    func: Callable[[], pl.Schema],
) -> _SchemaRun:
    t0 = time.perf_counter()
    schema = func()
    return _SchemaRun(
        label=label,
        table_path=table_path,
        elapsed_seconds=time.perf_counter() - t0,
        column_count=len(schema),
        schema=schema,
    )


def _print_run(run: _SchemaRun) -> None:
    print(
        f"{run.label:18s} {run.table_path:18s} "
        f"{run.column_count:4d} columns {run.elapsed_seconds:8.3f}s"
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    main()
