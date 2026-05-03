"""Benchmark aggregate HDF5 schema latency for the trials + units workload.

The benchmark is intentionally manual: it uses public S3 inputs by default, so it
is not part of the default unit-test suite. It records cold and warm timings for
the performance-investigation workload:

- ``/intervals/trials`` from two dynamic-routing NWB versions.
- ``/units`` from a configurable set of NWB inputs.

Examples
--------
    uv run python benchmarks/schema_benchmark.py
    LAZYNWB_SCHEMA_BENCH_JSON=metrics.json uv run python benchmarks/schema_benchmark.py
    uv run python benchmarks/schema_benchmark.py --units-sources-file tests/paths.txt
    uv run python benchmarks/schema_benchmark.py --budget --max-cold-seconds 2.0

Environment variables
---------------------
    LAZYNWB_SCHEMA_BENCH_ANON
        Defaults to true. Set to false to use credentialed object-store access.
    LAZYNWB_SCHEMA_BENCH_UNITS_SOURCES
        Comma-separated or newline-separated source URLs for the units workload.
    LAZYNWB_SCHEMA_BENCH_UNITS_SOURCES_FILE
        Text file containing one units source URL per line.
    LAZYNWB_SCHEMA_BENCH_MAX_UNITS_FILES
        Limit the units input set after it has been resolved. Defaults to all.
    LAZYNWB_SCHEMA_BENCH_CACHE_DIR
        Directory for the isolated cold/warm cache. Defaults to a temporary dir.
    LAZYNWB_SCHEMA_BENCH_JSON
        Optional JSON metrics output path.
    LAZYNWB_SCHEMA_BENCH_MAX_COLD_SECONDS
    LAZYNWB_SCHEMA_BENCH_MAX_WARM_SECONDS
    LAZYNWB_SCHEMA_BENCH_MAX_COLD_GETS
    LAZYNWB_SCHEMA_BENCH_MAX_COLD_BYTES
        Optional budget limits. Supplying any limit enables budget checking.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import pathlib
import tempfile
import time
from collections.abc import Iterable, Sequence

import lazynwb._hdf5.reader as hdf5_reader
import lazynwb.file_io as file_io
import lazynwb.utils as utils

_DEFAULT_TRIALS_SOURCES = (
    "s3://aind-scratch-data/dynamic-routing/cache/nwb/v0.0.272/"
    "620263_2022-07-26.nwb",
    "s3://aind-scratch-data/dynamic-routing/cache/nwb/v0.0.255/"
    "620263_2022-07-26.nwb",
)
_DEFAULT_UNITS_SOURCES = (_DEFAULT_TRIALS_SOURCES[0],)
_TRIALS_TABLE_PATH = "/intervals/trials"
_UNITS_TABLE_PATH = "/units"


@dataclasses.dataclass(frozen=True, slots=True)
class _SchemaWorkItem:
    table_path: str
    source_url: str


@dataclasses.dataclass(frozen=True, slots=True)
class _SchemaMetric:
    phase: str
    table_path: str
    source_url: str
    elapsed_seconds: float
    request_count: int
    fetched_bytes: int
    cache_status: str
    column_count: int
    table_length: int
    validator_kind: str

    def to_json_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class _PhaseSummary:
    phase: str
    elapsed_seconds: float
    request_count: int
    fetched_bytes: int
    item_count: int

    def to_json_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    file_io.config.anon = args.anon

    if args.cache_dir is None:
        with tempfile.TemporaryDirectory(prefix="lazynwb-schema-bench-") as cache_dir:
            _run_benchmark(args, pathlib.Path(cache_dir))
    else:
        _run_benchmark(args, args.cache_dir)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark lazynwb aggregate HDF5 schema latency for "
            "dynamic-routing trials and units."
        )
    )
    parser.add_argument(
        "--trials-source",
        action="append",
        dest="trials_sources",
        help=(
            "Source URL for the trials workload. Repeat to override the default "
            "two-version dynamic-routing set."
        ),
    )
    parser.add_argument(
        "--units-source",
        action="append",
        dest="units_sources",
        help="Source URL for the units workload. Repeat for multiple inputs.",
    )
    parser.add_argument(
        "--units-sources-file",
        type=pathlib.Path,
        default=_env_path("LAZYNWB_SCHEMA_BENCH_UNITS_SOURCES_FILE"),
        help="Text file with one units source URL per line.",
    )
    parser.add_argument(
        "--max-units-files",
        type=int,
        default=_env_int("LAZYNWB_SCHEMA_BENCH_MAX_UNITS_FILES", 0),
        help="Limit units inputs after resolution. 0 means no limit.",
    )
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=_env_path("LAZYNWB_SCHEMA_BENCH_CACHE_DIR"),
        help="Directory for the isolated cache used by cold/warm phases.",
    )
    parser.add_argument(
        "--json-output",
        type=pathlib.Path,
        default=_env_path("LAZYNWB_SCHEMA_BENCH_JSON"),
        help="Write metrics and summaries as JSON.",
    )
    parser.add_argument(
        "--anon",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("LAZYNWB_SCHEMA_BENCH_ANON", True),
        help="Use anonymous public object-store access. Defaults to true.",
    )
    parser.add_argument(
        "--budget",
        action="store_true",
        help="Fail non-zero when configured metric budgets are exceeded.",
    )
    parser.add_argument(
        "--max-cold-seconds",
        type=float,
        default=_env_float("LAZYNWB_SCHEMA_BENCH_MAX_COLD_SECONDS"),
    )
    parser.add_argument(
        "--max-warm-seconds",
        type=float,
        default=_env_float("LAZYNWB_SCHEMA_BENCH_MAX_WARM_SECONDS"),
    )
    parser.add_argument(
        "--max-cold-gets",
        type=int,
        default=_env_optional_int("LAZYNWB_SCHEMA_BENCH_MAX_COLD_GETS"),
    )
    parser.add_argument(
        "--max-cold-bytes",
        type=int,
        default=_env_optional_int("LAZYNWB_SCHEMA_BENCH_MAX_COLD_BYTES"),
    )
    return parser.parse_args(argv)


def _run_benchmark(args: argparse.Namespace, cache_dir: pathlib.Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "catalog.sqlite"
    if cache_path.exists():
        cache_path.unlink()
    os.environ["LAZYNWB_CATALOG_CACHE_PATH"] = cache_path.as_posix()

    work_items = _build_workload(args)
    print(f"cache_path={cache_path}")
    print(f"anonymous={file_io.config.anon}")
    print("target_profile=~2.0s cold, ~0.44s warm on the reference environment")
    print(f"work_items={len(work_items)}")
    print()

    file_io.clear_cache()
    cold_summary, cold_metrics = _run_phase("cold", work_items)
    warm_summary, warm_metrics = _run_phase("warm", work_items)
    summaries = (cold_summary, warm_summary)
    metrics = (*cold_metrics, *warm_metrics)

    _print_metrics(metrics)
    _print_summaries(summaries)
    _write_json(args.json_output, summaries, metrics)
    _check_budgets(args, summaries)


def _build_workload(args: argparse.Namespace) -> tuple[_SchemaWorkItem, ...]:
    trials_sources = tuple(args.trials_sources or _DEFAULT_TRIALS_SOURCES)
    units_sources = _resolve_units_sources(args)
    return (
        *(
            _SchemaWorkItem(table_path=_TRIALS_TABLE_PATH, source_url=source_url)
            for source_url in trials_sources
        ),
        *(
            _SchemaWorkItem(table_path=_UNITS_TABLE_PATH, source_url=source_url)
            for source_url in units_sources
        ),
    )


def _resolve_units_sources(args: argparse.Namespace) -> tuple[str, ...]:
    if args.units_sources:
        sources = tuple(args.units_sources)
    elif os.environ.get("LAZYNWB_SCHEMA_BENCH_UNITS_SOURCES"):
        sources = _split_sources(os.environ["LAZYNWB_SCHEMA_BENCH_UNITS_SOURCES"])
    elif args.units_sources_file is not None:
        sources = _read_sources_file(args.units_sources_file)
    else:
        sources = _DEFAULT_UNITS_SOURCES
    if args.max_units_files > 0:
        sources = sources[: args.max_units_files]
    if not sources:
        raise ValueError("units workload must include at least one source")
    return sources


def _run_phase(
    phase: str,
    work_items: Sequence[_SchemaWorkItem],
) -> tuple[_PhaseSummary, tuple[_SchemaMetric, ...]]:
    t0 = time.perf_counter()
    metrics = tuple(asyncio.run(_read_schema_metric(phase, item)) for item in work_items)
    summary = _PhaseSummary(
        phase=phase,
        elapsed_seconds=time.perf_counter() - t0,
        request_count=sum(metric.request_count for metric in metrics),
        fetched_bytes=sum(metric.fetched_bytes for metric in metrics),
        item_count=len(metrics),
    )
    return summary, metrics


async def _read_schema_metric(
    phase: str,
    work_item: _SchemaWorkItem,
) -> _SchemaMetric:
    reader = hdf5_reader._default_hdf5_backend_reader(work_item.source_url)
    normalized_table_path = utils.normalize_internal_file_path(work_item.table_path)
    t0 = time.perf_counter()
    try:
        snapshot = await reader.read_table_schema_snapshot(normalized_table_path)
    finally:
        await reader.close()
    elapsed_seconds = time.perf_counter() - t0
    request_count = int(getattr(reader._range_reader, "request_count", 0))
    fetched_bytes = int(getattr(reader._range_reader, "bytes_fetched", 0))
    return _SchemaMetric(
        phase=phase,
        table_path=work_item.table_path,
        source_url=work_item.source_url,
        elapsed_seconds=elapsed_seconds,
        request_count=request_count,
        fetched_bytes=fetched_bytes,
        cache_status="hit" if request_count == 0 else "miss",
        column_count=len(snapshot.columns),
        table_length=snapshot.table_length,
        validator_kind=snapshot.source_identity.validator_kind,
    )


def _print_metrics(metrics: Iterable[_SchemaMetric]) -> None:
    print(
        "phase  table              requests      bytes  seconds columns rows "
        "cache validator                         source"
    )
    print("-" * 118)
    for metric in metrics:
        print(
            f"{metric.phase:5s} "
            f"{metric.table_path:18s} "
            f"{metric.request_count:8d} "
            f"{metric.fetched_bytes:10d} "
            f"{metric.elapsed_seconds:8.3f} "
            f"{metric.column_count:7d} "
            f"{metric.table_length:5d} "
            f"{metric.cache_status:5s} "
            f"{metric.validator_kind:32s} "
            f"{_source_label(metric.source_url)}"
        )
    print()


def _print_summaries(summaries: Iterable[_PhaseSummary]) -> None:
    print("summary")
    for summary in summaries:
        print(
            f"{summary.phase:5s} total={summary.elapsed_seconds:.3f}s "
            f"requests={summary.request_count} bytes={summary.fetched_bytes} "
            f"items={summary.item_count}"
        )


def _write_json(
    path: pathlib.Path | None,
    summaries: Sequence[_PhaseSummary],
    metrics: Sequence[_SchemaMetric],
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summaries": [summary.to_json_dict() for summary in summaries],
        "metrics": [metric.to_json_dict() for metric in metrics],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"json_output={path}")


def _check_budgets(
    args: argparse.Namespace,
    summaries: Sequence[_PhaseSummary],
) -> None:
    limits = {
        "max_cold_seconds": args.max_cold_seconds,
        "max_warm_seconds": args.max_warm_seconds,
        "max_cold_gets": args.max_cold_gets,
        "max_cold_bytes": args.max_cold_bytes,
    }
    budget_enabled = args.budget or any(value is not None for value in limits.values())
    if not budget_enabled:
        return
    by_phase = {summary.phase: summary for summary in summaries}
    failures: list[str] = []
    cold = by_phase["cold"]
    warm = by_phase["warm"]
    _append_budget_failure(
        failures,
        "cold seconds",
        cold.elapsed_seconds,
        args.max_cold_seconds,
    )
    _append_budget_failure(
        failures,
        "warm seconds",
        warm.elapsed_seconds,
        args.max_warm_seconds,
    )
    _append_budget_failure(failures, "cold GETs", cold.request_count, args.max_cold_gets)
    _append_budget_failure(
        failures,
        "cold bytes",
        cold.fetched_bytes,
        args.max_cold_bytes,
    )
    if failures:
        raise SystemExit("schema benchmark budget exceeded: " + "; ".join(failures))


def _append_budget_failure(
    failures: list[str],
    label: str,
    actual: float | int,
    limit: float | int | None,
) -> None:
    if limit is not None and actual > limit:
        failures.append(f"{label} {actual} > {limit}")


def _read_sources_file(path: pathlib.Path) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _split_sources(value: str) -> tuple[str, ...]:
    return tuple(
        source.strip()
        for chunk in value.splitlines()
        for source in chunk.split(",")
        if source.strip()
    )


def _source_label(source_url: str) -> str:
    path = source_url.rstrip("/").split("/")
    if len(path) >= 3:
        return "/".join(path[-3:])
    return source_url


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value)


def _env_optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    return int(value)


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    return float(value)


def _env_path(name: str) -> pathlib.Path | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    return pathlib.Path(value)


if __name__ == "__main__":
    main()
