"""Benchmark cold and warm remote Zarr schema discovery.

The benchmark is intentionally manual because it uses public object-store inputs.
It prints ``cold_zarr_schema_seconds=<value>`` for autoresearch and can also
write structured JSON for comparing experiments.
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

import lazynwb._cache.sqlite as cache_sqlite
import lazynwb._zarr.reader as zarr_reader
import lazynwb.file_io as file_io
import lazynwb.utils as utils

_DEFAULT_SOURCES_FILE = pathlib.Path(__file__).with_name("zarr_paths.json")
_DEFAULT_TABLE_PATHS = ("/intervals/trials", "/units")


@dataclasses.dataclass(frozen=True, slots=True)
class _ZarrSchemaWorkItem:
    source_url: str
    table_path: str


@dataclasses.dataclass(frozen=True, slots=True)
class _ZarrSchemaMetric:
    phase: str
    source_url: str
    table_path: str
    elapsed_seconds: float
    metadata_read_count: int
    fetched_bytes: int
    used_consolidated_metadata: bool
    cache_status: str
    column_count: int
    table_length: int
    validator_kind: str

    def to_json_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class _ZarrPhaseSummary:
    phase: str
    elapsed_seconds: float
    metadata_read_count: int
    fetched_bytes: int
    item_count: int

    def to_json_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    file_io.config.anon = args.anon
    if args.cache_dir is None:
        with tempfile.TemporaryDirectory(prefix="lazynwb-zarr-schema-bench-") as cache:
            _run_benchmark(args, pathlib.Path(cache))
    else:
        _run_benchmark(args, args.cache_dir)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark lazynwb remote Zarr schema discovery latency."
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        help="Remote Zarr source URL. Repeat to override the sources file.",
    )
    parser.add_argument(
        "--sources-file",
        type=pathlib.Path,
        default=_env_path("LAZYNWB_ZARR_SCHEMA_BENCH_SOURCES_FILE")
        or _DEFAULT_SOURCES_FILE,
        help="JSON file containing a list of remote Zarr source URLs.",
    )
    parser.add_argument(
        "--table",
        action="append",
        dest="tables",
        help="Exact table path to read. Repeat to override default tables.",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=_env_int("LAZYNWB_ZARR_SCHEMA_BENCH_MAX_SOURCES", 0),
        help="Limit source count after loading. 0 means all sources.",
    )
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=_env_path("LAZYNWB_ZARR_SCHEMA_BENCH_CACHE_DIR"),
        help="Directory for the isolated SQLite cache.",
    )
    parser.add_argument(
        "--json-output",
        type=pathlib.Path,
        default=_env_path("LAZYNWB_ZARR_SCHEMA_BENCH_JSON"),
        help="Write metrics and summaries as JSON.",
    )
    parser.add_argument(
        "--anon",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("LAZYNWB_ZARR_SCHEMA_BENCH_ANON", True),
        help="Use anonymous public object-store access. Defaults to true.",
    )
    return parser.parse_args(argv)


def _run_benchmark(args: argparse.Namespace, cache_dir: pathlib.Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "catalog.sqlite"
    if cache_path.exists():
        cache_path.unlink()
    os.environ["LAZYNWB_CATALOG_CACHE_PATH"] = cache_path.as_posix()
    zarr_reader._clear_shared_metadata_catalog_cache()
    file_io.clear_cache()

    work_items = _build_workload(args)
    print(f"cache_path={cache_path}")
    print(f"anonymous={file_io.config.anon}")
    print(f"work_items={len(work_items)}")
    print()

    cold_summary, cold_metrics = _run_phase("cold", work_items)
    warm_summary, warm_metrics = _run_phase("warm", work_items)
    summaries = (cold_summary, warm_summary)
    metrics = (*cold_metrics, *warm_metrics)

    _print_metrics(metrics)
    _print_summaries(summaries)
    _write_json(args.json_output, summaries, metrics)
    print(f"cold_zarr_schema_seconds={cold_summary.elapsed_seconds:.6f}")


def _build_workload(args: argparse.Namespace) -> tuple[_ZarrSchemaWorkItem, ...]:
    sources = tuple(args.sources or _read_sources_file(args.sources_file))
    if args.max_sources > 0:
        sources = sources[: args.max_sources]
    if not sources:
        raise ValueError("Zarr schema benchmark requires at least one source")
    tables = tuple(args.tables or _DEFAULT_TABLE_PATHS)
    return tuple(
        _ZarrSchemaWorkItem(source_url=source, table_path=table)
        for source in sources
        for table in tables
    )


def _run_phase(
    phase: str,
    work_items: Sequence[_ZarrSchemaWorkItem],
) -> tuple[_ZarrPhaseSummary, tuple[_ZarrSchemaMetric, ...]]:
    started = time.perf_counter()
    metrics = tuple(asyncio.run(_read_phase_metrics(phase, work_items)))
    summary = _ZarrPhaseSummary(
        phase=phase,
        elapsed_seconds=time.perf_counter() - started,
        metadata_read_count=sum(metric.metadata_read_count for metric in metrics),
        fetched_bytes=sum(metric.fetched_bytes for metric in metrics),
        item_count=len(metrics),
    )
    return summary, metrics


async def _read_phase_metrics(
    phase: str,
    work_items: Sequence[_ZarrSchemaWorkItem],
) -> tuple[_ZarrSchemaMetric, ...]:
    metrics: list[_ZarrSchemaMetric] = []
    for source_url, source_items in _group_by_source(work_items):
        reader = zarr_reader._ZarrBackendReader(
            source_url,
            cache=cache_sqlite._SQLiteSnapshotCache(cache_sqlite._default_cache_path()),
        )
        try:
            for item in source_items:
                metrics.append(await _read_one_metric(phase, reader, item))
        finally:
            await reader.close()
    return tuple(metrics)


async def _read_one_metric(
    phase: str,
    reader: zarr_reader._ZarrBackendReader,
    item: _ZarrSchemaWorkItem,
) -> _ZarrSchemaMetric:
    metadata_reads_before = reader.metadata_read_count
    fetched_bytes_before = int(getattr(reader, "metadata_bytes_fetched", 0))
    started = time.perf_counter()
    normalized_path = utils.normalize_internal_file_path(item.table_path)
    snapshot = await reader.read_table_schema_snapshot(normalized_path)
    elapsed_seconds = time.perf_counter() - started
    metadata_read_count = reader.metadata_read_count - metadata_reads_before
    fetched_bytes = int(getattr(reader, "metadata_bytes_fetched", 0)) - fetched_bytes_before
    return _ZarrSchemaMetric(
        phase=phase,
        source_url=item.source_url,
        table_path=item.table_path,
        elapsed_seconds=elapsed_seconds,
        metadata_read_count=metadata_read_count,
        fetched_bytes=fetched_bytes,
        used_consolidated_metadata=reader.used_consolidated_metadata,
        cache_status="hit" if metadata_read_count == 0 else "miss",
        column_count=len(snapshot.columns),
        table_length=snapshot.table_length,
        validator_kind=snapshot.source_identity.validator_kind,
    )


def _group_by_source(
    work_items: Sequence[_ZarrSchemaWorkItem],
) -> tuple[tuple[str, tuple[_ZarrSchemaWorkItem, ...]], ...]:
    grouped: dict[str, list[_ZarrSchemaWorkItem]] = {}
    for item in work_items:
        grouped.setdefault(item.source_url, []).append(item)
    return tuple((source_url, tuple(items)) for source_url, items in grouped.items())


def _read_sources_file(path: pathlib.Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise TypeError(f"{path} must contain a JSON list")
    return tuple(str(item) for item in payload)


def _print_metrics(metrics: Iterable[_ZarrSchemaMetric]) -> None:
    print(
        "phase  table              metadata       bytes  seconds columns rows "
        "cache consolidated validator                         source"
    )
    print("-" * 132)
    for metric in metrics:
        print(
            f"{metric.phase:5s} "
            f"{metric.table_path:18s} "
            f"{metric.metadata_read_count:8d} "
            f"{metric.fetched_bytes:11d} "
            f"{metric.elapsed_seconds:8.3f} "
            f"{metric.column_count:7d} "
            f"{metric.table_length:5d} "
            f"{metric.cache_status:5s} "
            f"{str(metric.used_consolidated_metadata):12s} "
            f"{metric.validator_kind:32s} "
            f"{_source_label(metric.source_url)}"
        )
    print()


def _print_summaries(summaries: Iterable[_ZarrPhaseSummary]) -> None:
    print("summary")
    for summary in summaries:
        print(
            f"{summary.phase:5s} total={summary.elapsed_seconds:.3f}s "
            f"metadata={summary.metadata_read_count} bytes={summary.fetched_bytes} "
            f"items={summary.item_count}"
        )


def _write_json(
    path: pathlib.Path | None,
    summaries: Sequence[_ZarrPhaseSummary],
    metrics: Sequence[_ZarrSchemaMetric],
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


def _source_label(source_url: str) -> str:
    return source_url.rsplit("/", 1)[-1]


def _env_path(name: str) -> pathlib.Path | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    return pathlib.Path(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be boolean-like, got {value!r}")


if __name__ == "__main__":
    main()
