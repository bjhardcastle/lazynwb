from __future__ import annotations

import argparse
import contextlib
import dataclasses
import statistics
import time
from collections.abc import Callable, Iterator
from typing import Any

import polars as pl

import lazynwb
import lazynwb.dandi
import lazynwb.file_io
import lazynwb.tables
import lazynwb.utils

try:
    import lindi
except ImportError as exc:  # pragma: no cover - script guard
    raise SystemExit(
        "This benchmark requires `lindi`. Install it with `uv pip install lindi`."
    ) from exc


DEFAULT_DANDISET_ID = "000363"
DEFAULT_VERSION = "0.231012.2129"
DEFAULT_ASSET_ID = "21c622b7-6d8e-459b-98e8-b968a97a1585"
DEFAULT_TABLE_PATH = "/intervals/trials"
DEFAULT_MODES = (
    "lazynwb_reference_zarr",
    "lindi_h5py",
    "lindi_zarr_group",
    "raw_hdf5",
)
DEFAULT_REPEATS = 1


@dataclasses.dataclass
class OpenedTable:
    table: Any
    close: Callable[[], None]


@dataclasses.dataclass
class RunResult:
    mode: str
    source: str
    open_seconds: float
    collect_seconds: float
    total_seconds: float
    row_count: int
    columns: list[str]
    preview_rows: list[dict[str, Any]]
    normalized_rows: list[dict[str, Any]]


@contextlib.contextmanager
def dandi_mode(*, prefer_neurosift: bool) -> Iterator[None]:
    original = lazynwb.dandi.dandi_config.prefer_neurosift
    lazynwb.dandi.dandi_config.prefer_neurosift = prefer_neurosift
    try:
        yield
    finally:
        lazynwb.dandi.dandi_config.prefer_neurosift = original


def parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(part.strip() for part in value.split(",") if part.strip())
    allowed = set(DEFAULT_MODES)
    invalid = sorted(set(modes) - allowed)
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported mode(s): {invalid}. Expected one of {sorted(allowed)}"
        )
    if not modes:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated mode")
    return modes


def resolve_source(
    *,
    prefer_neurosift: bool,
    dandiset_id: str,
    asset_id: str,
    version: str,
) -> str:
    with dandi_mode(prefer_neurosift=prefer_neurosift):
        source = lazynwb.dandi.resolve_dandi_asset_source(
            dandiset_id=dandiset_id,
            asset_id=asset_id,
            version=version,
            use_local_cache=False,
        )
    if prefer_neurosift and not str(source).endswith(".lindi.json"):
        raise ValueError(f"Expected a Neurosift LINDI JSON URL, got {source!r}")
    if not prefer_neurosift and str(source).endswith(".lindi.json"):
        raise ValueError(f"Expected a raw HDF5 URL, got {source!r}")
    return source


def open_with_lazynwb_reference_zarr(
    source: str, table_path: str
) -> tuple[float, OpenedTable]:
    lazynwb.clear_cache()
    t0 = time.perf_counter()
    accessor = lazynwb.file_io._get_accessor(source)
    table = accessor[table_path]
    open_seconds = time.perf_counter() - t0
    return open_seconds, OpenedTable(table=table, close=lazynwb.clear_cache)


def open_with_lindi_h5py(source: str, table_path: str) -> tuple[float, OpenedTable]:
    normalized_table_path = lazynwb.utils.normalize_internal_file_path(table_path)
    t0 = time.perf_counter()
    file = lindi.LindiH5pyFile.from_lindi_file(source)
    table = file[normalized_table_path]
    open_seconds = time.perf_counter() - t0
    return open_seconds, OpenedTable(table=table, close=file.close)


def open_with_lindi_zarr_group(
    source: str, table_path: str
) -> tuple[float, OpenedTable]:
    normalized_table_path = lazynwb.utils.normalize_internal_file_path(table_path)
    t0 = time.perf_counter()
    file = lindi.LindiH5pyFile.from_lindi_file(source)
    table = file._zarr_group[normalized_table_path]
    open_seconds = time.perf_counter() - t0
    return open_seconds, OpenedTable(table=table, close=file.close)


def open_with_raw_hdf5(source: str, table_path: str) -> tuple[float, OpenedTable]:
    lazynwb.clear_cache()
    t0 = time.perf_counter()
    accessor = lazynwb.file_io._get_accessor(source)
    table = accessor[table_path]
    open_seconds = time.perf_counter() - t0
    return open_seconds, OpenedTable(table=table, close=lazynwb.clear_cache)


OPENERS: dict[str, Callable[[str, str], tuple[float, OpenedTable]]] = {
    "lazynwb_reference_zarr": open_with_lazynwb_reference_zarr,
    "lindi_h5py": open_with_lindi_h5py,
    "lindi_zarr_group": open_with_lindi_zarr_group,
    "raw_hdf5": open_with_raw_hdf5,
}


def _read_column_values(dataset: Any) -> list[Any]:
    if dataset.ndim == 0:
        values = [dataset[()]]
    else:
        if getattr(dataset, "dtype", None) is not None and dataset.dtype.kind in ("S", "O"):
            try:
                values = dataset.asstr()[:]
            except (AttributeError, TypeError):
                values = dataset[:].astype(str)
        else:
            values = dataset[:]
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [
        value.decode() if isinstance(value, bytes) else value for value in values
    ]


def materialize_scalar_table(table: Any) -> pl.DataFrame:
    table_keys = tuple(table.keys())
    indexed_columns = lazynwb.tables._get_indexed_column_names(table_keys)
    if indexed_columns:
        raise ValueError(
            "This benchmark only supports scalar tables; found indexed columns "
            f"{sorted(indexed_columns)}"
        )

    data: dict[str, list[Any]] = {}
    for column_name in table_keys:
        dataset = table.get(column_name)
        if lazynwb.file_io.is_group(dataset):
            continue
        data[column_name] = _read_column_values(dataset)
    return pl.DataFrame(data)


def run_benchmark(
    *,
    mode: str,
    source: str,
    table_path: str,
) -> RunResult:
    open_table = OPENERS[mode]
    t0 = time.perf_counter()
    open_seconds, opened = open_table(source, table_path)
    try:
        t_collect = time.perf_counter()
        df = materialize_scalar_table(opened.table)
        collect_seconds = time.perf_counter() - t_collect
    finally:
        opened.close()

    total_seconds = time.perf_counter() - t0
    return RunResult(
        mode=mode,
        source=source,
        open_seconds=open_seconds,
        collect_seconds=collect_seconds,
        total_seconds=total_seconds,
        row_count=df.height,
        columns=df.columns,
        preview_rows=df.head(3).to_dicts(),
        normalized_rows=df.to_dicts(),
    )


def summarize(results: list[RunResult]) -> None:
    print("\nSummary")
    print("-------")
    by_mode: dict[str, list[RunResult]] = {}
    for result in results:
        by_mode.setdefault(result.mode, []).append(result)

    baseline_rows: list[dict[str, Any]] | None = None
    baseline_columns: list[str] | None = None
    for mode, mode_results in by_mode.items():
        total_timings = [r.total_seconds for r in mode_results]
        open_timings = [r.open_seconds for r in mode_results]
        collect_timings = [r.collect_seconds for r in mode_results]
        row_counts = {r.row_count for r in mode_results}
        print(
            f"{mode:>22}: total median={statistics.median(total_timings):.3f}s "
            f"open median={statistics.median(open_timings):.3f}s "
            f"collect median={statistics.median(collect_timings):.3f}s "
            f"rows={sorted(row_counts)}"
        )
        print(f"{'':>22}  columns={mode_results[0].columns}")
        print(f"{'':>22}  preview_rows={mode_results[0].preview_rows}")
        if baseline_columns is None:
            baseline_columns = mode_results[0].columns
        elif mode_results[0].columns != baseline_columns:
            print(f"{'':>22}  WARNING: columns differ from the first mode")
        if baseline_rows is None:
            baseline_rows = mode_results[0].normalized_rows
        elif mode_results[0].normalized_rows != baseline_rows:
            print(f"{'':>22}  WARNING: row contents differ from the first mode")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark time to materialize the full /intervals/trials table from "
            "a Neurosift LINDI asset via different access layers."
        )
    )
    parser.add_argument("--dandiset-id", default=DEFAULT_DANDISET_ID)
    parser.add_argument("--asset-id", default=DEFAULT_ASSET_ID)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--table-path", default=DEFAULT_TABLE_PATH)
    parser.add_argument(
        "--modes",
        type=parse_modes,
        default=DEFAULT_MODES,
        help=(
            "Comma-separated modes to benchmark. Default: "
            f"{','.join(DEFAULT_MODES)}"
        ),
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mode_to_source = {
        mode: resolve_source(
            prefer_neurosift=(mode != "raw_hdf5"),
            dandiset_id=args.dandiset_id,
            asset_id=args.asset_id,
            version=args.version,
        )
        for mode in args.modes
    }

    print("Benchmark configuration")
    print("-----------------------")
    print(f"dandiset_id      : {args.dandiset_id}")
    print(f"asset_id         : {args.asset_id}")
    print(f"version          : {args.version}")
    print(f"table_path       : {args.table_path}")
    print(f"modes            : {args.modes}")
    print(f"repeats          : {args.repeats}")
    for mode, source in mode_to_source.items():
        print(f"source[{mode}]    : {source}")

    results: list[RunResult] = []
    for mode in args.modes:
        for i in range(args.repeats):
            result = run_benchmark(
                mode=mode,
                source=mode_to_source[mode],
                table_path=args.table_path,
            )
            results.append(result)
            print(
                f"{mode:>22} run {i + 1}/{args.repeats}: "
                f"total={result.total_seconds:.3f}s "
                f"open={result.open_seconds:.3f}s "
                f"collect={result.collect_seconds:.3f}s "
                f"rows={result.row_count}"
            )

    summarize(results)


if __name__ == "__main__":
    main()
