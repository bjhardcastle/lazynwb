from __future__ import annotations

import argparse
import contextlib
import dataclasses
import statistics
import time
from collections.abc import Iterator
from typing import Any

import polars as pl

import lazynwb
import lazynwb.dandi
import lazynwb.utils

try:
    import lindi
except ImportError:
    lindi = None


DEFAULT_DANDISET_ID = "000363"
DEFAULT_VERSION = "0.231012.2129"
DEFAULT_ASSET_ID = "21c622b7-6d8e-459b-98e8-b968a97a1585"
DEFAULT_TABLE_PATH = "/intervals/trials"
DEFAULT_SELECT_COLUMNS = ("start_time", "stop_time")
DEFAULT_MODES = ("lazynwb_reference_zarr", "lindi_h5py", "raw_hdf5")
DEFAULT_ROWS = 10
DEFAULT_REPEATS = 3


@dataclasses.dataclass
class RunResult:
    mode: str
    source: str
    elapsed_seconds: float
    row_count: int
    total_duration_seconds: float
    row_indices: list[int]
    preview_rows: list[dict[str, Any]]


@contextlib.contextmanager
def dandi_mode(*, prefer_neurosift: bool) -> Iterator[None]:
    original = lazynwb.dandi.dandi_config.prefer_neurosift
    lazynwb.dandi.dandi_config.prefer_neurosift = prefer_neurosift
    try:
        yield
    finally:
        lazynwb.dandi.dandi_config.prefer_neurosift = original


def parse_select_columns(value: str) -> tuple[str, ...]:
    columns = tuple(column.strip() for column in value.split(",") if column.strip())
    if not columns:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated column")
    return columns


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


def _require_lindi() -> None:
    if lindi is None:
        raise SystemExit(
            "This benchmark requires `lindi` for mode `lindi_h5py`. "
            "Install the repo benchmarking dependencies with `uv sync`."
        )


def resolve_source_for_mode(
    *,
    mode: str,
    dandiset_id: str,
    asset_id: str,
    version: str,
    use_local_cache: bool,
) -> str:
    prefer_neurosift = mode != "raw_hdf5"
    with dandi_mode(prefer_neurosift=prefer_neurosift):
        source = lazynwb.dandi.resolve_dandi_asset_source(
            dandiset_id=dandiset_id,
            asset_id=asset_id,
            version=version,
            use_local_cache=use_local_cache if mode == "raw_hdf5" else False,
        )
    if prefer_neurosift and not str(source).endswith(".lindi.json"):
        raise ValueError(
            f"Expected a Neurosift LINDI JSON URL for {mode}, got {source!r}"
        )
    if not prefer_neurosift and str(source).endswith(".lindi.json"):
        raise ValueError(f"Expected a raw HDF5 URL for {mode}, got {source!r}")
    return source


def pick_row_indices(
    *,
    dandiset_id: str,
    asset_id: str,
    version: str,
    table_path: str,
    row_limit: int,
    start_row: int | None,
) -> tuple[list[int], int]:
    source = resolve_source_for_mode(
        mode="raw_hdf5",
        dandiset_id=dandiset_id,
        asset_id=asset_id,
        version=version,
        use_local_cache=False,
    )
    df = lazynwb.get_df(
        source,
        table_path,
        as_polars=True,
        include_column_names=[lazynwb.TABLE_INDEX_COLUMN_NAME],
        disable_progress=True,
    )
    row_indices = df[lazynwb.TABLE_INDEX_COLUMN_NAME].to_list()
    if not row_indices:
        raise ValueError(f"Could not find any rows in {table_path}")

    window_size = min(row_limit, len(row_indices))
    max_start = max(len(row_indices) - window_size, 0)
    start_pos = (
        max(0, min(start_row, max_start)) if start_row is not None else max_start // 2
    )
    return row_indices[start_pos : start_pos + window_size], len(row_indices)


def _normalize_values(values: object) -> list[Any]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [value.decode() if isinstance(value, bytes) else value for value in values]


def _read_lindi_column_values(dataset: object, row_indices: list[int]) -> list[Any]:
    start = row_indices[0]
    stop = row_indices[-1] + 1
    contiguous = row_indices == list(range(start, stop))
    dtype = getattr(dataset, "dtype", None)
    dtype_kind = getattr(dtype, "kind", None)

    if dtype_kind in ("S", "O"):
        try:
            string_dataset = dataset.asstr()
            values = (
                string_dataset[start:stop]
                if contiguous
                else string_dataset[:][row_indices]
            )
        except (AttributeError, TypeError):
            values = dataset[:].astype(str)
            if contiguous:
                values = values[start:stop]
            else:
                values = values[row_indices]
        return _normalize_values(values)

    values = dataset[start:stop] if contiguous else dataset[:][row_indices]
    return _normalize_values(values)


def collect_with_lindi_h5py(
    *,
    source: str,
    table_path: str,
    row_indices: list[int],
    select_columns: tuple[str, ...],
) -> pl.DataFrame:
    _require_lindi()
    normalized_table_path = lazynwb.utils.normalize_internal_file_path(table_path)
    file = lindi.LindiH5pyFile.from_lindi_file(source)
    try:
        table = file[normalized_table_path]
        data: dict[str, list[Any]] = {
            lazynwb.TABLE_INDEX_COLUMN_NAME: row_indices,
        }
        for column_name in select_columns:
            data[column_name] = _read_lindi_column_values(
                table[column_name], row_indices
            )
        return pl.DataFrame(data)
    finally:
        file.close()


def run_query(
    *,
    mode: str,
    dandiset_id: str,
    asset_id: str,
    version: str,
    table_path: str,
    row_indices: list[int],
    select_columns: tuple[str, ...],
    use_local_cache: bool,
) -> RunResult:
    lazynwb.clear_cache()
    t0 = time.perf_counter()
    source = resolve_source_for_mode(
        mode=mode,
        dandiset_id=dandiset_id,
        asset_id=asset_id,
        version=version,
        use_local_cache=use_local_cache,
    )

    if mode == "lindi_h5py":
        df = collect_with_lindi_h5py(
            source=source,
            table_path=table_path,
            row_indices=row_indices,
            select_columns=select_columns,
        )
    else:
        lf = lazynwb.scan_nwb(
            source,
            table_path,
            infer_schema_length=1,
            disable_progress=True,
        )
        selected_columns = (lazynwb.TABLE_INDEX_COLUMN_NAME, *select_columns)
        df = (
            lf.filter(pl.col(lazynwb.TABLE_INDEX_COLUMN_NAME).is_in(row_indices))
            .select(*selected_columns)
            .sort(lazynwb.TABLE_INDEX_COLUMN_NAME)
            .collect()
        )

    elapsed_seconds = time.perf_counter() - t0
    total_duration_seconds = float(
        (df["stop_time"] - df["start_time"]).cast(pl.Float64).sum() or 0.0
    )
    return RunResult(
        mode=mode,
        source=source,
        elapsed_seconds=elapsed_seconds,
        row_count=df.height,
        total_duration_seconds=total_duration_seconds,
        row_indices=df[lazynwb.TABLE_INDEX_COLUMN_NAME].to_list(),
        preview_rows=df.to_dicts(),
    )


def summarize(results: list[RunResult]) -> None:
    print("\nSummary")
    print("-------")
    by_mode: dict[str, list[RunResult]] = {}
    for result in results:
        by_mode.setdefault(result.mode, []).append(result)

    baseline_row_indices: list[int] | None = None
    for mode, mode_results in by_mode.items():
        timings = [r.elapsed_seconds for r in mode_results]
        row_counts = {r.row_count for r in mode_results}
        duration_totals = {round(r.total_duration_seconds, 6) for r in mode_results}
        print(
            f"{mode:>22}: median={statistics.median(timings):.3f}s "
            f"min={min(timings):.3f}s max={max(timings):.3f}s "
            f"rows={sorted(row_counts)} total_duration_s={sorted(duration_totals)}"
        )
        print(f"{'':>22}  source={mode_results[0].source}")
        print(f"{'':>22}  row_indices={mode_results[0].row_indices}")
        print(f"{'':>22}  preview_rows={mode_results[0].preview_rows}")
        if baseline_row_indices is None:
            baseline_row_indices = mode_results[0].row_indices
        elif mode_results[0].row_indices != baseline_row_indices:
            print(f"{'':>22}  WARNING: row indices differ from the first mode")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark filtering, selecting, and collecting a small number of "
            "/intervals/trials rows from a DANDI asset via raw HDF5, the current "
            "lazynwb reference-Zarr path, and the lindi package."
        )
    )
    parser.add_argument("--dandiset-id", default=DEFAULT_DANDISET_ID)
    parser.add_argument("--asset-id", default=DEFAULT_ASSET_ID)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--table-path", default=DEFAULT_TABLE_PATH)
    parser.add_argument(
        "--select-columns",
        type=parse_select_columns,
        default=DEFAULT_SELECT_COLUMNS,
        help=(
            "Comma-separated scalar columns to fetch after the row filter. "
            f"Default: {','.join(DEFAULT_SELECT_COLUMNS)}"
        ),
    )
    parser.add_argument(
        "--modes",
        type=parse_modes,
        default=DEFAULT_MODES,
        help=(
            "Comma-separated modes. Choices: "
            f"{', '.join(DEFAULT_MODES)}. Default: {', '.join(DEFAULT_MODES)}"
        ),
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=None,
        help=(
            "Optional zero-based row offset within the trials table. If omitted, "
            "the benchmark uses a centered window."
        ),
    )
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--use-local-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If set, allow only the raw HDF5 mode to use the persistent local "
            "cache. Default is false for a cleaner network benchmark."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.rows <= 0:
        raise ValueError("--rows must be a positive integer")
    if args.repeats <= 0:
        raise ValueError("--repeats must be a positive integer")

    row_indices, total_rows = pick_row_indices(
        dandiset_id=args.dandiset_id,
        asset_id=args.asset_id,
        version=args.version,
        table_path=args.table_path,
        row_limit=args.rows,
        start_row=args.start_row,
    )

    print("Benchmark configuration")
    print("-----------------------")
    print(f"dandiset_id      : {args.dandiset_id}")
    print(f"asset_id         : {args.asset_id}")
    print(f"version          : {args.version}")
    print(f"table_path       : {args.table_path}")
    print(f"select_columns   : {args.select_columns}")
    print(f"modes            : {args.modes}")
    print(f"row_limit        : {args.rows}")
    print(f"start_row        : {args.start_row}")
    print(f"selected_rows    : {row_indices}")
    print(f"total_rows       : {total_rows}")
    print(f"repeats          : {args.repeats}")
    print(f"use_local_cache  : {args.use_local_cache}")

    results: list[RunResult] = []
    for mode in args.modes:
        for i in range(args.repeats):
            result = run_query(
                mode=mode,
                dandiset_id=args.dandiset_id,
                asset_id=args.asset_id,
                version=args.version,
                table_path=args.table_path,
                row_indices=row_indices,
                select_columns=args.select_columns,
                use_local_cache=args.use_local_cache,
            )
            results.append(result)
            print(
                f"{result.mode:>22} run {i + 1}/{args.repeats}: "
                f"{result.elapsed_seconds:.3f}s rows={result.row_count} "
                f"total_duration_s={result.total_duration_seconds:.6f}"
            )

    summarize(results)


if __name__ == "__main__":
    main()
