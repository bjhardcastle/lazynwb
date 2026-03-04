"""
Benchmarks for streaming performance: pynwb vs lazynwb for tables and timeseries.

Primary target: s3://aind-scratch-data/tree/dynamic-routing/cache/nwb/v0.0.273/664851_2023-11-16.nwb
Fallback (public): Steinmetz 2019 from DANDI:000017

Usage:
    python benchmarks/streaming_benchmark.py [NWB_PATH]
"""

from __future__ import annotations

import gc
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field

import lazynwb

logger = logging.getLogger(__name__)

NWB_S3_PATH = "s3://aind-scratch-data/tree/dynamic-routing/cache/nwb/v0.0.273/664851_2023-11-16.nwb"
# public fallback: Steinmetz 2019 (DANDI:000017) ~312 MB HDF5 with trials, units, and timeseries
DANDI_FALLBACK_URL = "https://api.dandiarchive.org/api/assets/92694e6e-84fd-4198-a7e3-64e764f8e086/download/"

TABLE_PATH_TRIALS = "/intervals/trials"
TABLE_PATH_UNITS = "/units"
TIMESERIES_SEARCH_SMALL = "lick_times"
TIMESERIES_SEARCH_LARGE = "wheel_position"


@dataclass
class BenchmarkResult:
    name: str
    times: list[float] = field(default_factory=list)
    rows: int | None = None
    cols: int | None = None
    error: str | None = None

    @property
    def median(self) -> float:
        return statistics.median(self.times) if self.times else float("nan")

    @property
    def best(self) -> float:
        return min(self.times) if self.times else float("nan")

    def __str__(self) -> str:
        if self.error:
            return f"  {self.name}: ERROR - {self.error}"
        shape = f" ({self.rows} rows x {self.cols} cols)" if self.rows is not None else ""
        times_str = ", ".join(f"{t:.3f}s" for t in self.times)
        if len(self.times) == 1:
            return f"  {self.name}: {self.times[0]:.3f}s{shape}"
        return (
            f"  {self.name}: median={self.median:.3f}s  best={self.best:.3f}s"
            f"  [{times_str}]{shape}"
        )


def timed(fn, *, warmup: int = 0, repeats: int = 1) -> BenchmarkResult:
    result = BenchmarkResult(name="")
    for i in range(warmup + repeats):
        lazynwb.clear_cache()
        gc.collect()
        t0 = time.perf_counter()
        try:
            out = fn()
        except Exception as e:
            result.error = repr(e)
            return result
        elapsed = time.perf_counter() - t0
        if i >= warmup:
            result.times.append(elapsed)
        if hasattr(out, "shape") and len(out.shape) >= 2:
            result.rows, result.cols = out.shape[:2]
        elif hasattr(out, "shape"):
            result.rows = out.shape[0]
        elif hasattr(out, "height"):
            result.rows, result.cols = out.height, out.width
    return result


# ---------------------------------------------------------------------------
# pynwb helpers
# ---------------------------------------------------------------------------

def _open_pynwb(nwb_path: str):
    """Open an NWB file with pynwb, handling both S3 and HTTPS URLs."""
    import h5py
    import pynwb

    if nwb_path.startswith("s3://"):
        import fsspec

        fs = fsspec.filesystem("s3", anon=False)
        f = fs.open(nwb_path, "rb")
        h5_file = h5py.File(f, "r")
    elif nwb_path.startswith("http"):
        import remfile

        f = remfile.File(nwb_path)
        h5_file = h5py.File(f, "r")
    else:
        h5_file = h5py.File(nwb_path, "r")

    io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
    nwbfile = io.read()
    return nwbfile, io, h5_file


def _navigate_to_table(nwbfile, table_path: str):
    """Navigate pynwb object hierarchy to reach a table."""
    parts = [p for p in table_path.strip("/").split("/") if p]
    obj = nwbfile
    for part in parts:
        if hasattr(obj, part):
            obj = getattr(obj, part)
        elif hasattr(obj, "get"):
            obj = obj[part]
        else:
            raise AttributeError(f"Cannot navigate to {part!r} from {obj}")
    return obj


def _find_pynwb_timeseries(nwbfile, ts_search: str):
    """Search pynwb object for a timeseries matching the search term."""
    for name in nwbfile.acquisition:
        if ts_search.lower() in name.lower():
            return nwbfile.acquisition[name]

    for module_name in nwbfile.processing:
        module = nwbfile.processing[module_name]
        for container_name in module.data_interfaces:
            container = module.data_interfaces[container_name]
            if ts_search.lower() in container_name.lower():
                return container
            if hasattr(container, "time_series"):
                for ts_name in container.time_series:
                    if ts_search.lower() in ts_name.lower():
                        return container.time_series[ts_name]

    raise KeyError(f"TimeSeries matching {ts_search!r} not found")


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def discover_contents(nwb_path: str) -> dict:
    print(f"Discovering contents of {nwb_path} ...")
    t0 = time.perf_counter()
    paths = lazynwb.get_internal_paths(nwb_path, include_arrays=True)
    elapsed = time.perf_counter() - t0
    print(f"  Found {len(paths)} internal paths in {elapsed:.2f}s")

    tables = []
    timeseries = []
    for p, obj in paths.items():
        attrs = dict(getattr(obj, "attrs", {}))
        if "colnames" in attrs:
            tables.append(p)
        elif p.endswith("/data") or p.endswith("/timestamps"):
            parent = p.rsplit("/", 1)[0]
            if parent not in timeseries:
                timeseries.append(parent)

    print(f"  Tables: {tables}")
    print(f"  TimeSeries: {timeseries}")
    return {"tables": tables, "timeseries": timeseries}


# ---------------------------------------------------------------------------
# table benchmarks
# ---------------------------------------------------------------------------

def bench_table_pynwb(nwb_path: str, table_path: str) -> BenchmarkResult:
    """pynwb: open file, read(), .to_dataframe() — always reads all columns."""
    def _read():
        nwbfile, io, h5_file = _open_pynwb(nwb_path)
        obj = _navigate_to_table(nwbfile, table_path)
        df = obj.to_dataframe()
        io.close()
        h5_file.close()
        return df

    r = timed(_read)
    r.name = f"pynwb .to_dataframe()"
    return r


def bench_table_get_df_all(nwb_path: str, table_path: str) -> BenchmarkResult:
    """lazynwb.get_df with all columns (including arrays) — equivalent to pynwb."""
    def _read():
        return lazynwb.get_df(
            nwb_path,
            table_path,
            exact_path=True,
            exclude_array_columns=False,
            disable_progress=True,
        )

    r = timed(_read)
    r.name = "lazynwb.get_df (all columns)"
    return r


def bench_table_get_df_scalar(nwb_path: str, table_path: str) -> BenchmarkResult:
    """lazynwb.get_df excluding array columns — skips spike_times, waveform_mean, etc."""
    def _read():
        return lazynwb.get_df(
            nwb_path,
            table_path,
            exact_path=True,
            exclude_array_columns=True,
            disable_progress=True,
        )

    r = timed(_read)
    r.name = "lazynwb.get_df (scalar columns only)"
    return r


def bench_table_scan_nwb(nwb_path: str, table_path: str) -> BenchmarkResult:
    """lazynwb.scan_nwb: lazy polars scan, collect all scalar columns."""
    def _read():
        return lazynwb.scan_nwb(
            nwb_path,
            table_path=table_path,
            exclude_array_columns=True,
            disable_progress=True,
        ).collect()

    r = timed(_read)
    r.name = "lazynwb.scan_nwb (scalar columns)"
    return r


def bench_table_scan_nwb_filter(nwb_path: str, table_path: str) -> BenchmarkResult:
    """lazynwb.scan_nwb: filter + select — only reads predicate columns, then
    fetches remaining columns for matching rows."""
    import polars as pl

    # pick a numeric column to filter on
    schema = lazynwb.tables.get_table_schema(
        file_paths=(nwb_path,),
        table_path=table_path,
        exclude_array_columns=True,
        exclude_internal_columns=True,
    )
    numeric_col = None
    for col, dtype in schema.items():
        if dtype.is_numeric() and col not in ("id",):
            numeric_col = col
            break
    if numeric_col is None:
        r = BenchmarkResult(name="lazynwb.scan_nwb (filter + select)")
        r.error = "No numeric column found for filtering"
        return r

    select_cols = [c for c in list(schema.keys())[:5] if c != numeric_col]

    def _read():
        return (
            lazynwb.scan_nwb(
                nwb_path,
                table_path=table_path,
                exclude_array_columns=True,
                disable_progress=True,
            )
            .filter(pl.col(numeric_col).is_not_null())
            .select(select_cols)
            .collect()
        )

    r = timed(_read)
    r.name = f"lazynwb.scan_nwb (filter + select {len(select_cols)} cols)"
    return r


# ---------------------------------------------------------------------------
# timeseries benchmarks
# ---------------------------------------------------------------------------

def bench_ts_pynwb(
    nwb_path: str, ts_search: str, n_samples: int | None = None,
) -> BenchmarkResult:
    """pynwb: open file, read(), find timeseries, read data."""
    def _read():
        nwbfile, io, h5_file = _open_pynwb(nwb_path)
        ts = _find_pynwb_timeseries(nwbfile, ts_search)
        slc = slice(None, n_samples)
        data = ts.data[slc]
        if ts.timestamps is not None:
            _ = ts.timestamps[slc]
        io.close()
        h5_file.close()
        return data

    r = timed(_read)
    suffix = f" [{n_samples} samples]" if n_samples else ""
    r.name = f"pynwb{suffix}"
    return r


def bench_ts_lazynwb(
    nwb_path: str, ts_search: str, n_samples: int | None = None,
) -> BenchmarkResult:
    """lazynwb.get_timeseries: open file, read data."""
    def _read():
        ts = lazynwb.get_timeseries(nwb_path, ts_search)
        slc = slice(None, n_samples)
        data = ts.data[slc]
        try:
            _ = ts.timestamps[slc]
        except Exception:
            pass
        return data

    r = timed(_read)
    suffix = f" [{n_samples} samples]" if n_samples else ""
    r.name = f"lazynwb.get_timeseries{suffix}"
    return r


def bench_ts_lazynwb_metadata_only(nwb_path: str, ts_search: str) -> BenchmarkResult:
    """lazynwb.get_timeseries: access metadata only (shape, dtype, unit) — no data download."""
    def _read():
        ts = lazynwb.get_timeseries(nwb_path, ts_search)
        return {
            "shape": ts.data.shape,
            "dtype": ts.data.dtype,
            "unit": ts.unit,
        }

    r = timed(_read)
    r.name = "lazynwb.get_timeseries (metadata only)"
    return r


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def resolve_nwb_path(nwb_path: str) -> str:
    """Try the given path; fall back to DANDI if S3 credentials are missing."""
    try:
        lazynwb.FileAccessor(nwb_path)
        lazynwb.clear_cache()
        return nwb_path
    except Exception as exc:
        print(f"Cannot open {nwb_path}: {exc}")
        if nwb_path != DANDI_FALLBACK_URL:
            print(f"Falling back to public DANDI NWB: {DANDI_FALLBACK_URL}")
            return DANDI_FALLBACK_URL
        raise


def _resolve_timeseries(available_ts: list[str]):
    ts_small = None
    ts_large = None
    for candidate in [TIMESERIES_SEARCH_SMALL, "lick", "running_speed"]:
        if any(candidate.lower() in t.lower() for t in available_ts):
            ts_small = candidate
            break
    for candidate in [TIMESERIES_SEARCH_LARGE, "wheel", "LFP"]:
        if any(candidate.lower() in t.lower() for t in available_ts):
            ts_large = candidate
            break
    if ts_small is None and available_ts:
        ts_small = available_ts[0].rsplit("/", 1)[-1]
    return ts_small, ts_large


def run_benchmarks(nwb_path: str = NWB_S3_PATH) -> None:
    nwb_path = resolve_nwb_path(nwb_path)

    print("=" * 80)
    print("NWB Streaming Benchmark")
    print(f"File: {nwb_path}")
    print("=" * 80)

    contents = discover_contents(nwb_path)
    lazynwb.clear_cache()

    # --- resolve table paths ---
    table_paths = []
    for candidate in [TABLE_PATH_TRIALS, TABLE_PATH_UNITS]:
        normalized = candidate.strip("/")
        if any(normalized in t or t == candidate for t in contents["tables"]):
            table_paths.append(candidate)
    if not table_paths:
        table_paths = contents["tables"][:2]

    ts_small, ts_large = _resolve_timeseries(contents["timeseries"])

    # ================================================================
    # TABLES
    # ================================================================
    for table_path in table_paths:
        has_arrays = table_path == TABLE_PATH_UNITS

        print()
        print("-" * 80)
        if has_arrays:
            print(f"TABLE: {table_path}  (has large array columns like spike_times)")
        else:
            print(f"TABLE: {table_path}  (scalar columns only)")
        print("-" * 80)

        if has_arrays:
            # --- scenario 1: all columns (apples-to-apples) ---
            print(f"\n  1. All columns (equivalent comparison):\n")
            for r in [
                bench_table_pynwb(nwb_path, table_path),
                bench_table_get_df_all(nwb_path, table_path),
            ]:
                print(r)

            # --- scenario 2: scalar columns only ---
            print(f"\n  2. Scalar columns only (skip array columns like spike_times):\n")
            for r in [
                bench_table_get_df_scalar(nwb_path, table_path),
                bench_table_scan_nwb(nwb_path, table_path),
            ]:
                print(r)

            # --- scenario 3: filter + select ---
            print(f"\n  3. Filter rows, then select columns (realistic QC workflow):\n")
            print(bench_table_scan_nwb_filter(nwb_path, table_path))
        else:
            # no array columns — all methods are equivalent
            print(f"\n  All methods read equivalent data:\n")
            for r in [
                bench_table_pynwb(nwb_path, table_path),
                bench_table_get_df_scalar(nwb_path, table_path),
                bench_table_scan_nwb(nwb_path, table_path),
                bench_table_scan_nwb_filter(nwb_path, table_path),
            ]:
                print(r)

        print()

    # ================================================================
    # TIMESERIES
    # ================================================================
    print("-" * 80)
    print("TIMESERIES")
    print("-" * 80)

    if ts_small:
        print(f"\n  Full download: {ts_small!r}\n")
        for r in [
            bench_ts_pynwb(nwb_path, ts_small),
            bench_ts_lazynwb(nwb_path, ts_small),
            bench_ts_lazynwb_metadata_only(nwb_path, ts_small),
        ]:
            print(r)

    if ts_large:
        print(f"\n  Partial read (first 10k samples): {ts_large!r}\n")
        for r in [
            bench_ts_pynwb(nwb_path, ts_large, n_samples=10_000),
            bench_ts_lazynwb(nwb_path, ts_large, n_samples=10_000),
            bench_ts_lazynwb_metadata_only(nwb_path, ts_large),
        ]:
            print(r)

    print()
    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    path = sys.argv[1] if len(sys.argv) > 1 else NWB_S3_PATH
    run_benchmarks(path)
