from __future__ import annotations

import argparse
import contextlib
import dataclasses
import math
import statistics
import time
from collections.abc import Iterator

import numpy as np

import lazynwb
import lazynwb.dandi
import lazynwb.utils

try:
    import lindi
except ImportError:
    lindi = None


DEFAULT_DANDISET_ID = "000232"
DEFAULT_VERSION = "0.240510.2038"
DEFAULT_ASSET_ID = "4c440a73-9250-45bb-a342-a7da4d01b2fd"
DEFAULT_LFP_PATH = "/processing/ecephys/LFP/ElectricalSeries_0"
DEFAULT_TRIALS_PATH = "/intervals/trials"
DEFAULT_MODES = ("lazynwb_reference_zarr", "lindi_h5py", "raw_hdf5")
DEFAULT_TRIALS = 8
DEFAULT_REPEATS = 3


@dataclasses.dataclass
class TrialWindow:
    trial_index: int
    trial_id: int | None
    start_time: float
    stop_time: float
    start_sample: int
    stop_sample: int


@dataclasses.dataclass
class PreparedTrialWindows:
    windows: list[TrialWindow]
    total_trials: int
    overlapping_trials: int
    n_channels_total: int
    lfp_start_time: float
    lfp_stop_time: float


@dataclasses.dataclass
class RunResult:
    mode: str
    source: str
    elapsed_seconds: float
    trial_count: int
    channel_count: int
    total_samples: int
    total_values: int
    total_bytes: int
    value_sum: int
    trial_ids: list[int | None]
    chunk_shapes: list[tuple[int, int]]


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
) -> str:
    prefer_neurosift = mode != "raw_hdf5"
    with dandi_mode(prefer_neurosift=prefer_neurosift):
        source = lazynwb.dandi.resolve_dandi_asset_source(
            dandiset_id=dandiset_id,
            asset_id=asset_id,
            version=version,
            use_local_cache=False,
        )
    if prefer_neurosift and not str(source).endswith(".lindi.json"):
        raise ValueError(
            f"Expected a Neurosift LINDI JSON URL for {mode}, got {source!r}"
        )
    if not prefer_neurosift and str(source).endswith(".lindi.json"):
        raise ValueError(f"Expected a raw HDF5 URL for {mode}, got {source!r}")
    return source


def pick_trial_windows(
    *,
    source: str,
    trials_path: str,
    lfp_path: str,
    trial_limit: int,
    start_row: int | None,
    pre_seconds: float,
    post_seconds: float,
) -> PreparedTrialWindows:
    trials = lazynwb.get_df(
        source,
        trials_path,
        exact_path=True,
        as_polars=True,
        include_column_names=["start_time", "stop_time", "id"],
        disable_progress=True,
    ).sort(lazynwb.TABLE_INDEX_COLUMN_NAME)
    total_trials = trials.height
    if total_trials == 0:
        raise ValueError(f"No trials found in {trials_path}")

    ts = lazynwb.get_timeseries(source, lfp_path, exact_path=True)
    rate = ts.rate
    if rate is None:
        raise ValueError(
            f"{lfp_path} does not expose a sampling rate via starting_time/rate"
        )
    start_time0 = float(ts.starting_time)
    n_samples = int(ts.data.shape[0])
    lfp_stop_time = start_time0 + (n_samples / rate)

    overlapping_windows: list[TrialWindow] = []
    for row in trials.iter_rows(named=True):
        trial_start = float(row["start_time"]) - pre_seconds
        trial_stop = float(row["stop_time"]) + post_seconds
        start_sample = max(0, math.floor((trial_start - start_time0) * rate))
        stop_sample = min(n_samples, math.ceil((trial_stop - start_time0) * rate))
        if stop_sample <= start_sample:
            continue
        overlapping_windows.append(
            TrialWindow(
                trial_index=int(row[lazynwb.TABLE_INDEX_COLUMN_NAME]),
                trial_id=int(row["id"]) if row.get("id") is not None else None,
                start_time=trial_start,
                stop_time=trial_stop,
                start_sample=start_sample,
                stop_sample=stop_sample,
            )
        )

    overlapping_trials = len(overlapping_windows)
    if overlapping_trials == 0:
        raise ValueError("No trials overlap the LFP series after alignment")

    window_size = min(trial_limit, overlapping_trials)
    max_start = max(overlapping_trials - window_size, 0)
    start_pos = (
        max(0, min(start_row, max_start)) if start_row is not None else max_start // 2
    )
    windows = overlapping_windows[start_pos : start_pos + window_size]

    return PreparedTrialWindows(
        windows=windows,
        total_trials=total_trials,
        overlapping_trials=overlapping_trials,
        n_channels_total=int(ts.data.shape[1]),
        lfp_start_time=start_time0,
        lfp_stop_time=lfp_stop_time,
    )


def _resolve_channel_stop(
    *,
    n_channels_total: int,
    channel_start: int,
    channel_limit: int | None,
) -> int:
    channel_stop = (
        n_channels_total
        if channel_limit is None
        else min(n_channels_total, channel_start + channel_limit)
    )
    if channel_start < 0 or channel_start >= n_channels_total:
        raise ValueError(
            f"channel_start={channel_start} is outside the valid range 0..{n_channels_total - 1}"
        )
    if channel_stop <= channel_start:
        raise ValueError(
            f"Requested channel slice [{channel_start}:{channel_stop}] is empty"
        )
    return channel_stop


def _collect_chunk_stats(
    *,
    dataset: object,
    trial_windows: list[TrialWindow],
    channel_start: int,
    channel_stop: int,
) -> tuple[int, int, int, int, list[int | None], list[tuple[int, int]]]:
    total_samples = 0
    total_values = 0
    total_bytes = 0
    value_sum = 0
    chunk_shapes: list[tuple[int, int]] = []
    trial_ids: list[int | None] = []

    for window in trial_windows:
        chunk = np.asarray(
            dataset[
                window.start_sample : window.stop_sample,
                channel_start:channel_stop,
            ]
        )
        chunk_shapes.append((int(chunk.shape[0]), int(chunk.shape[1])))
        trial_ids.append(window.trial_id)
        total_samples += int(chunk.shape[0])
        total_values += int(chunk.size)
        total_bytes += int(chunk.nbytes)
        value_sum += int(chunk.astype(np.int64, copy=False).sum())

    return total_samples, total_values, total_bytes, value_sum, trial_ids, chunk_shapes


def run_query(
    *,
    mode: str,
    dandiset_id: str,
    asset_id: str,
    version: str,
    lfp_path: str,
    trial_windows: list[TrialWindow],
    channel_start: int,
    channel_limit: int | None,
) -> RunResult:
    lazynwb.clear_cache()
    t0 = time.perf_counter()
    source = resolve_source_for_mode(
        mode=mode,
        dandiset_id=dandiset_id,
        asset_id=asset_id,
        version=version,
    )

    if mode == "lindi_h5py":
        _require_lindi()
        normalized_lfp_path = lazynwb.utils.normalize_internal_file_path(lfp_path)
        file = lindi.LindiH5pyFile.from_lindi_file(source)
        try:
            series = file[normalized_lfp_path]
            data = series["data"]
            channel_stop = _resolve_channel_stop(
                n_channels_total=int(data.shape[1]),
                channel_start=channel_start,
                channel_limit=channel_limit,
            )
            (
                total_samples,
                total_values,
                total_bytes,
                value_sum,
                trial_ids,
                chunk_shapes,
            ) = _collect_chunk_stats(
                dataset=data,
                trial_windows=trial_windows,
                channel_start=channel_start,
                channel_stop=channel_stop,
            )
        finally:
            file.close()
    else:
        ts = lazynwb.get_timeseries(source, lfp_path, exact_path=True)
        channel_stop = _resolve_channel_stop(
            n_channels_total=int(ts.data.shape[1]),
            channel_start=channel_start,
            channel_limit=channel_limit,
        )
        (
            total_samples,
            total_values,
            total_bytes,
            value_sum,
            trial_ids,
            chunk_shapes,
        ) = _collect_chunk_stats(
            dataset=ts.data,
            trial_windows=trial_windows,
            channel_start=channel_start,
            channel_stop=channel_stop,
        )

    elapsed_seconds = time.perf_counter() - t0
    return RunResult(
        mode=mode,
        source=source,
        elapsed_seconds=elapsed_seconds,
        trial_count=len(trial_windows),
        channel_count=channel_stop - channel_start,
        total_samples=total_samples,
        total_values=total_values,
        total_bytes=total_bytes,
        value_sum=value_sum,
        trial_ids=trial_ids,
        chunk_shapes=chunk_shapes,
    )


def summarize(results: list[RunResult]) -> None:
    print("\nSummary")
    print("-------")
    by_mode: dict[str, list[RunResult]] = {}
    for result in results:
        by_mode.setdefault(result.mode, []).append(result)

    baseline_trial_ids: list[int | None] | None = None
    baseline_shapes: list[tuple[int, int]] | None = None
    baseline_sum: int | None = None
    for mode, mode_results in by_mode.items():
        timings = [r.elapsed_seconds for r in mode_results]
        total_samples = {r.total_samples for r in mode_results}
        total_bytes = {r.total_bytes for r in mode_results}
        value_sums = {r.value_sum for r in mode_results}
        print(
            f"{mode:>22}: median={statistics.median(timings):.3f}s "
            f"min={min(timings):.3f}s max={max(timings):.3f}s "
            f"samples={sorted(total_samples)} bytes={sorted(total_bytes)} "
            f"value_sum={sorted(value_sums)}"
        )
        print(f"{'':>22}  source={mode_results[0].source}")
        print(f"{'':>22}  trial_ids={mode_results[0].trial_ids}")
        print(f"{'':>22}  chunk_shapes={mode_results[0].chunk_shapes}")
        if baseline_trial_ids is None:
            baseline_trial_ids = mode_results[0].trial_ids
        elif mode_results[0].trial_ids != baseline_trial_ids:
            print(f"{'':>22}  WARNING: trial ids differ from the first mode")
        if baseline_shapes is None:
            baseline_shapes = mode_results[0].chunk_shapes
        elif mode_results[0].chunk_shapes != baseline_shapes:
            print(f"{'':>22}  WARNING: chunk shapes differ from the first mode")
        if baseline_sum is None:
            baseline_sum = mode_results[0].value_sum
        elif mode_results[0].value_sum != baseline_sum:
            print(f"{'':>22}  WARNING: value_sum differs from the first mode")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark reading trial-aligned chunks of LFP data from a DANDI asset "
            "via raw HDF5, the current lazynwb reference-Zarr path, and the lindi "
            "package."
        )
    )
    parser.add_argument("--dandiset-id", default=DEFAULT_DANDISET_ID)
    parser.add_argument("--asset-id", default=DEFAULT_ASSET_ID)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--lfp-path", default=DEFAULT_LFP_PATH)
    parser.add_argument("--trials-path", default=DEFAULT_TRIALS_PATH)
    parser.add_argument(
        "--modes",
        type=parse_modes,
        default=DEFAULT_MODES,
        help=(
            "Comma-separated modes. Choices: "
            f"{', '.join(DEFAULT_MODES)}. Default: {', '.join(DEFAULT_MODES)}"
        ),
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--start-row",
        type=int,
        default=None,
        help=(
            "Optional start index within the subset of trials that overlap the LFP "
            "recording. Default centers the selected window."
        ),
    )
    parser.add_argument("--channel-start", type=int, default=0)
    parser.add_argument(
        "--channel-limit",
        type=int,
        default=None,
        help="Optional number of channels to read per trial. Default is all channels.",
    )
    parser.add_argument("--pre-seconds", type=float, default=0.0)
    parser.add_argument("--post-seconds", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be a positive integer")
    if args.repeats <= 0:
        raise ValueError("--repeats must be a positive integer")
    if args.channel_limit is not None and args.channel_limit <= 0:
        raise ValueError("--channel-limit must be positive when provided")

    raw_source = resolve_source_for_mode(
        mode="raw_hdf5",
        dandiset_id=args.dandiset_id,
        asset_id=args.asset_id,
        version=args.version,
    )
    prepared = pick_trial_windows(
        source=raw_source,
        trials_path=args.trials_path,
        lfp_path=args.lfp_path,
        trial_limit=args.trials,
        start_row=args.start_row,
        pre_seconds=args.pre_seconds,
        post_seconds=args.post_seconds,
    )

    channel_count = (
        prepared.n_channels_total - args.channel_start
        if args.channel_limit is None
        else min(args.channel_limit, prepared.n_channels_total - args.channel_start)
    )

    print("Benchmark configuration")
    print("-----------------------")
    print(f"dandiset_id      : {args.dandiset_id}")
    print(f"asset_id         : {args.asset_id}")
    print(f"version          : {args.version}")
    print(f"lfp_path         : {args.lfp_path}")
    print(f"trials_path      : {args.trials_path}")
    print(f"modes            : {args.modes}")
    print(
        f"lfp_time_range   : "
        f"({prepared.lfp_start_time:.6f}, {prepared.lfp_stop_time:.6f})"
    )
    print(f"total_trials     : {prepared.total_trials}")
    print(f"overlapping      : {prepared.overlapping_trials}")
    print(f"selected_trials  : {[w.trial_id for w in prepared.windows]}")
    print(f"selected_rows    : {[w.trial_index for w in prepared.windows]}")
    print(
        f"sample_windows   : "
        f"{[(w.start_sample, w.stop_sample) for w in prepared.windows]}"
    )
    print(f"channel_start    : {args.channel_start}")
    print(f"channel_count    : {channel_count}")
    print(f"pre_seconds      : {args.pre_seconds}")
    print(f"post_seconds     : {args.post_seconds}")
    print(f"repeats          : {args.repeats}")

    results: list[RunResult] = []
    for mode in args.modes:
        for i in range(args.repeats):
            result = run_query(
                mode=mode,
                dandiset_id=args.dandiset_id,
                asset_id=args.asset_id,
                version=args.version,
                lfp_path=args.lfp_path,
                trial_windows=prepared.windows,
                channel_start=args.channel_start,
                channel_limit=args.channel_limit,
            )
            results.append(result)
            print(
                f"{result.mode:>22} run {i + 1}/{args.repeats}: "
                f"{result.elapsed_seconds:.3f}s trials={result.trial_count} "
                f"channels={result.channel_count} values={result.total_values} "
                f"bytes={result.total_bytes}"
            )

    summarize(results)


if __name__ == "__main__":
    main()
