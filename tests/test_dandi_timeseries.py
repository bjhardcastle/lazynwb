"""Opt-in DANDI:001637 TimeSeries integration tests."""

from __future__ import annotations

import logging
import typing

import numpy as np
import pytest

import lazynwb
import lazynwb.file_io as file_io
import tests._dandi_sample as dandi_sample

pytestmark = [pytest.mark.integration, pytest.mark.dandi_sample]

_LOGGER = logging.getLogger(__name__)
_MAX_BOUNDED_AXIS_EXTENT = 32


class _ArrayLike(typing.Protocol):
    shape: tuple[int, ...]
    dtype: object

    def __getitem__(self, index: tuple[slice, ...]) -> object:
        ...


def test_dandi_001637_timeseries_discovery_exact_lookup_and_bounded_reads(
    dandi_001637_resolved_sample_assets: tuple[
        dandi_sample._DandiSampleResolvedAsset, ...
    ],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_timeseries_debug_logs(caplog)
    resolved_asset, timeseries_path = _discover_first_timeseries_path(
        dandi_001637_resolved_sample_assets,
        monkeypatch,
    )
    source_url = resolved_asset.source_url
    sanitized_source_url = dandi_sample._sanitize_source_url_for_logging(source_url)

    timeseries = lazynwb.get_timeseries(
        source_url,
        timeseries_path,
        exact_path=True,
    )

    assert isinstance(timeseries, lazynwb.TimeSeries)
    assert timeseries._table_path == timeseries_path
    _LOGGER.debug(
        "selected DANDI sample TimeSeries for bounded verification: "
        "source_url=%s exact_timeseries_path=%s asset_path=%s",
        sanitized_source_url,
        timeseries_path,
        resolved_asset.asset.path,
    )

    data = timeseries.data
    data_shape = _shape_tuple(data.shape)
    assert data_shape
    assert data_shape[0] > 0
    data_values, data_slice = _read_bounded_array(
        data,
        resolved_asset.asset.max_bounded_read_bytes,
    )
    assert data_values.size > 0
    assert data_values.nbytes <= resolved_asset.asset.max_bounded_read_bytes
    _LOGGER.debug(
        "bounded TimeSeries data read: source_url=%s exact_timeseries_path=%s "
        "data_shape=%s bounded_slice=%s bounded_read_shape=%s "
        "bounded_read_size=%d bounded_read_bytes=%d max_bounded_read_bytes=%d",
        sanitized_source_url,
        timeseries_path,
        data_shape,
        data_slice,
        data_values.shape,
        data_values.size,
        data_values.nbytes,
        resolved_asset.asset.max_bounded_read_bytes,
    )

    unit = timeseries.unit
    assert unit is not None
    _LOGGER.debug(
        "read TimeSeries unit metadata: source_url=%s exact_timeseries_path=%s "
        "unit=%r",
        sanitized_source_url,
        timeseries_path,
        unit,
    )

    timestamp_values = _read_bounded_timestamps_or_rate_derived(
        timeseries,
        source_url=source_url,
        sanitized_source_url=sanitized_source_url,
        timeseries_path=timeseries_path,
        bounded_sample_count=data_values.shape[0],
        max_bounded_read_bytes=resolved_asset.asset.max_bounded_read_bytes,
    )
    assert timestamp_values.size > 0
    assert timestamp_values.nbytes <= resolved_asset.asset.max_bounded_read_bytes

    log_text = caplog.text
    assert sanitized_source_url in log_text
    assert timeseries_path in log_text
    assert "bounded_read_size=" in log_text
    assert "bounded_read_bytes=" in log_text
    assert "built HDF5 path summary" in log_text
    assert "internal path discovery falling back to accessor traversal" not in log_text


def _discover_first_timeseries_path(
    resolved_assets: tuple[dandi_sample._DandiSampleResolvedAsset, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dandi_sample._DandiSampleResolvedAsset, str]:
    for resolved_asset in resolved_assets:
        sanitized_source_url = dandi_sample._sanitize_source_url_for_logging(
            resolved_asset.source_url
        )
        with monkeypatch.context() as discovery_patch:
            discovery_patch.setattr(file_io, "_get_accessor", _fail_accessor_traversal)
            path_info = file_io.get_internal_path_info(
                resolved_asset.source_url,
                include_child_datasets=True,
                parents=True,
            )
        timeseries_paths = _candidate_timeseries_paths(path_info)
        _LOGGER.debug(
            "discovered DANDI sample TimeSeries paths without broad raw reads: "
            "source_url=%s asset_path=%s exact_timeseries_paths=%s count=%d",
            sanitized_source_url,
            resolved_asset.asset.path,
            timeseries_paths,
            len(timeseries_paths),
        )
        if timeseries_paths:
            return resolved_asset, timeseries_paths[0]
    raise AssertionError("No bounded-readable TimeSeries paths discovered")


def _fail_accessor_traversal(*args: object, **kwargs: object) -> None:
    raise AssertionError(
        "DANDI TimeSeries discovery should use the catalog summary, not accessor traversal"
    )


def _candidate_timeseries_paths(
    path_info: dict[str, dict[str, object]],
) -> tuple[str, ...]:
    candidates: list[str] = []
    for path, metadata in path_info.items():
        if not metadata["is_timeseries"]:
            continue
        if f"{path}/data" not in path_info:
            continue
        starting_time_metadata = path_info.get(f"{path}/starting_time", {})
        starting_time_attrs = starting_time_metadata.get("attrs", {})
        has_timestamps = f"{path}/timestamps" in path_info
        has_rate_derived_timestamps = (
            isinstance(starting_time_attrs, dict) and "rate" in starting_time_attrs
        )
        if has_timestamps or has_rate_derived_timestamps:
            candidates.append(path)
    return tuple(sorted(candidates))


def _read_bounded_timestamps_or_rate_derived(
    timeseries: lazynwb.TimeSeries,
    *,
    source_url: str,
    sanitized_source_url: str,
    timeseries_path: str,
    bounded_sample_count: int,
    max_bounded_read_bytes: int,
) -> np.ndarray:
    file_accessor = file_io._get_accessor(source_url)
    timestamps_path = f"{timeseries_path}/timestamps"
    if timestamps_path in file_accessor:
        timestamps = timeseries.timestamps
        timestamp_values, timestamp_slice = _read_bounded_array(
            timestamps,
            max_bounded_read_bytes,
        )
        _LOGGER.debug(
            "bounded TimeSeries timestamps read: source_url=%s "
            "exact_timeseries_path=%s timestamps_shape=%s bounded_slice=%s "
            "bounded_read_shape=%s bounded_read_size=%d bounded_read_bytes=%d "
            "max_bounded_read_bytes=%d",
            sanitized_source_url,
            timeseries_path,
            _shape_tuple(timestamps.shape),
            timestamp_slice,
            timestamp_values.shape,
            timestamp_values.size,
            timestamp_values.nbytes,
            max_bounded_read_bytes,
        )
        return timestamp_values

    starting_time = timeseries._starting_time
    assert starting_time is not None
    rate = timeseries.rate
    assert rate is not None
    max_timestamp_count = max(
        1, max_bounded_read_bytes // np.dtype(np.float64).itemsize
    )
    timestamp_count = min(
        max(1, bounded_sample_count),
        _MAX_BOUNDED_AXIS_EXTENT,
        max_timestamp_count,
    )
    starting_time_value = float(np.asarray(starting_time[()]).item())
    timestamp_values = (
        np.arange(timestamp_count, dtype=np.float64) / float(rate)
    ) + starting_time_value
    _LOGGER.debug(
        "bounded TimeSeries rate-derived timestamps read: source_url=%s "
        "exact_timeseries_path=%s starting_time=%s rate=%s "
        "bounded_read_shape=%s bounded_read_size=%d bounded_read_bytes=%d "
        "max_bounded_read_bytes=%d",
        sanitized_source_url,
        timeseries_path,
        starting_time_value,
        rate,
        timestamp_values.shape,
        timestamp_values.size,
        timestamp_values.nbytes,
        max_bounded_read_bytes,
    )
    return timestamp_values


def _read_bounded_array(
    array: _ArrayLike,
    max_bounded_read_bytes: int,
) -> tuple[np.ndarray, tuple[slice, ...]]:
    dtype = np.dtype(array.dtype)
    bounded_slice = _bounded_slices(
        _shape_tuple(array.shape),
        dtype,
        max_bounded_read_bytes,
    )
    return np.asarray(array[bounded_slice]), bounded_slice


def _bounded_slices(
    shape: tuple[int, ...],
    dtype: np.dtype,
    max_bounded_read_bytes: int,
) -> tuple[slice, ...]:
    remaining_elements = max(1, max_bounded_read_bytes // max(1, dtype.itemsize))
    bounded_slices: list[slice] = []
    for axis_size in shape:
        if axis_size <= 0:
            bounded_slices.append(slice(0, 0))
            continue
        axis_count = min(
            axis_size,
            _MAX_BOUNDED_AXIS_EXTENT,
            remaining_elements,
        )
        axis_count = max(1, axis_count)
        bounded_slices.append(slice(0, axis_count))
        remaining_elements = max(1, remaining_elements // axis_count)
    return tuple(bounded_slices)


def _shape_tuple(shape: object) -> tuple[int, ...]:
    assert isinstance(shape, tuple)
    return tuple(int(dimension) for dimension in shape)


def _capture_timeseries_debug_logs(caplog: pytest.LogCaptureFixture) -> None:
    for logger_name in (
        __name__,
        dandi_sample._LOGGER_NAME,
        "lazynwb.timeseries",
        "lazynwb.file_io",
        "lazynwb._hdf5.reader",
        "lazynwb._hdf5.range_reader",
        "lazynwb._storage_options",
    ):
        caplog.set_level(logging.DEBUG, logger=logger_name)
