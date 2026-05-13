"""Opt-in DANDI:001637 table access integration tests."""

from __future__ import annotations

import logging

import polars as pl
import pytest

import lazynwb
import tests._dandi_sample as dandi_sample

pytestmark = [pytest.mark.integration, pytest.mark.dandi_sample]

_TABLE_PATH = "/units"
_NORMALIZED_TABLE_PATH = "units"
_SCALAR_COLUMNS = ("id", "firing_rate")
_ARRAY_COLUMN = "spike_times"
_BOUNDED_ROW_INDICES = (0, 1)
_BOUNDED_ARRAY_ROW_INDICES = (0,)


def test_dandi_001637_get_df_units_bounded_scalar_columns(
    dandi_001637_resolved_sample_assets: tuple[
        dandi_sample._DandiSampleResolvedAsset, ...
    ],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _capture_table_debug_logs(caplog)
    source_urls = _source_urls(dandi_001637_resolved_sample_assets)
    nwb_path_to_row_indices = _row_indices_by_source(
        source_urls,
        _BOUNDED_ROW_INDICES,
    )

    df = lazynwb.get_df(
        source_urls,
        _TABLE_PATH,
        exact_path=True,
        include_column_names=(
            *_SCALAR_COLUMNS,
            lazynwb.NWB_PATH_COLUMN_NAME,
            lazynwb.TABLE_PATH_COLUMN_NAME,
            lazynwb.TABLE_INDEX_COLUMN_NAME,
        ),
        exclude_array_columns=True,
        nwb_path_to_row_indices=nwb_path_to_row_indices,
        as_polars=True,
        disable_progress=True,
    )

    assert df.height == len(source_urls) * len(_BOUNDED_ROW_INDICES)
    assert set(_SCALAR_COLUMNS).issubset(df.columns)
    assert _ARRAY_COLUMN not in df.columns
    _assert_internal_source_columns(df, source_urls, _BOUNDED_ROW_INDICES)
    _assert_table_debug_logs(caplog)


def test_dandi_001637_scan_nwb_units_projection_filter_head_and_schema(
    dandi_001637_resolved_sample_assets: tuple[
        dandi_sample._DandiSampleResolvedAsset, ...
    ],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _capture_table_debug_logs(caplog)
    source_urls = _source_urls(dandi_001637_resolved_sample_assets)

    lf = lazynwb.scan_nwb(
        source_urls,
        _TABLE_PATH,
        infer_schema_length=len(source_urls),
        exclude_array_columns=True,
        disable_progress=True,
    )
    schema = lf.collect_schema()
    assert set(_SCALAR_COLUMNS).issubset(schema)
    assert _ARRAY_COLUMN not in schema

    df = (
        lf.select(
            *_SCALAR_COLUMNS,
            lazynwb.NWB_PATH_COLUMN_NAME,
            lazynwb.TABLE_PATH_COLUMN_NAME,
            lazynwb.TABLE_INDEX_COLUMN_NAME,
        )
        .filter(pl.col(lazynwb.TABLE_INDEX_COLUMN_NAME) < len(_BOUNDED_ROW_INDICES))
        .head(len(source_urls) * len(_BOUNDED_ROW_INDICES))
        .collect()
    )

    assert df.height == len(source_urls) * len(_BOUNDED_ROW_INDICES)
    _assert_internal_source_columns(df, source_urls, _BOUNDED_ROW_INDICES)
    assert "using batched fast HDF5 schema path" in caplog.text
    assert "Predicate specified: fetching initial columns" in caplog.text
    _assert_table_debug_logs(caplog)


def test_dandi_001637_get_df_units_array_column_exclusion_and_bounded_include(
    dandi_001637_resolved_sample_assets: tuple[
        dandi_sample._DandiSampleResolvedAsset, ...
    ],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _capture_table_debug_logs(caplog)
    source_url = _source_urls(dandi_001637_resolved_sample_assets)[0]

    excluded_df = lazynwb.get_df(
        source_url,
        _TABLE_PATH,
        exact_path=True,
        include_column_names=(
            *_SCALAR_COLUMNS,
            lazynwb.NWB_PATH_COLUMN_NAME,
            lazynwb.TABLE_PATH_COLUMN_NAME,
            lazynwb.TABLE_INDEX_COLUMN_NAME,
        ),
        exclude_array_columns=True,
        nwb_path_to_row_indices={source_url: _BOUNDED_ARRAY_ROW_INDICES},
        as_polars=True,
        disable_progress=True,
    )

    assert _ARRAY_COLUMN not in excluded_df.columns

    included_df = lazynwb.get_df(
        source_url,
        _TABLE_PATH,
        exact_path=True,
        include_column_names=(
            _ARRAY_COLUMN,
            lazynwb.NWB_PATH_COLUMN_NAME,
            lazynwb.TABLE_PATH_COLUMN_NAME,
            lazynwb.TABLE_INDEX_COLUMN_NAME,
        ),
        exclude_array_columns=False,
        nwb_path_to_row_indices={source_url: _BOUNDED_ARRAY_ROW_INDICES},
        as_polars=True,
        disable_progress=True,
    )

    assert included_df.height == 1
    assert isinstance(included_df.schema[_ARRAY_COLUMN], pl.List)
    assert 0 < len(included_df[_ARRAY_COLUMN][0]) < 10_000
    _assert_internal_source_columns(
        included_df,
        (source_url,),
        _BOUNDED_ARRAY_ROW_INDICES,
    )
    assert f"direct HDF5 indexed materialization for '{_ARRAY_COLUMN}'" in caplog.text
    assert "rows=1" in caplog.text
    _assert_table_debug_logs(caplog)


def _capture_table_debug_logs(caplog: pytest.LogCaptureFixture) -> None:
    for logger_name in (
        "lazynwb.lazyframe",
        "lazynwb.tables",
        "lazynwb._hdf5.reader",
        "lazynwb._hdf5.range_reader",
        "lazynwb._storage_options",
    ):
        caplog.set_level(logging.DEBUG, logger=logger_name)


def _source_urls(
    resolved_assets: tuple[dandi_sample._DandiSampleResolvedAsset, ...],
) -> tuple[str, ...]:
    return tuple(resolved_asset.source_url for resolved_asset in resolved_assets)


def _row_indices_by_source(
    source_urls: tuple[str, ...],
    row_indices: tuple[int, ...],
) -> dict[str, tuple[int, ...]]:
    return dict.fromkeys(source_urls, row_indices)


def _assert_internal_source_columns(
    df: pl.DataFrame,
    source_urls: tuple[str, ...],
    expected_table_indices: tuple[int, ...],
) -> None:
    assert set(source_urls) == set(df[lazynwb.NWB_PATH_COLUMN_NAME].unique())
    assert df[lazynwb.TABLE_PATH_COLUMN_NAME].unique().to_list() == [
        _NORMALIZED_TABLE_PATH
    ]
    for source_url in source_urls:
        source_df = df.filter(pl.col(lazynwb.NWB_PATH_COLUMN_NAME) == source_url)
        assert source_df[lazynwb.TABLE_INDEX_COLUMN_NAME].to_list() == list(
            expected_table_indices
        )


def _assert_table_debug_logs(caplog: pytest.LogCaptureFixture) -> None:
    log_text = caplog.text
    assert (
        "using fast catalog backend order" in log_text
        or "using scan-carried fast catalog snapshot" in log_text
        or "using batched fast HDF5 schema path" in log_text
    )
    assert "single-table HDF5 schema scan" in log_text
    assert (
        "built HDF5 table schema snapshot" in log_text
        or "loaded HDF5 table schema snapshot" in log_text
    )
    assert "direct HDF5 materialization" in log_text
    assert "requests=" in log_text
    assert "bytes=" in log_text
