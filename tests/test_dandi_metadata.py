"""Opt-in DANDI:001637 metadata and internal path integration tests."""

from __future__ import annotations

import datetime
import logging

import polars as pl
import pytest

import lazynwb
import lazynwb.file_io as file_io
import tests._dandi_sample as dandi_sample

pytestmark = [pytest.mark.integration, pytest.mark.dandi_sample]

_CORE_METADATA_COLUMNS = (
    "identifier",
    "session_description",
    "session_start_time",
    "subject_id",
    "species",
    lazynwb.NWB_PATH_COLUMN_NAME,
)


def test_dandi_001637_metadata_df_describe_paths_and_attribute_access(
    dandi_001637_resolved_sample_assets: tuple[
        dandi_sample._DandiSampleResolvedAsset, ...
    ],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_metadata_debug_logs(caplog)
    source_urls = _source_urls(dandi_001637_resolved_sample_assets)
    source_url = source_urls[0]

    metadata_df = lazynwb.get_metadata_df(
        source_urls,
        disable_progress=True,
        as_polars=True,
    )

    assert isinstance(metadata_df, pl.DataFrame)
    assert metadata_df.height == len(source_urls)
    assert set(_CORE_METADATA_COLUMNS).issubset(metadata_df.columns)
    assert metadata_df["identifier"].null_count() == 0
    assert metadata_df["session_description"].null_count() == 0
    assert metadata_df["subject_id"].null_count() == 0
    assert set(metadata_df[lazynwb.NWB_PATH_COLUMN_NAME].to_list()) == set(source_urls)

    nwb = lazynwb.LazyNWB(source_url)
    assert isinstance(nwb.identifier, str)
    assert isinstance(nwb.session_description, str)
    assert isinstance(nwb.session_start_time, datetime.datetime)
    assert isinstance(nwb.subject.subject_id, str)
    assert nwb.subject.species is None or isinstance(nwb.subject.species, str)

    description = nwb.describe()
    assert {"paths", "identifier", "session_description", "subject_id"}.issubset(
        description
    )
    assert "/units" in description["paths"]

    with monkeypatch.context() as path_summary_patch:
        path_summary_patch.setattr(file_io, "_get_accessor", _fail_accessor_traversal)
        paths = file_io.get_internal_paths(
            source_url,
            include_child_datasets=True,
            include_metadata=True,
            parents=True,
        )
        path_info = file_io.get_internal_path_info(
            source_url,
            include_child_datasets=True,
            include_metadata=True,
            parents=True,
        )

    assert "/general/subject" in paths
    assert "/units" in paths
    assert "/general/subject" in path_info
    assert "/units" in path_info
    assert path_info["/general/subject"]["is_group"] is True
    assert isinstance(path_info["/general/subject"]["attrs"], dict)
    assert isinstance(path_info["/units"]["attrs"], dict)
    assert any(entry["is_timeseries"] for entry in path_info.values())

    _assert_metadata_debug_logs(caplog)


def _fail_accessor_traversal(*args: object, **kwargs: object) -> None:
    raise AssertionError(
        "DANDI path discovery should use the catalog summary, not accessor traversal"
    )


def _source_urls(
    resolved_assets: tuple[dandi_sample._DandiSampleResolvedAsset, ...],
) -> tuple[str, ...]:
    return tuple(resolved_asset.source_url for resolved_asset in resolved_assets)


def _capture_metadata_debug_logs(caplog: pytest.LogCaptureFixture) -> None:
    for logger_name in (
        "lazynwb.base",
        "lazynwb.file_io",
        "lazynwb._cache.sqlite",
        "lazynwb._hdf5.reader",
        "lazynwb._hdf5.range_reader",
        "lazynwb._storage_options",
        dandi_sample._LOGGER_NAME,
    ):
        caplog.set_level(logging.DEBUG, logger=logger_name)


def _assert_metadata_debug_logs(caplog: pytest.LogCaptureFixture) -> None:
    log_text = caplog.text
    assert "metadata read scope=session+subject" in log_text
    assert "metadata read scope=top_level" in log_text
    assert "metadata read scope=subject" in log_text
    assert "using catalog path summary backend order" in log_text
    assert (
        "source identity cache hit" in log_text
        or "source identity cache miss" in log_text
        or "parsed HDF5 metadata cache hit" in log_text
        or "parsed HDF5 metadata cache miss" in log_text
    )
    assert (
        "built HDF5 path summary" in log_text
        or "built parser-backed HDF5 path summary" in log_text
    )
    assert "internal path discovery falling back to accessor traversal" not in log_text
