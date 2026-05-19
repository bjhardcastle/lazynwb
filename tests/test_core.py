import json
import logging
import pathlib

import numcodecs
import numpy as np
import pandas as pd
import polars as pl
import pytest

import lazynwb


def test_import_package():
    import lazynwb  # noqa: F401


def test_describe(local_hdf5_path):
    nwb = lazynwb.LazyNWB(local_hdf5_path)
    result = nwb.describe()
    assert isinstance(result, dict)
    assert "paths" in result
    assert isinstance(result["paths"], list)
    assert len(result["paths"]) > 0
    assert "session_description" in result
    assert "session_start_time" in result
    assert "subject_id" in result
    assert "species" in result


def test_describe_zarr(local_zarr_path):
    nwb = lazynwb.LazyNWB(local_zarr_path)
    result = nwb.describe()
    assert isinstance(result, dict)
    assert "paths" in result
    assert len(result["paths"]) > 0
    assert "session_description" in result
    assert "subject_id" in result


def test_repr_html_tolerates_zarr_metadata_read_failure(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "bad-metadata.nwb.zarr"
    _write_zarr_group(zarr_path)
    _write_zarr_group(zarr_path / "general")
    _write_zarr_group(zarr_path / "general" / "subject")
    _write_invalid_vlen_bytes_array(zarr_path / "identifier", b"test-id")
    _write_invalid_vlen_bytes_array(zarr_path / "session_description", b"test session")
    _write_invalid_vlen_bytes_array(
        zarr_path / "session_start_time",
        b"2024-01-02T03:04:05+00:00",
    )
    monkeypatch.setattr(
        lazynwb.file_io,
        "_get_zarr_v3_catalog_accessor",
        lambda *args, **kwargs: None,
    )

    html = lazynwb.LazyNWB(zarr_path)._repr_html_()

    assert "NWB file:" in html
    assert "identifier" in html
    assert "unavailable" in html
    assert "/identifier" in html
    assert "/session_start_time" in html
    assert "/general/subject" in html
    assert "Paths" in html


def test_get_metadata_df_single(local_hdf5_path):
    df = lazynwb.get_metadata_df(local_hdf5_path, disable_progress=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "session_description" in df.columns
    assert "subject_id" in df.columns
    assert lazynwb.NWB_PATH_COLUMN_NAME in df.columns


def test_get_metadata_df_multiple(local_hdf5_paths):
    df = lazynwb.get_metadata_df(local_hdf5_paths, disable_progress=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(local_hdf5_paths)
    assert lazynwb.NWB_PATH_COLUMN_NAME in df.columns


def test_get_metadata_df_polars(local_hdf5_path):
    df = lazynwb.get_metadata_df(local_hdf5_path, disable_progress=True, as_polars=True)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 1
    assert "session_description" in df.columns
    assert lazynwb.NWB_PATH_COLUMN_NAME in df.columns


def test_get_metadata_df_uses_configured_polars_default(local_hdf5_path, monkeypatch):
    monkeypatch.setattr(lazynwb.config, "use_polars", True)

    df = lazynwb.get_metadata_df(local_hdf5_path, disable_progress=True)
    pandas_df = lazynwb.get_metadata_df(
        local_hdf5_path,
        disable_progress=True,
        as_polars=False,
    )

    assert isinstance(df, pl.DataFrame)
    assert isinstance(pandas_df, pd.DataFrame)


def test_lazynwb_get_df_uses_configured_polars_default(local_hdf5_path, monkeypatch):
    monkeypatch.setattr(lazynwb.config, "use_polars", True)
    nwb = lazynwb.LazyNWB(local_hdf5_path)

    df = nwb.get_df("/intervals/trials", exact_path=True)
    pandas_df = nwb.get_df("/intervals/trials", exact_path=True, as_polars=False)

    assert isinstance(df, pl.DataFrame)
    assert isinstance(pandas_df, pd.DataFrame)


def test_get_returns_dataframe_for_table(local_hdf5_path):
    df = lazynwb.get(local_hdf5_path, "/intervals/trials", exact_path=True)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_get_returns_timeseries_for_timeseries_container(local_hdf5_path):
    result = lazynwb.get(
        local_hdf5_path,
        "/processing/behavior/running_speed_with_timestamps",
        exact_path=True,
    )

    assert isinstance(result, lazynwb.TimeSeries)
    assert result._table_path == "/processing/behavior/running_speed_with_timestamps"


def test_get_force_as_df_for_timeseries(local_hdf5_path):
    df = lazynwb.get(
        local_hdf5_path,
        "/processing/behavior/running_speed_with_timestamps",
        exact_path=True,
        as_df=True,
        as_polars=True,
    )

    assert isinstance(df, pl.DataFrame)
    assert "timestamps" in df.columns
    assert "data" in df.columns


def test_lazynwb_get_uses_general_get(local_hdf5_path):
    nwb = lazynwb.LazyNWB(local_hdf5_path)

    result = nwb.get("running_speed_with_rate")

    assert isinstance(result, lazynwb.TimeSeries)


def test_get_metadata_df_zarr(local_zarr_path):
    df = lazynwb.get_metadata_df(local_zarr_path, disable_progress=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "subject_id" in df.columns


def _write_zarr_group(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (path / ".zattrs").write_text("{}")


def _write_invalid_vlen_bytes_array(path: pathlib.Path, value: bytes) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".zarray").write_text(
        json.dumps(
            {
                "zarr_format": 2,
                "shape": [1],
                "chunks": [1],
                "dtype": "|O",
                "compressor": {"id": "vlen-bytes"},
                "fill_value": 0,
                "order": "C",
                "filters": None,
            }
        )
    )
    (path / ".zattrs").write_text("{}")
    chunk = numcodecs.VLenBytes().encode(np.array([value], dtype=object))
    (path / "0").write_bytes(chunk)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v"])
