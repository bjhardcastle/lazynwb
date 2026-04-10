import logging

import pandas as pd
import polars as pl
import pytest
import lazynwb


def test_import_package():
    pass


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


def test_get_metadata_df_zarr(local_zarr_path):
    df = lazynwb.get_metadata_df(local_zarr_path, disable_progress=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "subject_id" in df.columns


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, '-v'])
