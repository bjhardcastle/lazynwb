import logging

import numpy as np
import pandas as pd
import polars as pl
import pynwb
import pytest

import lazynwb
import lazynwb._catalog.models as catalog_models
import lazynwb.tables


@pytest.mark.parametrize(
    "nwb_fixture_name",
    [
        "local_hdf5_path",
        "local_hdf5_paths",
        "local_zarr_path",
        "local_zarr_paths",
    ],
)
def test_sources(nwb_fixture_name, request):
    """Test get_df with various NWB file/store inputs."""
    # Resolve the fixture name to its value (the path or list of paths)
    nwb_path_or_paths = request.getfixturevalue(nwb_fixture_name)

    df = lazynwb.get_df(nwb_path_or_paths, "/intervals/trials", as_polars=True)
    assert not df.is_empty(), f"DataFrame is empty for {nwb_fixture_name}"


def test_internal_column_names(local_hdf5_path): 
    df = lazynwb.get_df(
        local_hdf5_path, "/intervals/trials"
    )
    for col in lazynwb.INTERNAL_COLUMN_NAMES:
        assert col in df.columns, f"Internal column {col!r} not found"


def test_get_df_uses_configured_polars_default(local_hdf5_path, monkeypatch):
    monkeypatch.setattr(lazynwb.config, "use_polars", True)

    df = lazynwb.get_df(local_hdf5_path, "/intervals/trials", exact_path=True)
    pandas_df = lazynwb.get_df(
        local_hdf5_path,
        "/intervals/trials",
        exact_path=True,
        as_polars=False,
    )

    assert isinstance(df, pl.DataFrame)
    assert isinstance(pandas_df, pd.DataFrame)


@pytest.mark.parametrize("table_name", ["trials", "units"])
def test_contents(local_hdf5_path, table_name):
    """Validate contents of dataframes against those obtained via pynwb"""
    exact_table_paths = {
        'trials': "/intervals/trials",
        'units': '/units',
    }
    df = (
        lazynwb.get_df(
            local_hdf5_path,
            exact_table_paths[table_name],
            exact_path=True,
            exclude_array_columns=False,
        )
        # we add internal columns for identifying source of rows when concatenating across files: 
        # drop them for comparison
        .drop(columns=lazynwb.INTERNAL_COLUMN_NAMES)
        .set_index('id')
    )
    nwb = pynwb.read_nwb(local_hdf5_path)
    reference_df = getattr(nwb, table_name).to_dataframe()
    pd.testing.assert_frame_equal(
        df,
        reference_df,
        check_dtype=True,
        check_exact=False,
        check_like=True,
    )

@pytest.mark.parametrize("table_shortcut", ['trials', 'epochs', 'session'])
def test_shortcuts(local_hdf5_path, table_shortcut: str):
    """Test that table shortcuts work as expected."""
    expected_path = lazynwb.TABLE_SHORTCUTS[table_shortcut]
    df = lazynwb.get_df(local_hdf5_path, table_shortcut, as_polars=True)
    assert not df.is_empty(), f"DataFrame fetched with {table_shortcut=} should not be empty"
    assert df['_table_path'].first() == expected_path, f"Table path should be full path, not {table_shortcut=}"

def test_general(local_hdf5_path):
    df = lazynwb.get_df(local_hdf5_path, "/general", as_polars=True)
    assert not df.is_empty(), f"'general' table should provide metadata from /general and top-level of file"
    assert 'session_start_time' in df.columns, f"'general' table should provide metadata from /general and top-level of file"

@pytest.mark.parametrize(
    "nwb_fixture_name",
    [
        "local_hdf5_path",
        "local_hdf5_paths",
    ],
)
def test_timeseries_with_rate(nwb_fixture_name, request):
    nwb_path_or_paths = request.getfixturevalue(nwb_fixture_name)
    # without timestamps, the default TimeSeries object has two keys: 'data' and
    # 'starting_time' which is another Group.
    # get_df() interprets them as 'data': List[float], 'starting_time': float
    # it needs to be aware of this possibility and generate a timestamps column
    df = lazynwb.get_df(nwb_path_or_paths, "processing/behavior/running_speed_with_rate", as_polars=True)
    assert 'timestamps' in df.columns, f"'trials' table should provide a 'timestamps' column"
    assert isinstance(df.schema['timestamps'], pl.Float64), f"'timestamps' column should be a float type, not {df.schema['timestamps']}"


def test_indexed_column_subset_reads_data_slices_not_full_column() -> None:
    data_accessor = _CountingArray([10, 11, 20, 21, 22, 30, 40, 41])
    index_accessor = _CountingArray([2, 5, 6, 8])

    result = lazynwb.tables._get_indexed_column_data(
        data_column_accessor=data_accessor,
        index_column_accessor=index_accessor,
        table_row_indices=[1, 3],
    )

    assert result == [[20, 21, 22], [40, 41]]
    assert data_accessor.keys == [slice(2, 5), slice(6, 8)]


def test_indexed_column_subset_coalesces_contiguous_rows_into_one_slice() -> None:
    data_accessor = _CountingArray([10, 11, 20, 21, 22, 30, 40, 41])
    index_accessor = _CountingArray([2, 5, 6, 8])

    result = lazynwb.tables._get_indexed_column_data(
        data_column_accessor=data_accessor,
        index_column_accessor=index_accessor,
        table_row_indices=[1, 2],
    )

    assert result == [[20, 21, 22], [30]]
    assert data_accessor.keys == [slice(2, 6)]


def test_indexed_column_full_table_reads_full_column_once() -> None:
    data_accessor = _CountingArray([10, 11, 20, 21, 22, 30, 40, 41])
    index_accessor = _CountingArray([2, 5, 6, 8])

    result = lazynwb.tables._get_indexed_column_data(
        data_column_accessor=data_accessor,
        index_column_accessor=index_accessor,
    )

    assert result == [[10, 11], [20, 21, 22], [30], [40, 41]]
    assert data_accessor.keys == [slice(None)]


def test_direct_zarr_numeric_column_reads_contiguous_row_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _FakeNativeZarrReader(
        {
            "units/id": np.asarray([10, 11, 12, 13], dtype=np.int64),
        }
    )
    monkeypatch.setattr(
        lazynwb.tables.zarr_reader,
        "_default_zarr_backend_reader",
        lambda path: reader,
    )
    id_column = _zarr_table_column(
        name="id",
        dataset_path="units/id",
        shape=(4,),
        chunks=(2,),
        dtype=np.dtype("int64"),
    )
    read_plan = lazynwb.tables._plan_direct_zarr_table_reads(
        (id_column,),
        table_row_indices=[1, 2],
    )

    result = lazynwb.tables._materialize_direct_zarr_read_plan(
        "s3://example-bucket/test.nwb.zarr",
        read_plan,
        table_row_indices=[1, 2],
        as_polars=True,
    )

    assert result is not None
    np.testing.assert_array_equal(result["id"], np.asarray([11, 12], dtype=np.int64))
    assert reader.calls == [("units/id", slice(1, 3))]
    assert reader.closed


def test_direct_zarr_indexed_column_reads_selected_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _FakeNativeZarrReader(
        {
            "units/spike_times": np.asarray(
                [10, 11, 20, 21, 22, 30, 40, 41],
                dtype=np.float64,
            ),
            "units/spike_times_index": np.asarray([2, 5, 6, 8], dtype=np.uint32),
        }
    )
    monkeypatch.setattr(
        lazynwb.tables.zarr_reader,
        "_default_zarr_backend_reader",
        lambda path: reader,
    )
    data_column = _zarr_table_column(
        name="spike_times",
        dataset_path="units/spike_times",
        shape=(8,),
        chunks=(4,),
        dtype=np.dtype("float64"),
        is_nominally_indexed=True,
        index_column_name="spike_times_index",
    )
    index_column = _zarr_table_column(
        name="spike_times_index",
        dataset_path="units/spike_times_index",
        shape=(4,),
        chunks=(4,),
        dtype=np.dtype("uint32"),
        is_index_column=True,
        data_column_name="spike_times",
    )
    read_plan = lazynwb.tables._plan_direct_zarr_table_reads(
        (data_column, index_column),
        table_row_indices=[1, 3],
    )

    result = lazynwb.tables._materialize_direct_zarr_read_plan(
        "s3://example-bucket/test.nwb.zarr",
        read_plan,
        table_row_indices=[1, 3],
        as_polars=True,
    )

    assert result == {"spike_times": [[20.0, 21.0, 22.0], [40.0, 41.0]]}
    assert reader.calls == [
        ("units/spike_times_index", slice(None)),
        ("units/spike_times", slice(2, 8)),
    ]
    assert reader.closed


def test_direct_zarr_materialization_falls_back_without_native_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _FakeNativeZarrReader(
        {
            "units/id": np.asarray([10, 11, 12, 13], dtype=np.int64),
        },
        native_transport=False,
    )
    monkeypatch.setattr(
        lazynwb.tables.zarr_reader,
        "_default_zarr_backend_reader",
        lambda path: reader,
    )
    id_column = _zarr_table_column(
        name="id",
        dataset_path="units/id",
        shape=(4,),
        chunks=(2,),
        dtype=np.dtype("int64"),
    )
    read_plan = lazynwb.tables._plan_direct_zarr_table_reads(
        (id_column,),
        table_row_indices=None,
    )

    result = lazynwb.tables._materialize_direct_zarr_read_plan(
        "file:///tmp/test.nwb.zarr",
        read_plan,
        table_row_indices=None,
        as_polars=True,
    )

    assert result is None
    assert reader.calls == []
    assert reader.closed


def test_get_df_row_index_lookup_uses_normalized_source_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def _fake_get_table_data(*args: object, **kwargs: object) -> dict[str, object]:
        del args
        calls.append(kwargs["table_row_indices"])
        return {"value": [10, 20]}

    monkeypatch.setattr(
        lazynwb.tables,
        "_catalog_snapshot_key",
        lambda path: f"normalized:{path}",
    )
    monkeypatch.setattr(lazynwb.tables, "_get_table_data", _fake_get_table_data)

    df = lazynwb.tables.get_df(
        "source",
        "/units",
        nwb_path_to_row_indices={"source": [2, 0]},
        parallel=False,
        as_polars=True,
    )

    assert calls == [[2, 0]]
    assert df["value"].to_list() == [10, 20]


class _CountingArray:
    def __init__(
        self,
        values: list[int],
        chunks: tuple[int, ...] | None = None,
    ) -> None:
        self._values = np.asarray(values)
        self.chunks = chunks
        self.keys: list[object] = []

    def __getitem__(self, key: object) -> np.ndarray:
        self.keys.append(key)
        return self._values[key]


class _FakeNativeZarrReader:
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        *,
        native_transport: bool = True,
    ) -> None:
        self._arrays = arrays
        self._remote_metadata_client = object() if native_transport else None
        self.metadata_read_count = 0
        self.metadata_bytes_fetched = 0
        self.chunk_read_count = 0
        self.chunk_bytes_fetched = 0
        self.calls: list[tuple[str, object]] = []
        self.closed = False

    async def read_array_selection(
        self,
        exact_array_path: str,
        selection: object = None,
    ) -> np.ndarray:
        self.calls.append((exact_array_path, selection))
        value = self._arrays[exact_array_path][selection]
        self.chunk_read_count += 1
        self.chunk_bytes_fetched += int(getattr(value, "nbytes", 0))
        return value

    async def close(self) -> None:
        self.closed = True


def _zarr_table_column(
    *,
    name: str,
    dataset_path: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: np.dtype,
    is_nominally_indexed: bool = False,
    is_index_column: bool = False,
    index_column_name: str | None = None,
    data_column_name: str | None = None,
) -> catalog_models._TableColumnSchema:
    return catalog_models._TableColumnSchema(
        name=name,
        table_path="units",
        source_path="s3://example-bucket/test.nwb.zarr",
        backend="zarr",
        dataset=catalog_models._DatasetSchema(
            path=dataset_path,
            dtype=catalog_models._NeutralDType.from_backend_dtype(dtype),
            shape=shape,
            ndim=len(shape),
            chunks=chunks,
            storage_layout="chunked",
            read_capabilities=("metadata", "shape", "dtype", "slice", "chunked"),
            is_dataset=True,
        ),
        is_nominally_indexed=is_nominally_indexed,
        is_index_column=is_index_column,
        index_column_name=index_column_name,
        data_column_name=data_column_name,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__])
