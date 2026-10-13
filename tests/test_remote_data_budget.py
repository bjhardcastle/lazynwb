from __future__ import annotations

import asyncio
import dataclasses
import os
import pathlib
import time
import tracemalloc
from collections.abc import Callable

import h5py
import numpy as np
import pytest

import lazynwb
import lazynwb._cache.sqlite as cache_sqlite
import lazynwb._catalog.models as catalog_models
import lazynwb._hdf5.range_reader as hdf5_range_reader
import lazynwb._hdf5.reader as hdf5_reader
import lazynwb.types_

_REMOTE_DATA_SOURCE = (
    "s3://aind-scratch-data/dynamic-routing/cache/nwb/v0.0.272/"
    "620263_2022-07-26.nwb"
)
_REMOTE_DATA_TABLE = "/units"


@dataclasses.dataclass(frozen=True, slots=True)
class _QuerySpec:
    name: str
    column_names: tuple[str, ...]
    row_indices: tuple[int, ...]
    max_requests: int
    max_bytes: int
    max_peak_bytes: int


@dataclasses.dataclass(frozen=True, slots=True)
class _MaterializationBudgetMetric:
    source_url: str
    table_path: str
    query_shape: str
    request_count: int
    fetched_bytes: int
    elapsed_seconds: float
    peak_bytes: int
    row_count: int

    def as_properties(self) -> dict[str, object]:
        return dataclasses.asdict(self)

    def failure_detail(self) -> str:
        return (
            f"source={self.source_url!r} table={self.table_path!r} "
            f"query_shape={self.query_shape!r} rows={self.row_count} "
            f"requests={self.request_count} bytes={self.fetched_bytes} "
            f"elapsed={self.elapsed_seconds:.3f}s peak_bytes={self.peak_bytes}"
        )


_LOCAL_QUERY_SPECS = (
    _QuerySpec("scalar", ("scalar",), (0, 127, 255), 8, 64 * 1024, 4 * 1024 * 1024),
    _QuerySpec(
        "string",
        ("fixed_string", "vlen_string"),
        (0, 127, 255),
        12,
        128 * 1024,
        4 * 1024 * 1024,
    ),
    _QuerySpec(
        "compressed",
        ("compressed",),
        (0, 127, 255),
        10,
        96 * 1024,
        4 * 1024 * 1024,
    ),
    _QuerySpec(
        "sparse_ragged",
        ("ragged",),
        (0, 127, 255),
        10,
        96 * 1024,
        4 * 1024 * 1024,
    ),
    _QuerySpec(
        "large_array",
        ("large_array",),
        (200,),
        8,
        64 * 1024,
        4 * 1024 * 1024,
    ),
)


@pytest.mark.parametrize("query_spec", _LOCAL_QUERY_SPECS, ids=lambda spec: spec.name)
def test_direct_hdf5_materialization_stays_within_stable_budgets(
    query_spec: _QuerySpec,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    nwb_path = tmp_path / "materialization-budget.nwb"
    _write_materialization_budget_fixture(nwb_path)
    source_url = "https://performance-budget.test/materialization.nwb"
    range_reader = hdf5_range_reader._BufferRangeReader(nwb_path.read_bytes())
    reader = hdf5_reader._HDF5BackendReader(
        source_url,
        range_reader=range_reader,
        cache=cache_sqlite._SQLiteSnapshotCache(tmp_path / "catalog.sqlite"),
    )
    monkeypatch.setattr(
        hdf5_reader,
        "_default_hdf5_backend_reader",
        lambda source: reader,
    )

    tracemalloc.start()
    started = time.perf_counter()
    try:
        frame = lazynwb.get_df(
            source_url,
            "/table",
            exact_path=True,
            include_column_names=query_spec.column_names,
            exclude_array_columns=False,
            nwb_path_to_row_indices={source_url: query_spec.row_indices},
            as_polars=True,
            disable_progress=True,
        )
        elapsed_seconds = time.perf_counter() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    metric = _MaterializationBudgetMetric(
        source_url=source_url,
        table_path="/table",
        query_shape=query_spec.name,
        request_count=range_reader.request_count,
        fetched_bytes=range_reader.bytes_fetched,
        elapsed_seconds=elapsed_seconds,
        peak_bytes=peak_bytes,
        row_count=frame.height,
    )
    _record_metric(record_property, metric)

    assert frame.height == len(query_spec.row_indices), metric.failure_detail()
    assert set(query_spec.column_names).issubset(frame.columns), metric.failure_detail()
    assert metric.request_count <= query_spec.max_requests, metric.failure_detail()
    assert metric.fetched_bytes <= query_spec.max_bytes, metric.failure_detail()
    assert metric.peak_bytes <= query_spec.max_peak_bytes, metric.failure_detail()


@pytest.mark.remote_data_budget
@pytest.mark.skipif(
    os.environ.get("LAZYNWB_REMOTE_DATA_TESTS") != "1",
    reason="set LAZYNWB_REMOTE_DATA_TESTS=1 to run remote data budget tests",
)
def test_remote_hdf5_representative_materialization_budgets(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    monkeypatch.setenv("LAZYNWB_CATALOG_CACHE_PATH", str(tmp_path / "catalog.sqlite"))
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    lazynwb.config.anon = True
    schema_reader = hdf5_reader._default_hdf5_backend_reader(_REMOTE_DATA_SOURCE)
    snapshot = asyncio.run(
        schema_reader.read_table_schema_snapshot(
            lazynwb.normalize_internal_file_path(_REMOTE_DATA_TABLE)
        )
    )
    asyncio.run(schema_reader.close())
    representative_columns = _representative_remote_columns(snapshot)
    required_query_shapes = {
        "scalar",
        "string",
        "sparse_ragged",
        "large_array",
    }
    assert required_query_shapes.issubset(representative_columns), (
        f"remote source no longer covers required query shapes: {representative_columns}"
    )
    missing_query_shapes = sorted(
        {"scalar", "string", "compressed", "sparse_ragged", "large_array"}
        - representative_columns.keys()
    )
    record_property("remote_data_missing_query_shapes", ",".join(missing_query_shapes))

    max_requests = _env_int("LAZYNWB_REMOTE_DATA_MAX_GETS", 32)
    max_bytes = _env_int("LAZYNWB_REMOTE_DATA_MAX_BYTES", 16 * 1024 * 1024)
    max_seconds = _env_float("LAZYNWB_REMOTE_DATA_MAX_SECONDS", 10.0)
    max_peak_bytes = _env_int("LAZYNWB_REMOTE_DATA_MAX_PEAK_BYTES", 64 * 1024 * 1024)
    for query_shape, column_name in representative_columns.items():
        lazynwb.clear_cache()
        created_readers: list[hdf5_reader._HDF5BackendReader] = []
        original_reader_factory = hdf5_reader._default_hdf5_backend_reader

        with monkeypatch.context() as query_monkeypatch:
            query_monkeypatch.setattr(
                hdf5_reader,
                "_default_hdf5_backend_reader",
                _recording_hdf5_reader_factory(
                    original_reader_factory,
                    created_readers,
                ),
            )
            tracemalloc.start()
            started = time.perf_counter()
            try:
                frame = lazynwb.get_df(
                    _REMOTE_DATA_SOURCE,
                    _REMOTE_DATA_TABLE,
                    exact_path=True,
                    include_column_names=(column_name,),
                    exclude_array_columns=False,
                    nwb_path_to_row_indices={_REMOTE_DATA_SOURCE: (0,)},
                    _catalog_snapshots={
                        lazynwb.tables._catalog_snapshot_key(_REMOTE_DATA_SOURCE): snapshot
                    },
                    as_polars=True,
                    disable_progress=True,
                )
                elapsed_seconds = time.perf_counter() - started
                _, peak_bytes = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()

        metric = _MaterializationBudgetMetric(
            source_url=_REMOTE_DATA_SOURCE,
            table_path=_REMOTE_DATA_TABLE,
            query_shape=query_shape,
            request_count=sum(
                int(getattr(reader._range_reader, "request_count", 0))
                for reader in created_readers
            ),
            fetched_bytes=sum(
                int(getattr(reader._range_reader, "bytes_fetched", 0))
                for reader in created_readers
            ),
            elapsed_seconds=elapsed_seconds,
            peak_bytes=peak_bytes,
            row_count=frame.height,
        )
        _record_metric(record_property, metric)

        assert frame.height == 1, metric.failure_detail()
        assert column_name in frame.columns, metric.failure_detail()
        assert metric.request_count <= max_requests, metric.failure_detail()
        assert metric.fetched_bytes <= max_bytes, metric.failure_detail()
        assert metric.elapsed_seconds <= max_seconds, metric.failure_detail()
        assert metric.peak_bytes <= max_peak_bytes, metric.failure_detail()


def _write_materialization_budget_fixture(path: pathlib.Path) -> None:
    row_count = 256
    with h5py.File(path, "w") as h5_file:
        table = h5_file.create_group("table")
        table.create_dataset("scalar", data=np.arange(row_count, dtype=np.int64))
        table.create_dataset(
            "fixed_string",
            data=np.asarray([f"fixed-{index}" for index in range(row_count)], dtype="S16"),
        )
        table.create_dataset(
            "vlen_string",
            data=np.asarray([f"vlen-{index}" for index in range(row_count)], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        table.create_dataset(
            "compressed",
            data=np.arange(row_count, dtype=np.float64),
            chunks=(32,),
            compression="gzip",
            shuffle=True,
        )
        table.create_dataset(
            "ragged",
            data=np.arange(row_count * 2, dtype=np.float64),
        )
        table.create_dataset(
            "ragged_index",
            data=np.arange(2, (row_count * 2) + 1, 2, dtype=np.uint64),
        )
        table.create_dataset(
            "large_array",
            data=np.arange(row_count * 128, dtype=np.float32).reshape(row_count, 128),
        )


def _recording_hdf5_reader_factory(
    original_factory: Callable[
        [lazynwb.types_.PathLike], hdf5_reader._HDF5BackendReader
    ],
    created_readers: list[hdf5_reader._HDF5BackendReader],
) -> Callable[[lazynwb.types_.PathLike], hdf5_reader._HDF5BackendReader]:
    def _recording_factory(
        source: lazynwb.types_.PathLike,
    ) -> hdf5_reader._HDF5BackendReader:
        reader = original_factory(source)
        created_readers.append(reader)
        return reader

    return _recording_factory


def _representative_remote_columns(
    snapshot: catalog_models._TableSchemaSnapshot,
) -> dict[str, str]:
    columns_by_name = {column.name: column for column in snapshot.columns}
    candidates = {
        "scalar": next(
            (
                column.name
                for column in snapshot.columns
                if column.ndim == 1
                and column.dtype.kind in {"numeric", "bool"}
                and not column.is_index_column
                and not column.is_nominally_indexed
            ),
            None,
        ),
        "string": next(
            (
                column.name
                for column in snapshot.columns
                if column.ndim == 1 and column.dtype.kind in {"string", "vlen_string"}
                and not column.is_nominally_indexed
            ),
            None,
        ),
        "compressed": next(
            (
                column.name
                for column in snapshot.columns
                if column.ndim == 1
                and bool(column.dataset.filters)
                and not column.is_index_column
            ),
            None,
        ),
        "sparse_ragged": next(
            (
                column.name
                for column in snapshot.columns
                if column.is_nominally_indexed
                and not column.is_index_column
                and (column.index_column_name or f"{column.name}_index")
                in columns_by_name
            ),
            None,
        ),
        "large_array": next(
            (column.name for column in snapshot.columns if (column.ndim or 0) > 1),
            None,
        ),
    }
    return {name: column for name, column in candidates.items() if column is not None}


def _record_metric(
    record_property: Callable[[str, object], None],
    metric: _MaterializationBudgetMetric,
) -> None:
    prefix = f"{metric.query_shape}_materialization"
    for name, value in metric.as_properties().items():
        record_property(f"{prefix}_{name}", value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)
