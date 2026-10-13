from __future__ import annotations

import contextlib
import dataclasses
import logging
import threading
import typing
from typing import Literal

import h5py
import numpy as np
import zarr

import lazynwb._catalog.models as catalog_models
import lazynwb._hdf5.reader as hdf5_reader
import lazynwb._zarr.reader as zarr_reader
import lazynwb.exceptions
import lazynwb.file_io
import lazynwb.tables
import lazynwb.types_
import lazynwb.utils

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class _RangeBackedHDF5Dataset:
    _source_url: str
    _column: catalog_models._TableColumnSchema

    @property
    def name(self) -> str:
        return f"/{self._column.dataset.path.removeprefix('/')}"

    @property
    def shape(self) -> tuple[int, ...]:
        return self._column.shape or ()

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.intp)) if self.shape else 1

    @property
    def dtype(self) -> np.dtype[typing.Any]:
        dtype = self._column.dtype
        if dtype.kind == "vlen_string":
            encoding = "ascii" if (dtype.detail or "").startswith("ascii") else "utf-8"
            return h5py.string_dtype(encoding=encoding)
        if dtype.kind == "array" and dtype.element_numpy_dtype is not None:
            element_dtype = np.dtype(dtype.element_numpy_dtype)
            if dtype.element_shape:
                return np.dtype((element_dtype, dtype.element_shape))
        if dtype.numpy_dtype is None:
            raise TypeError(f"{self.name} has no NumPy-compatible dtype")
        return np.dtype(dtype.numpy_dtype)

    @property
    def attrs(self) -> typing.Mapping[str, typing.Any]:
        return self._column.attrs

    @property
    def chunks(self) -> tuple[int, ...] | None:
        return self._column.dataset.chunks

    @property
    def compression(self) -> str | None:
        return self._column.dataset.compression

    @property
    def compression_opts(self) -> object | None:
        return self._column.dataset.compression_opts

    @property
    def maxshape(self) -> tuple[int | None, ...]:
        return self._column.dataset.maxshape or self.shape

    def __len__(self) -> int:
        if not self.shape:
            raise TypeError("Attempt to take len() of scalar dataset")
        return self.shape[0]

    def __getitem__(self, index: object) -> object:
        table_row_indices, local_index = _range_backed_first_axis_selection(
            index,
            shape=self.shape,
        )
        if not lazynwb.tables._has_direct_hdf5_layout(
            self._column
        ) or not lazynwb.tables._has_direct_hdf5_dtype(self._column):
            reason = lazynwb.tables._direct_hdf5_unsupported_reason(self._column)
            raise lazynwb.tables._unsupported_hdf5_layout_error(
                path=self._source_url,
                table_path=self._column.table_path,
                columns_and_reasons=((self._column.name, reason),),
            )

        reader = hdf5_reader._default_hdf5_backend_reader(self._source_url)
        request_count_before = int(getattr(reader._range_reader, "request_count", 0))
        fetched_bytes_before = int(getattr(reader._range_reader, "bytes_fetched", 0))
        try:
            values = lazynwb.tables._run_async_value(
                lazynwb.tables._read_direct_hdf5_column_array(
                    reader._range_reader,
                    self._column,
                    table_row_indices,
                    arrow_native=False,
                )
            )
        finally:
            lazynwb.tables._run_async_value(reader.close())
        value_array = np.asarray(values)
        if self._column.dtype.kind == "string":
            value_array = value_array.astype(self.dtype, copy=False)
        elif self._column.dtype.kind == "vlen_string":
            value_array = np.asarray(
                [
                    value.encode("utf-8") if isinstance(value, str) else value
                    for value in value_array.reshape(-1)
                ],
                dtype=object,
            ).reshape(value_array.shape)
        result = typing.cast(typing.Any, value_array)[local_index]
        logger.debug(
            "range-backed TimeSeries dataset read: source_url=%s dataset_path=%s "
            "index=%r selected_rows=%s result_shape=%s requests=%d bytes=%d",
            self._source_url,
            self.name,
            index,
            "all" if table_row_indices is None else len(table_row_indices),
            getattr(result, "shape", ()),
            int(getattr(reader._range_reader, "request_count", 0))
            - request_count_before,
            int(getattr(reader._range_reader, "bytes_fetched", 0))
            - fetched_bytes_before,
        )
        return result

    def __array__(
        self,
        dtype: np.dtype[typing.Any] | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        values = np.asarray(self[()])
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        if copy:
            values = values.copy()
        return values

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"shape={self.shape!r}, dtype={self.dtype!r})"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _RateBasedTimestamps:
    _starting_time: float
    _rate: float
    _sample_count: int
    _unit: str | None = None

    @property
    def name(self) -> str:
        return "timestamps"

    @property
    def shape(self) -> tuple[int]:
        return (self._sample_count,)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def size(self) -> int:
        return self._sample_count

    @property
    def dtype(self) -> np.dtype[np.float64]:
        return np.dtype(np.float64)

    @property
    def attrs(self) -> typing.Mapping[str, typing.Any]:
        return {} if self._unit is None else {"unit": self._unit}

    def __len__(self) -> int:
        return self._sample_count

    def __getitem__(self, index: object) -> object:
        row_indices, local_index = _range_backed_first_axis_selection(
            index,
            shape=self.shape,
        )
        if row_indices is None:
            timestamp_indices = np.arange(self._sample_count, dtype=np.float64)
        else:
            timestamp_indices = np.asarray(row_indices, dtype=np.float64)
        values = self._starting_time + timestamp_indices / self._rate
        result = typing.cast(typing.Any, values)[local_index]
        logger.debug(
            "generated bounded rate-derived TimeSeries timestamps: "
            "selected_samples=%d total_samples=%d starting_time=%s rate=%s",
            len(timestamp_indices),
            self._sample_count,
            self._starting_time,
            self._rate,
        )
        return result

    def __array__(
        self,
        dtype: np.dtype[typing.Any] | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        values = np.asarray(self[()])
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        if copy:
            values = values.copy()
        return values


def _range_backed_first_axis_selection(
    index: object,
    *,
    shape: tuple[int, ...],
) -> tuple[list[int] | None, tuple[object, ...]]:
    if not shape:
        is_empty_tuple = isinstance(index, tuple) and not index
        if not is_empty_tuple and index is not Ellipsis:
            raise ValueError("Illegal slicing argument for scalar dataspace")
        return None, ()

    selectors = index if isinstance(index, tuple) else (index,)
    ellipsis_count = sum(selector is Ellipsis for selector in selectors)
    if ellipsis_count > 1:
        raise ValueError("Only one ellipsis may be used in a dataset index")
    if ellipsis_count:
        ellipsis_index = next(
            selector_index
            for selector_index, selector in enumerate(selectors)
            if selector is Ellipsis
        )
        missing_count = len(shape) - (len(selectors) - 1)
        if missing_count < 0:
            raise ValueError(f"{len(selectors)} indexing arguments for {len(shape)} dimensions")
        selectors = (
            *selectors[:ellipsis_index],
            *(slice(None),) * missing_count,
            *selectors[ellipsis_index + 1 :],
        )
    if len(selectors) > len(shape):
        raise ValueError(f"{len(selectors)} indexing arguments for {len(shape)} dimensions")
    selectors = (*selectors, *(slice(None),) * (len(shape) - len(selectors)))
    if any(selector is None for selector in selectors):
        raise TypeError("New-axis indexing is not supported by HDF5 datasets")

    first_selector = selectors[0]
    if isinstance(first_selector, (int, np.integer)):
        first_index = int(first_selector)
        if first_index < 0:
            first_index += shape[0]
        if first_index < 0 or first_index >= shape[0]:
            raise IndexError(f"Index ({first_selector}) out of range for (0-{shape[0] - 1})")
        return [first_index], (0, *selectors[1:])
    if isinstance(first_selector, slice):
        start, stop, step = first_selector.indices(shape[0])
        if start == 0 and stop == shape[0] and step == 1:
            return None, (slice(None), *selectors[1:])
        return list(range(start, stop, step)), (slice(None), *selectors[1:])

    first_indices = np.asarray(first_selector)
    if first_indices.ndim != 1:
        raise TypeError("First-axis dataset indices must be one-dimensional")
    if first_indices.dtype.kind == "b":
        if len(first_indices) != shape[0]:
            raise TypeError(
                f"Boolean index has length {len(first_indices)}, expected {shape[0]}"
            )
        normalized = np.flatnonzero(first_indices).astype(np.intp, copy=False)
    elif first_indices.dtype.kind in {"i", "u"}:
        normalized = first_indices.astype(np.intp, copy=True)
        normalized[normalized < 0] += shape[0]
        if normalized.size and (normalized.min() < 0 or normalized.max() >= shape[0]):
            raise IndexError(f"Fancy index out of range for axis with size {shape[0]}")
    else:
        raise TypeError("First-axis dataset indices must be integers or booleans")
    return normalized.tolist(), (np.arange(len(normalized)), *selectors[1:])


@dataclasses.dataclass
class TimeSeries:
    _file_path: lazynwb.types_.PathLike
    _table_path: str
    _hdf5_source_url: str | None = dataclasses.field(default=None, repr=False)
    _path_metadata: typing.Mapping[str, typing.Any] | None = dataclasses.field(
        default=None,
        repr=False,
    )
    _hdf5_snapshot: catalog_models._TableSchemaSnapshot | None = dataclasses.field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _rate_timestamps: _RateBasedTimestamps | None = dataclasses.field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _snapshot_lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def _file(self) -> lazynwb.file_io.FileAccessor:
        return lazynwb.file_io._get_accessor(self._file_path)

    @property
    def data(self) -> h5py.Dataset | zarr.Array | _RangeBackedHDF5Dataset:
        if self._hdf5_source_url is not None:
            data = self._range_backed_dataset("data")
            if data is None:
                raise AttributeError(
                    f"{self._table_path} has no data: use event timestamps alone"
                ) from None
            return data
        file = self._file
        data_path = f"{self._table_path}/data"
        try:
            data = file[data_path]
        except KeyError:
            if self._table_path not in file:
                raise lazynwb.exceptions.InternalPathError(
                    f"{self._table_path} not found in file"
                ) from None
            raise AttributeError(
                f"{self._table_path} has no data: use event timestamps alone"
            ) from None
        logger.debug(
            "resolved TimeSeries data accessor: source_url=%s "
            "timeseries_path=%s data_path=%s shape=%s dtype=%s",
            file._path.as_posix(),
            self._table_path,
            data_path,
            getattr(data, "shape", None),
            getattr(data, "dtype", None),
        )
        return data

    @property
    def timestamps(
        self,
    ) -> h5py.Dataset | zarr.Array | _RangeBackedHDF5Dataset | _RateBasedTimestamps:
        if self._hdf5_source_url is not None:
            timestamps = self._range_backed_dataset("timestamps")
            if timestamps is not None:
                return timestamps
            if self._rate_timestamps is not None:
                return self._rate_timestamps
            rate = self.rate
            starting_time = self._starting_time
            if rate is None or starting_time is None:
                raise AssertionError(
                    "Not enough information to calculate timestamps for "
                    f"{self._table_path}: need rate and starting_time"
                ) from None
            starting_time_value = float(np.asarray(starting_time[()]).item())
            unit = starting_time.attrs.get("unit")
            generated = _RateBasedTimestamps(
                _starting_time=starting_time_value,
                _rate=float(rate),
                _sample_count=len(self.data),
                _unit=str(unit) if unit is not None else None,
            )
            self._rate_timestamps = generated
            logger.debug(
                "created lazy rate-derived TimeSeries timestamps: source_url=%s "
                "timeseries_path=%s sample_count=%d rate=%s",
                self._hdf5_source_url,
                self._table_path,
                len(generated),
                rate,
            )
            return generated
        file = self._file
        timestamps_path = f"{self._table_path}/timestamps"
        try:
            timestamps = file[timestamps_path]
        except KeyError:
            if self._table_path not in file:
                raise lazynwb.exceptions.InternalPathError(
                    f"{self._table_path} not found in file"
                ) from None
            rate = self.rate
            starting_time = self._starting_time
            if rate is None or starting_time is None:
                raise AssertionError(
                    "Not enough information to calculate timestamps for "
                    f"{self._table_path}: need rate and starting_time"
                ) from None
            generated_timestamps = (np.arange(len(self.data)) / rate) + starting_time
            logger.debug(
                "generated rate-derived TimeSeries timestamps: source_url=%s "
                "timeseries_path=%s sample_count=%d rate=%s",
                file._path.as_posix(),
                self._table_path,
                len(generated_timestamps),
                rate,
            )
            return generated_timestamps
        logger.debug(
            "resolved TimeSeries timestamps accessor: source_url=%s "
            "timeseries_path=%s timestamps_path=%s shape=%s dtype=%s",
            file._path.as_posix(),
            self._table_path,
            timestamps_path,
            getattr(timestamps, "shape", None),
            getattr(timestamps, "dtype", None),
        )
        return timestamps

    @property
    def electrodes(self) -> h5py.Dataset | zarr.Array | _RangeBackedHDF5Dataset:
        if self._hdf5_source_url is not None:
            electrodes = self._range_backed_dataset("electrodes")
            if electrodes is None:
                raise AttributeError(f"{self._table_path} has no electrode data") from None
            return electrodes
        try:
            return self._file[f"{self._table_path}/electrodes"]
        except KeyError:
            if self._table_path not in self._file:
                raise lazynwb.exceptions.InternalPathError(
                    f"{self._table_path} not found in file"
                ) from None
            raise AttributeError(f"{self._table_path} has no electrode data") from None

    @property
    def conversion(self) -> float | None:
        return self.data.attrs.get("conversion", None)

    @property
    def description(self) -> str | None:
        if self._path_metadata is not None:
            attrs = self._path_metadata.get("attrs", {})
            if isinstance(attrs, typing.Mapping):
                value = attrs.get("description")
                return str(value) if value is not None else None
        return self._file[f"{self._table_path}"].attrs.get("description", None)

    @property
    def offset(self) -> float | None:
        return self.data.attrs.get("offset", None)

    @property
    def rate(self) -> float | None:
        if (_starting_time := self._starting_time) is not None:
            return _starting_time.attrs.get("rate", None)
        return None

    @property
    def resolution(self) -> float | None:
        return self.data.attrs.get("resolution", None)

    @property
    def _starting_time(
        self,
    ) -> h5py.Dataset | zarr.Array | _RangeBackedHDF5Dataset | None:
        if self._hdf5_source_url is not None:
            return self._range_backed_dataset("starting_time")
        try:
            return self._file[f"{self._table_path}/starting_time"]
        except KeyError:
            if self._table_path not in self._file:
                raise lazynwb.exceptions.InternalPathError(
                    f"{self._table_path} not found in file"
                ) from None
            return None

    @property
    def starting_time(self) -> float:
        return float(np.asarray(self.timestamps[0]).item())

    @property
    def timestamps_unit(self) -> str | None:
        if self._path_metadata is not None:
            attrs = self._path_metadata.get("attrs", {})
            if isinstance(attrs, typing.Mapping) and "timestamps_unit" in attrs:
                value = attrs["timestamps_unit"]
                return str(value) if value is not None else None
        if self._hdf5_source_url is not None:
            timestamps = self._range_backed_dataset("timestamps")
            if timestamps is not None and "unit" in timestamps.attrs:
                value = timestamps.attrs["unit"]
                return str(value) if value is not None else None
            starting_time = self._starting_time
            if starting_time is not None and "unit" in starting_time.attrs:
                value = starting_time.attrs["unit"]
                return str(value) if value is not None else None
            raise AttributeError(
                f"Cannot find timestamps unit for {self._table_path}: "
                "no timestamps or starting_time found"
            )
        with contextlib.suppress(KeyError):
            return self._file[self._table_path].attrs["timestamps_unit"]
        with contextlib.suppress(KeyError):
            return self._file[f"{self._table_path}/timestamps"].attrs.get("unit", None)
        with contextlib.suppress(KeyError):
            return self._file[f"{self._table_path}/starting_time"].attrs.get(
                "unit", None
            )
        raise AttributeError(
            f"Cannot find timestamps unit for {self._table_path}: "
            "no timestamps or starting_time found"
        )

    @property
    def unit(self) -> str | None:
        return self.data.attrs.get("unit", None)

    def _range_backed_dataset(self, name: str) -> _RangeBackedHDF5Dataset | None:
        snapshot = self._range_backed_snapshot()
        column = next((column for column in snapshot.columns if column.name == name), None)
        if column is None or not column.is_dataset:
            return None
        logger.debug(
            "resolved range-backed TimeSeries child dataset: source_url=%s "
            "timeseries_path=%s child=%s shape=%s dtype=%s",
            self._hdf5_source_url,
            self._table_path,
            name,
            column.shape,
            column.dtype.numpy_dtype,
        )
        assert self._hdf5_source_url is not None
        return _RangeBackedHDF5Dataset(self._hdf5_source_url, column)

    def _range_backed_snapshot(self) -> catalog_models._TableSchemaSnapshot:
        if self._hdf5_snapshot is not None:
            return self._hdf5_snapshot
        if self._hdf5_source_url is None:
            raise RuntimeError("range-backed HDF5 snapshot requested for accessor source")
        with self._snapshot_lock:
            if self._hdf5_snapshot is None:
                reader = hdf5_reader._default_hdf5_backend_reader(
                    self._hdf5_source_url,
                    resolve_vlen_attributes=True,
                )
                normalized_path = lazynwb.utils.normalize_internal_file_path(
                    self._table_path.rstrip("/")
                )
                try:
                    snapshot = lazynwb.tables._run_async_value(
                        reader.read_table_schema_snapshot(normalized_path)
                    )
                except KeyError:
                    raise lazynwb.exceptions.InternalPathError(
                        f"{self._table_path} not found in file"
                    ) from None
                finally:
                    lazynwb.tables._run_async_value(reader.close())
                self._hdf5_snapshot = snapshot
                logger.debug(
                    "loaded range-backed TimeSeries snapshot: source_url=%s "
                    "timeseries_path=%s columns=%s",
                    self._hdf5_source_url,
                    self._table_path,
                    [column.name for column in snapshot.columns],
                )
        assert self._hdf5_snapshot is not None
        return self._hdf5_snapshot

    def __getattr__(
        self,
        name: str,
    ) -> h5py.Dataset | zarr.Array | _RangeBackedHDF5Dataset:
        if name.startswith("_"):
            raise AttributeError(name)
        if self._hdf5_source_url is not None:
            child = self._range_backed_dataset(name)
            if child is not None:
                return child
            raise AttributeError(
                f"'{self._table_path}' has no attribute '{name}'"
            ) from None
        try:
            return self._file[f"{self._table_path}/{name}"]
        except KeyError:
            raise AttributeError(
                f"'{self._table_path}' has no attribute '{name}'"
            ) from None


def _range_backed_hdf5_source_url(
    nwb_path: lazynwb.types_.PathLike,
) -> str | None:
    if zarr_reader._source_name_has_zarr_suffix(nwb_path):
        return None
    source = lazynwb.file_io._hdf5_catalog_source_if_available(nwb_path)
    if source is None or not hdf5_reader._is_fast_hdf5_candidate(source):
        return None
    return str(source)


def _get_timeseries_path_info(
    nwb_path: lazynwb.types_.PathLike,
) -> tuple[dict[str, dict[str, typing.Any]], str | None]:
    hdf5_source_url = _range_backed_hdf5_source_url(nwb_path)
    if hdf5_source_url is None:
        return (
            lazynwb.file_io.get_internal_path_info(
                nwb_path,
                include_child_datasets=True,
                parents=True,
            ),
            None,
        )

    reader = hdf5_reader._default_hdf5_backend_reader(
        hdf5_source_url,
        resolve_vlen_attributes=True,
    )
    try:
        try:
            entries = lazynwb.tables._run_async_value(reader.read_path_summary())
        except hdf5_reader._NotHDF5Error:
            logger.debug(
                "range-backed TimeSeries discovery rejected non-HDF5 source %r",
                nwb_path,
            )
            return (
                lazynwb.file_io.get_internal_path_info(
                    nwb_path,
                    include_child_datasets=True,
                    parents=True,
                ),
                None,
            )
    finally:
        lazynwb.tables._run_async_value(reader.close())
    filtered_entries = lazynwb.file_io._filter_catalog_path_summary_entries(
        entries,
        include_child_datasets=True,
        include_table_columns=False,
        include_metadata=False,
        include_specifications=False,
        parents=True,
    )
    for entry in entries:
        if entry.is_group:
            filtered_entries.setdefault(entry.path, entry)
    path_info = {
        path: lazynwb.file_io._path_metadata_from_summary_entry(entry)
        for path, entry in filtered_entries.items()
    }
    logger.debug(
        "range-backed TimeSeries discovery used HDF5 catalog: "
        "source_url=%s paths=%d",
        hdf5_source_url,
        len(path_info),
    )
    return path_info, hdf5_source_url


def _normalize_timeseries_path(path: str) -> str:
    normalized = lazynwb.utils.normalize_internal_file_path(path.rstrip("/"))
    return "/" if normalized == "/" else f"/{normalized}"


def _timeseries_path_from_candidate(path: str) -> str:
    return _normalize_timeseries_path(
        path.removesuffix("/data").removesuffix("/timestamps")
    )


def _get_exact_range_backed_hdf5_timeseries(
    nwb_path: lazynwb.types_.PathLike,
    search_term: str,
    hdf5_source_url: str,
) -> TimeSeries | None:
    """Resolve one exact HDF5 path without scanning unrelated file metadata."""
    normalized_search_path = _normalize_timeseries_path(search_term)
    reader_path = lazynwb.utils.normalize_internal_file_path(normalized_search_path)
    reader = hdf5_reader._default_hdf5_backend_reader(
        hdf5_source_url,
        resolve_vlen_attributes=True,
    )
    try:
        try:
            requested_entry = lazynwb.tables._run_async_value(
                reader._read_path_entry(reader_path)
            )
        except hdf5_reader._NotHDF5Error:
            logger.debug(
                "targeted range-backed TimeSeries discovery rejected "
                "non-HDF5 source %r",
                nwb_path,
            )
            return None
        except KeyError:
            logger.debug(
                "targeted TimeSeries path was not found: "
                "source_url=%s search_term=%r",
                hdf5_source_url,
                search_term,
            )
            raise lazynwb.exceptions.InternalPathError(
                f"Exact path {search_term!r} not found in file {hdf5_source_url}"
            ) from None

        timeseries_path = _timeseries_path_from_candidate(normalized_search_path)
        metadata_entry = requested_entry
        if timeseries_path != normalized_search_path:
            metadata_path = lazynwb.utils.normalize_internal_file_path(timeseries_path)
            try:
                metadata_entry = lazynwb.tables._run_async_value(
                    reader._read_path_entry(metadata_path)
                )
            except KeyError:
                raise lazynwb.exceptions.InternalPathError(
                    f"Exact path {search_term!r} not found in file {hdf5_source_url}"
                ) from None
    finally:
        lazynwb.tables._run_async_value(reader.close())

    logger.debug(
        "targeted range-backed TimeSeries discovery used exact HDF5 path: "
        "source_url=%s requested_path=%s timeseries_path=%s",
        hdf5_source_url,
        normalized_search_path,
        timeseries_path,
    )
    return TimeSeries(
        _file_path=nwb_path,
        _table_path=timeseries_path,
        _hdf5_source_url=hdf5_source_url,
        _path_metadata=lazynwb.file_io._path_metadata_from_summary_entry(
            metadata_entry
        ),
    )


@typing.overload
def get_timeseries(
    nwb_path: lazynwb.types_.PathLike,
    search_term: str | None = None,
    exact_path: bool = False,
    match_all: Literal[True] = True,
) -> dict[str, TimeSeries]:
    ...


@typing.overload
def get_timeseries(
    nwb_path: lazynwb.types_.PathLike,
    search_term: str | None = None,
    exact_path: bool = False,
    match_all: Literal[False] = False,
) -> TimeSeries:
    ...


def get_timeseries(
    nwb_path: lazynwb.types_.PathLike,
    search_term: str | None = None,
    exact_path: bool = False,
    match_all: bool = False,
) -> dict[str, TimeSeries] | TimeSeries:
    """
    Retrieve a TimeSeries object from an NWB file.
    This function searches for TimeSeries in an NWB file and returns either a specific
    TimeSeries object or a dictionary of all TimeSeries objects if `match_all` is True.

    Parameters
    ----------
    nwb_path : PathLike
        Path to an NWB file. Can be an hdf5 or zarr NWB.
    search_term : str or None, default=None
        String to search for specific TimeSeries. If the search term exactly matches a path,
        only that TimeSeries will be returned. If it partially matches multiple paths,
        the first match will be returned with a warning.
    exact_path: bool, default=False
        If True, the search term must exactly match the path of the TimeSeries. This is preferred
        as it is faster and less ambiguous.
    match_all : bool, default=False
        If True, returns all TimeSeries in the NWB as a dictionary regardless of search_term.

    Returns
    -------
    dict[str, TimeSeries] or TimeSeries
        If match_all is True, returns a dictionary mapping paths to TimeSeries objects.
        Otherwise, returns a single TimeSeries object, which is a dataclass, with attributes common
        to all NWB TimeSeries objects exposed, e.g. data, timestamps, rate, unit.
        Child datasets on specialized TimeSeries objects are exposed through the same
        lazy, dataset-like attribute interface.

    Raises
    ------
    ValueError
        If neither search_term is provided nor match_all is set to True.

    Notes
    -----
    The function identifies TimeSeries by looking for paths ending with "/data"
    or "/timestamps", which are characteristic of TimeSeries objects in NWB files.
    """
    if not (search_term or match_all):
        raise ValueError(
            "Either `search_term` must be specified or `match_all` must be set to True"
        )

    hdf5_source_url = _range_backed_hdf5_source_url(nwb_path)
    if (
        hdf5_source_url is not None
        and exact_path
        and not match_all
        and search_term is not None
    ):
        exact_timeseries = _get_exact_range_backed_hdf5_timeseries(
            nwb_path,
            search_term,
            hdf5_source_url,
        )
        if exact_timeseries is not None:
            return exact_timeseries

    path_info, hdf5_source_url = _get_timeseries_path_info(nwb_path)
    source_url = hdf5_source_url or lazynwb.file_io.from_pathlike(nwb_path).as_posix()
    logger.debug(
        "searching TimeSeries: source_url=%s search_term=%r "
        "exact_path=%s match_all=%s",
        source_url,
        search_term,
        exact_path,
        match_all,
    )
    normalized_search_path = (
        _normalize_timeseries_path(search_term) if search_term is not None else None
    )
    is_in_file = normalized_search_path is not None and normalized_search_path in path_info
    if (
        exact_path
        and not is_in_file
        and hdf5_source_url is None
        and search_term is not None
    ):
        is_in_file = search_term in lazynwb.file_io._get_accessor(nwb_path)
    if exact_path and not is_in_file:
        logger.debug(
            "exact TimeSeries path was not found: source_url=%s search_term=%r",
            source_url,
            search_term,
        )
        raise lazynwb.exceptions.InternalPathError(
            f"Exact path {search_term!r} not found in file {source_url}"
        )
    elif not match_all and search_term and is_in_file:
        assert normalized_search_path is not None
        timeseries_path = _timeseries_path_from_candidate(normalized_search_path)
        logger.debug(
            "selected exact TimeSeries path: source_url=%s timeseries_path=%s",
            source_url,
            timeseries_path,
        )
        return TimeSeries(
            _file_path=nwb_path,
            _table_path=timeseries_path,
            _hdf5_source_url=hdf5_source_url,
            _path_metadata=path_info.get(timeseries_path),
        )
    else:
        path_to_timeseries = {
            path: TimeSeries(
                _file_path=nwb_path,
                _table_path=path,
                _hdf5_source_url=hdf5_source_url,
                _path_metadata=metadata,
            )
            for path, metadata in path_info.items()
            if metadata["is_timeseries"] and (not search_term or search_term in path)
        }
        logger.debug(
            "discovered TimeSeries paths: source_url=%s search_term=%r "
            "match_count=%d paths=%s",
            source_url,
            search_term,
            len(path_to_timeseries),
            list(path_to_timeseries),
        )
        if match_all:
            return path_to_timeseries
        if not path_to_timeseries:
            logger.debug(
                "no TimeSeries paths matched search term: source_url=%s "
                "search_term=%r",
                source_url,
                search_term,
            )
            raise lazynwb.exceptions.InternalPathError(
                f"No TimeSeries matching {search_term!r} found in file {source_url}"
            )
        if len(path_to_timeseries) > 1:
            logger.warning(
                "Found multiple timeseries matching %r: %s - returning first",
                search_term,
                list(path_to_timeseries.keys()),
            )
        selected_path, selected_timeseries = next(iter(path_to_timeseries.items()))
        logger.debug(
            "selected discovered TimeSeries path: source_url=%s timeseries_path=%s",
            source_url,
            selected_path,
        )
        return selected_timeseries


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
