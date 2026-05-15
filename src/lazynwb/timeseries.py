from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import time
import typing
from typing import Literal

import h5py
import numpy as np
import zarr

import lazynwb._zarr.reader as zarr_reader
import lazynwb.exceptions
import lazynwb.file_io
import lazynwb.types_
import lazynwb.utils

logger = logging.getLogger(__name__)


AsyncValueType = typing.TypeVar("AsyncValueType")


def _run_async_value(
    coroutine: typing.Coroutine[object, object, AsyncValueType],
) -> AsyncValueType:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    future = lazynwb.utils.get_threadpool_executor().submit(asyncio.run, coroutine)
    return future.result()


def _native_zarr_array_accessor_if_available(
    *,
    file: lazynwb.file_io.FileAccessor,
    array_path: str,
    accessor: h5py.Dataset | zarr.Array,
) -> h5py.Dataset | zarr.Array | _NativeZarrArrayAccessor:
    if file._hdmf_backend != lazynwb.file_io.FileAccessor.HDMFBackend.ZARR:
        return accessor
    if not zarr_reader._source_supports_native_zarr_array_reads(file._path):
        logger.debug(
            "leaving TimeSeries data on Zarr accessor path for %s/%s: "
            "native obstore transport unsupported",
            file._path.as_posix(),
            array_path,
        )
        return accessor
    if not _array_accessor_metadata_supports_native_zarr(accessor):
        logger.debug(
            "leaving TimeSeries data on Zarr accessor path for %s/%s: "
            "native array metadata unsupported (shape=%s dtype=%s chunks=%s)",
            file._path.as_posix(),
            array_path,
            getattr(accessor, "shape", None),
            getattr(accessor, "dtype", None),
            getattr(accessor, "chunks", None),
        )
        return accessor
    logger.debug(
        "wrapping TimeSeries data accessor with native Zarr selector for %s/%s",
        file._path.as_posix(),
        array_path,
    )
    return _NativeZarrArrayAccessor(
        source=file._path,
        array_path=array_path,
        fallback=accessor,
    )


def _array_accessor_metadata_supports_native_zarr(
    accessor: h5py.Dataset | zarr.Array,
) -> bool:
    dtype = getattr(accessor, "dtype", None)
    shape = getattr(accessor, "shape", None)
    chunks = getattr(accessor, "chunks", None)
    if dtype is None or shape is None or chunks is None:
        return False
    try:
        np_dtype = np.dtype(dtype)
    except TypeError:
        return False
    return len(tuple(shape)) in (1, 2) and np_dtype.kind in {"b", "i", "u", "f", "c"}


class _NativeZarrArrayAccessor:
    """Private array-like wrapper that tries native exact Zarr reads before fallback."""

    def __init__(
        self,
        *,
        source: lazynwb.types_.PathLike,
        array_path: str,
        fallback: h5py.Dataset | zarr.Array,
    ) -> None:
        self._source = source
        self._array_path = array_path
        self._fallback = fallback

    @property
    def attrs(self) -> object:
        return self._fallback.attrs

    @property
    def chunks(self) -> object:
        return getattr(self._fallback, "chunks", None)

    @property
    def dtype(self) -> object:
        return getattr(self._fallback, "dtype", None)

    @property
    def name(self) -> str | None:
        return getattr(self._fallback, "name", None)

    @property
    def ndim(self) -> int | None:
        return getattr(self._fallback, "ndim", None)

    @property
    def shape(self) -> tuple[int, ...] | None:
        return getattr(self._fallback, "shape", None)

    def __len__(self) -> int:
        return len(self._fallback)

    def __array__(
        self,
        dtype: object | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        value = np.asarray(self[:])
        if dtype is not None:
            value = value.astype(dtype)
        if copy:
            return value.copy()
        return value

    def __getattr__(self, name: str) -> object:
        return getattr(self._fallback, name)

    def __getitem__(self, selection: object) -> object:
        reader = zarr_reader._default_zarr_backend_reader(self._source)
        started = time.perf_counter()
        try:
            value = _run_async_value(
                reader.read_array_selection(self._array_path, selection)
            )
        except Exception as exc:
            logger.debug(
                "native TimeSeries Zarr data read unavailable for %r/%s "
                "selection=%r after %.3f s; using accessor fallback: %r",
                self._source,
                self._array_path,
                selection,
                time.perf_counter() - started,
                exc,
                exc_info=True,
            )
            return self._fallback[selection]
        finally:
            _run_async_value(reader.close())
        logger.debug(
            "read TimeSeries Zarr data via native selector for %r/%s "
            "selection=%r in %.3f s",
            self._source,
            self._array_path,
            selection,
            time.perf_counter() - started,
        )
        return value


@dataclasses.dataclass
class TimeSeries:
    _file_path: lazynwb.types_.PathLike
    _table_path: str

    @property
    def _file(self) -> lazynwb.file_io.FileAccessor:
        return lazynwb.file_io._get_accessor(self._file_path)

    @property
    def data(self) -> h5py.Dataset | zarr.Array | _NativeZarrArrayAccessor:
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
        return _native_zarr_array_accessor_if_available(
            file=file,
            array_path=data_path,
            accessor=data,
        )

    @property
    def timestamps(self) -> h5py.Dataset | zarr.Array:
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
    def electrodes(self) -> h5py.Dataset | zarr.Array:
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
    def _starting_time(self) -> h5py.Dataset | zarr.Array | None:
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
        return self.timestamps[0]

    @property
    def timestamps_unit(self) -> str | None:
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

    def __getattr__(self, name: str) -> h5py.Dataset | zarr.Array:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._file[f"{self._table_path}/{name}"]
        except KeyError:
            raise AttributeError(
                f"'{self._table_path}' has no attribute '{name}'"
            ) from None


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
        For specialized TimeSeries objects, other attributes may be accessed via the h5py/zarr
        accessor using the `file` and `path` attributes, e.g. `ts.file[ts.path + '/data']`

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

    def _format(name: str) -> str:
        return name.removesuffix("/data").removesuffix("/timestamps")

    file = lazynwb.file_io._get_accessor(nwb_path)
    source_url = file._path.as_posix()
    logger.debug(
        "searching TimeSeries: source_url=%s search_term=%r "
        "exact_path=%s match_all=%s",
        source_url,
        search_term,
        exact_path,
        match_all,
    )
    is_in_file = search_term is not None and search_term in file
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
        timeseries_path = _format(search_term)
        logger.debug(
            "selected exact TimeSeries path: source_url=%s timeseries_path=%s",
            source_url,
            timeseries_path,
        )
        return TimeSeries(_file_path=nwb_path, _table_path=timeseries_path)
    else:
        path_info = lazynwb.file_io.get_internal_path_info(nwb_path)
        path_to_timeseries = {
            path: TimeSeries(_file_path=nwb_path, _table_path=path)
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
