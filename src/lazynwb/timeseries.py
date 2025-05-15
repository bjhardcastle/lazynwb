from __future__ import annotations

import contextlib
import dataclasses
import logging
import typing
from typing import Literal

import h5py
import npc_io
import numpy as np
import zarr

import lazynwb.exceptions
import lazynwb.file_io
import lazynwb.utils

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TimeSeries:
    _file: lazynwb.file_io.FileAccessor
    _path: str

    # TODO add generic getattr that defers to attrs

    @property
    def data(self) -> h5py.Dataset | zarr.Array:
        try:
            return self._file[f"{self._path}/data"]
        except KeyError:
            if self._path not in self._file:
                raise lazynwb.exceptions.InternalPathError(
                    f"{self._path} not found in file"
                ) from None
            raise AttributeError(
                f"{self._path} has no data: use event timestamps alone"
            )

    @property
    def timestamps(self) -> h5py.Dataset | zarr.Array:
        try:
            return self._file[f"{self._path}/timestamps"]
        except KeyError:
            if self._path not in self._file:
                raise lazynwb.exceptions.InternalPathError(
                    f"{self._path} not found in file"
                ) from None
            rate = self.rate
            starting_time = self._starting_time
            if rate is None or starting_time is None:
                raise AssertionError(
                    f"Not enough information to calculate timestamps for {self._path}: need rate and starting_time"
                ) from None
            return (np.arange(len(self.data)) / rate) + starting_time

    @property
    def conversion(self) -> float | None:
        return self.data.attrs.get("conversion", None)

    @property
    def description(self) -> str | None:
        return self._file[f"{self._path}"].attrs.get("description", None)

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
            return self._file[f"{self._path}/starting_time"]
        except KeyError:
            if self._path not in self._file:
                raise lazynwb.exceptions.InternalPathError(
                    f"{self._path} not found in file"
                ) from None
            return None

    @property
    def starting_time(self) -> float:
        return self.timestamps[0]

    @property
    def timestamps_unit(self) -> str | None:
        with contextlib.suppress(KeyError):
            return self._file[self._path].attrs["timestamps_unit"]
        with contextlib.suppress(KeyError):
            return self._file[f"{self._path}/timestamps"].attrs.get("unit", None)
        with contextlib.suppress(KeyError):
            return self._file[f"{self._path}/starting_time"].attrs.get("unit", None)
        raise AttributeError(
            f"Cannot find timestamps unit for {self._path}: no timestamps or starting_time found"
        )

    @property
    def unit(self):
        return self.data.attrs.get("unit", None)


@typing.overload
def get_timeseries(
    nwb_path_or_accessor: npc_io.PathLike | lazynwb.file_io.FileAccessor,
    search_term: str | None = None,
    exact_path: bool = False,
    match_all: Literal[True] = True,
) -> dict[str, TimeSeries]: ...


@typing.overload
def get_timeseries(
    nwb_path_or_accessor: npc_io.PathLike | lazynwb.file_io.FileAccessor,
    search_term: str | None = None,
    exact_path: bool = False,
    match_all: Literal[False] = False,
) -> TimeSeries: ...


def get_timeseries(
    nwb_path_or_accessor: npc_io.PathLike | lazynwb.file_io.FileAccessor,
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
    nwb_path_or_accessor : PathLike or FileAccessor
        Path to an NWB file or a FileAccessor object. Can be an hdf5 or zarr NWB.
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
    if isinstance(nwb_path_or_accessor, lazynwb.file_io.FileAccessor):
        file = nwb_path_or_accessor
    else:
        file = lazynwb.file_io.FileAccessor(nwb_path_or_accessor)

    def _format(name: str) -> str:
        return name.removesuffix("/data").removesuffix("/timestamps")

    is_in_file = search_term in file
    if exact_path and not is_in_file:
        raise lazynwb.exceptions.InternalPathError(
            f"Exact path {search_term!r} not found in file {file._path.as_posix()}"
        )
    elif not match_all and search_term and is_in_file:
        return TimeSeries(_file=file, _path=_format(search_term))
    else:
        path_to_accessor = {
            _format(k): TimeSeries(_file=file, _path=_format(k))
            for k in lazynwb.utils.get_internal_file_paths(file._accessor)
            if k.split("/")[-1] in ("data", "timestamps")
            and (not search_term or search_term in k)
            # regular timeseries will be a dir with /data and optional /timestamps
            # eventseries will be a dir with /timestamps only
        }
        if match_all:
            return path_to_accessor
        if len(path_to_accessor) > 1:
            logger.warning(
                f"Found multiple timeseries matching {search_term!r}: {list(path_to_accessor.keys())} - returning first"
            )
        return next(iter(path_to_accessor.values()))


if __name__ == "__main__":
    from npc_io import testmod

    testmod()
