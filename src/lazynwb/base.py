from __future__ import annotations

import contextlib
from typing import Any

import h5py
import npc_io
import polars as pl
import upath
import zarr

import lazynwb.file_io


class LazyNWB:
    """

    Examples:
        >>> nwb = LazyNWB('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/39490bff-87c9-4ef2-b408-36334e748ac6/nwb/ecephys_620264_2022-08-02_15-39-59_experiment1_recording1.nwb')
        >>> nwb.units
        <zarr.hierarchy.Group '/units' read-only>
        >>> nwb['units']
        <zarr.hierarchy.Group '/units' read-only>
        >>> nwb.units.spike_times
        <zarr.core.Array '/units/spike_times' (18185563,) float64 read-only>
        >>> nwb.units.spike_times_index[0]
        6966
        >>> nwb.units.id[0]
        0
        >>> 'spike_times' in nwb.units
        True
        >>> next(iter(nwb))
        'acquisition'
        >>> next(iter(nwb.units))
        'amplitude'
    """

    _path: upath.UPath | None
    _nwb: h5py.File | h5py.Group | zarr.Group

    def __init__(
        self,
        path_or_data: npc_io.PathLike | h5py.File | h5py.Group | zarr.Group,
    ) -> None:
        if isinstance(path_or_data, (h5py.File, h5py.Group, zarr.Group)):
            self._path = None
            self._nwb = path_or_data
        else:
            self._path = npc_io.from_pathlike(path_or_data)
            self._nwb = lazynwb.file_io.open(self._path)

    def __getattr__(self, name) -> Any:
        with contextlib.suppress(AttributeError):
            return getattr(self._nwb, name)
        with contextlib.suppress(KeyError):
            component = self._nwb[name]
            if isinstance(component, (h5py.Group, zarr.Group)):
                return LazyNWB(component)
            return component
        raise AttributeError(f"No attribute named {name!r} in NWB file")

    def __getitem__(self, name) -> Any:
        return self._nwb[name]

    def __contains__(self, name) -> bool:
        return name in self._nwb

    def __iter__(self):
        return iter(self._nwb)

    def __repr__(self) -> str:
        if self._path is not None:
            return f"{self.__class__.__name__}({self._path.as_posix()!r})"
        return repr(self._nwb)


if __name__ == "__main__":
    from npc_io import testmod

    testmod()
