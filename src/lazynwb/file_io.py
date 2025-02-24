from __future__ import annotations

import contextlib
import enum
from typing import Any

import h5py
import npc_io
import remfile
import upath
import zarr


def open(
    path: npc_io.PathLike, use_remfile: bool = False, **fsspec_storage_options: Any
) -> h5py.File | zarr.Group:
    """
    Open a file that meets the NWB spec, minimizing the amount of data/metadata read.

    - file is opened in read-only mode
    - file is not closed when the function returns
    - currently supports NWB files saved in .hdf5 and .zarr format

    Examples:
        >>> nwb = open('https://dandiarchive.s3.amazonaws.com/blobs/f78/fe2/f78fe2a6-3dc9-4c12-a288-fbf31ce6fc1c')
        >>> nwb = open('https://dandiarchive.s3.amazonaws.com/blobs/f78/fe2/f78fe2a6-3dc9-4c12-a288-fbf31ce6fc1c', use_remfile=False)
        >>> nwb = open('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/39490bff-87c9-4ef2-b408-36334e748ac6/nwb/ecephys_620264_2022-08-02_15-39-59_experiment1_recording1.nwb')
    """
    path = npc_io.from_pathlike(path)
    path = upath.UPath(path, **fsspec_storage_options)

    # zarr ------------------------------------------------------------- #
    # there's no file-name convention for what is a zarr file, so we have to try opening it and see if it works
    # - zarr.open() is fast regardless of size
    with contextlib.suppress(Exception):
        return zarr.open(store=path, mode="r")

    # hdf5 ------------------------------------------------------------- #
    if not use_remfile:
        if path.protocol:
            # cloud path: open the file with fsspec and then pass the file handle to h5py
            file = path.open(mode="rb", cache_type="first")
        else:
            return h5py.File(path.as_posix(), mode="r")
    else:

        def s3_to_http(url: str) -> str:
            if url.startswith("s3://"):
                s3_path = url
                bucket = s3_path[5:].split("/")[0]
                object_name = "/".join(s3_path[5:].split("/")[1:])
                return f"https://s3.amazonaws.com/{bucket}/{object_name}"
            else:
                return url

        # but using remfile is slightly faster in practice, at least for the initial opening:
        file = remfile.File(url=s3_to_http(path.as_posix()))
    return h5py.File(file, mode="r")


class FileAccessor:
    """
    A wrapper that abstracts the storage backend (h5py.File, h5py.Group, or zarr.Group), forwarding
    all getattr/get item calls to the underlying object. Also stores the path to the file, and the
    type of backend as a string for convenience.

    - instantiate with a path to an NWB file or an open h5py.File, h5py.Group, or
    zarr.Group object
    - access components via the mapping interface
    - file accessor remains open in read-only mode unless used as a context manager

    Examples:
        >>> file = LazyFile('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/39490bff-87c9-4ef2-b408-36334e748ac6/nwb/ecephys_620264_2022-08-02_15-39-59_experiment1_recording1.nwb')
        >>> file.units
        <zarr.hierarchy.Group '/units' read-only>
        >>> file['units']
        <zarr.hierarchy.Group '/units' read-only>
        >>> file['units/spike_times']
        <zarr.core.Array '/units/spike_times' (18185563,) float64 read-only>
        >>> file['units/spike_times/index'][0]
        6966
        >>> 'spike_times' in file['units']
        True
        >>> next(iter(file))
        'acquisition'
        >>> next(iter(file['units']))
        'amplitude'
    """

    class HDMFBackend(enum.Enum):
        """Enum for file-type backend used by LazyFile instance (e.g. HDF5, ZARR)"""

        HDF5 = "hdf5"
        ZARR = "zarr"

    _path: upath.UPath
    _accessor: h5py.File | h5py.Group | zarr.Group
    _hdmf_backend: HDMFBackend
    """File-type backend used by this instance (e.g. HDF5, ZARR)"""

    def __init__(
        self,
        path: npc_io.PathLike,
        accessor: h5py.File | h5py.Group | zarr.Group | None = None,
        fsspec_storage_options: dict[str, Any] | None = None,
    ) -> None:
        self._path = npc_io.from_pathlike(path)
        if accessor is not None:
            self._accessor = accessor
        else:
            self._accessor = open(self._path, **(fsspec_storage_options or {}))
        self._hdmf_backend = self.get_hdmf_backend()

    def get_hdmf_backend(self) -> HDMFBackend:
        if isinstance(self._accessor, (h5py.File, h5py.Group)):
            return self.HDMFBackend.HDF5
        elif isinstance(self._accessor, zarr.Group):
            return self.HDMFBackend.ZARR
        raise NotImplementedError(f"Unknown backend for {self._accessor!r}")

    def __getattr__(self, name) -> Any:
        return getattr(self._accessor, name)

    def get(self, name: str, default: Any = None) -> Any:
        return self._accessor.get(name, default)
    
    def __getitem__(self, name) -> Any:
        return self._accessor[name]

    def __contains__(self, name) -> bool:
        return name in self._accessor

    def __iter__(self):
        return iter(self._accessor)

    def __repr__(self) -> str:
        if self._path is not None:
            return f"{self.__class__.__name__}({self._path.as_posix()!r})"
        return repr(self._accessor)

    def __enter__(self) -> FileAccessor:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        if self._path is not None:
            if isinstance(self._accessor, h5py.File):
                self._accessor.close()
            elif isinstance(self._accessor, zarr.Group):
                self._accessor.store.close()


def normalize_internal_file_path(path: str) -> str:
    """
    Normalize the internal file path for an NWB file.

    - add leading '/' if not present
    """
    return path if path.startswith("/") else f"/{path}"


if __name__ == "__main__":
    from npc_io import testmod

    testmod()
