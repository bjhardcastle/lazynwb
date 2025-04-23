from __future__ import annotations

import contextlib
import enum
import logging
from typing import Any

import h5py
import npc_io
import remfile
import upath
import zarr

logger = logging.getLogger(__name__)


def open(
    path: npc_io.PathLike,
    is_zarr: bool = False,
    use_remfile: bool = True,
    anon_s3: bool = False,
    **fsspec_storage_options: Any,
) -> h5py.File | zarr.Group:
    """
    Open a file that meets the NWB spec, minimizing the amount of data/metadata read.

    - file is opened in read-only mode
    - file is not closed when the function returns
    - currently supports NWB files saved in .hdf5 and .zarr format

    Examples:
        >>> nwb = open('https://dandiarchive.s3.amazonaws.com/blobs/f78/fe2/f78fe2a6-3dc9-4c12-a288-fbf31ce6fc1c')
        >>> nwb = open('https://dandiarchive.s3.amazonaws.com/blobs/f78/fe2/f78fe2a6-3dc9-4c12-a288-fbf31ce6fc1c', use_remfile=False)
        >>> nwb = open('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/00865745-db58-495d-9c5e-e28424bb4b97/nwb/ecephys_721536_2024-05-16_12-32-31_experiment1_recording1.nwb')
    """
    path = npc_io.from_pathlike(path)
    if anon_s3 and path.protocol == "s3":
        fsspec_storage_options.setdefault("anon", True)
    path = upath.UPath(path, **fsspec_storage_options)

    if "zarr" in path.as_posix():
        is_zarr = True

    # zarr ------------------------------------------------------------- #
    # there's no file-name convention for what is a zarr file, so we have to try opening it and see if it works
    # - zarr.open() is fast regardless of size
    if not is_zarr:
        with contextlib.suppress(Exception):
            return _open_hdf5(path, use_remfile=use_remfile)

    with contextlib.suppress(Exception):
        return zarr.open(store=path, mode="r")
    raise ValueError(
        f"Failed to open {path} as hdf5 or zarr. Is this the correct path to an NWB file?"
    )


def _s3_to_http(url: str) -> str:
    if url.startswith("s3://"):
        s3_path = url
        bucket = s3_path[5:].split("/")[0]
        object_name = "/".join(s3_path[5:].split("/")[1:])
        return f"https://{bucket}.s3.amazonaws.com/{object_name}"
    else:
        return url


def _open_hdf5(path: upath.UPath, use_remfile: bool = True) -> h5py.File:
    if not path.protocol:
        # local path: open the file with h5py directly
        return h5py.File(path.as_posix(), mode="r")
    file = None
    if use_remfile:
        try:
            file = remfile.File(url=_s3_to_http(path.as_posix()))
        except Exception as exc:  # remfile raises base Exception for many reasons
            logger.warning(
                f"remfile failed to open {path}, falling back to fsspec: {exc!r}"
            )
    if file is None:
        file = path.open(mode="rb", cache_type="first")
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
        >>> file = LazyFile('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/00865745-db58-495d-9c5e-e28424bb4b97/nwb/ecephys_721536_2024-05-16_12-32-31_experiment1_recording1.nwb')
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


if __name__ == "__main__":
    from npc_io import testmod

    testmod()
