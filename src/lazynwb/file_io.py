from __future__ import annotations

import contextlib
import enum
import logging
from typing import Any

import h5py
import npc_io
import pydantic
import remfile
import upath
import zarr


class FileIOConfig(pydantic.BaseModel):
    """
    Global configuration for file I/O behavior.
    """

    use_remfile: bool = True
    fsspec_storage_options: dict[str, Any] = {
        "anon": False,
    }


# singleton config
config = FileIOConfig()

# cache for FileAccessor instances by canonical path
_accessor_cache: dict[str, FileAccessor] = {}


def clear_cache() -> None:
    """
    Clear the FileAccessor caches.

    Users can call this to reset cached h5py and zarr accessors.
    """
    _accessor_cache.clear()


logger = logging.getLogger(__name__)


def _open_file(path: npc_io.PathLike) -> h5py.File | zarr.Group:
    """
    Internal: open raw HDF5 or Zarr backend using global config.
    """
    from contextlib import suppress

    p = npc_io.from_pathlike(path)
    u = upath.UPath(p, **config.fsspec_storage_options)
    key = u.as_posix()
    is_zarr = "zarr" in key
    if not is_zarr:
        with suppress(Exception):
            return _open_hdf5(u, use_remfile=config.use_remfile)
    with suppress(Exception):
        return zarr.open(store=u, mode="r")
    raise ValueError(f"Failed to open {u} as HDF5 or Zarr")


def open(path: npc_io.PathLike) -> FileAccessor:
    """
    Public API: get a cached FileAccessor for the given path.
    """
    return FileAccessor(path)


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
    _skip_init: bool
    _accessor: h5py.File | h5py.Group | zarr.Group
    _hdmf_backend: HDMFBackend
    """File-type backend used by this instance (e.g. HDF5, ZARR)"""

    def __new__(
        cls,
        path: npc_io.PathLike,
        accessor: h5py.File | h5py.Group | zarr.Group | None = None,
    ) -> FileAccessor:
        """
        Reuse existing FileAccessor for the same path if present in cache.
        """
        # allow passing through if already a FileAccessor
        if isinstance(path, FileAccessor):
            return path
        # normalize path and build key
        path_obj = npc_io.from_pathlike(path)
        u_path = upath.UPath(path_obj, **config.fsspec_storage_options)
        key = u_path.as_posix()
        # return cached if no custom accessor provided
        if accessor is None and key in _accessor_cache:
            instance = _accessor_cache[key]
            # mark to skip __init__ for cached instance
            instance._skip_init = True
            return instance
        # create new instance and cache
        instance = super().__new__(cls)
        _accessor_cache[key] = instance
        return instance

    def __init__(
        self,
        path: npc_io.PathLike,
        accessor: h5py.File | h5py.Group | zarr.Group | None = None,
    ) -> None:
        # skip init if returned from cache
        if getattr(self, "_skip_init", False):
            delattr(self, "_skip_init")
            return
        self._path = npc_io.from_pathlike(path)
        if accessor is not None:
            self._accessor = accessor
        else:
            self._accessor = _open_file(self._path)
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
