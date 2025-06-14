from __future__ import annotations

import contextlib
import enum
import logging
import threading
from collections.abc import Iterable
from typing import Any

import h5py
import npc_io
import pydantic
import remfile
import upath
import zarr

logger = logging.getLogger(__name__)


class FileIOConfig(pydantic.BaseModel):
    """
    Global configuration for file I/O behavior.
    """

    use_remfile: bool = True
    fsspec_storage_options: dict[str, Any] = {
        "anon": False,
    }
    disable_cache: bool = False


# singleton config
config = FileIOConfig()

# cache for FileAccessor instances by canonical path
_accessor_cache: dict[str, FileAccessor] = {}
_cache_lock = threading.RLock()  # RLock allows same thread to acquire multiple times


def clear_cache() -> None:
    """
    Clear the FileAccessor caches.

    Users can call this to reset cached h5py and zarr accessors.
    """
    with _cache_lock:
        FileAccessor._clear_cache()


def _get_accessor(path: npc_io.PathLike) -> FileAccessor:
    if isinstance(path, Iterable) and not isinstance(path, str):
        raise ValueError(f"Expected a single path, but received an iterable: {path!r}")
    return FileAccessor(path)


def _get_accessors(
    paths: npc_io.PathLike | Iterable[npc_io.PathLike],
) -> tuple[FileAccessor, ...]:
    if not isinstance(paths, Iterable) or isinstance(paths, str):
        paths = [paths]  # ensure we have an iterable of paths
    return tuple(FileAccessor(path) for path in paths)


def _s3_to_http(url: str) -> str:
    if url.startswith("s3://"):
        s3_path = url
        bucket = s3_path[5:].split("/")[0]
        object_name = "/".join(s3_path[5:].split("/")[1:])
        return f"https://{bucket}.s3.amazonaws.com/{object_name}"
    else:
        return url


def _open_file(path: npc_io.PathLike) -> h5py.File | zarr.Group:
    """
    open raw HDF5 or Zarr backend using global config
    """
    p = npc_io.from_pathlike(path)
    u = upath.UPath(p, **config.fsspec_storage_options)
    key = u.as_posix()
    is_zarr = "zarr" in key
    if not is_zarr:
        with contextlib.suppress(Exception):
            return _open_hdf5(u, use_remfile=config.use_remfile)
    with contextlib.suppress(Exception):
        return zarr.open(store=u, mode="r")
    raise ValueError(f"Failed to open {u} as HDF5 or Zarr")


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
    ) -> FileAccessor:
        """
        Reuse existing FileAccessor for the same path if present in cache.
        """
        # be careful not to access attributes created in __init__ before they exist

        # allow passing through if already a FileAccessor
        if isinstance(path, FileAccessor):
            logger.debug(
                "path input is already a FileAccessor instance: returning as-is"
            )
            return path

        # skip caching if disabled
        if config.disable_cache:
            return super().__new__(cls)

        # normalize path to get cache key
        # try lightweight version first:
        if isinstance(path, str):
            cache_key = path.replace("\\", "/")
        elif hasattr(path, "as_posix"):
            cache_key = path.as_posix()
        else:
            cache_key = npc_io.from_pathlike(
                path, **config.fsspec_storage_options
            ).as_posix()

        with _cache_lock:
            # return cached instance if it exists and is open
            if cache_key in _accessor_cache:
                instance = _accessor_cache[cache_key]

                if "_accessor" not in instance.__dict__:
                    logger.debug(
                        f"cached instance for {cache_key} is not properly initialized, removing from cache"
                    )
                    del _accessor_cache[cache_key]

                if instance._hdmf_backend == cls.HDMFBackend.ZARR:
                    if (
                        _is_open := getattr(instance._accessor.store, "_is_open", None)
                    ) is not None:
                        # zarr v3
                        is_readable = _is_open
                    else:
                        # zarr v2
                        is_readable = instance._accessor.store.is_readable()

                elif instance._hdmf_backend == cls.HDMFBackend.HDF5:
                    is_readable = bool(instance._accessor)
                if is_readable:
                    logger.debug(f"returning cached instance for {cache_key}")
                    # mark to skip __init__ for cached instance
                    instance._skip_init = True
                else:
                    instance._skip_init = False
                    logger.debug(
                        f"cached instance for {cache_key} is stale, will recreate"
                    )
                return instance

            # create new instance and cache
            logger.debug(f"creating new instance for {cache_key}")
            instance = super().__new__(cls)
            _accessor_cache[cache_key] = instance
            return instance

    def __init__(
        self,
        path: npc_io.PathLike,
    ) -> None:
        # skip init if returned from cache
        if self.__dict__.get(
            "_skip_init"
        ):  # don't check attr directly: __getattr__ is overloaded
            logger.debug("skipping init for cached instance")
            return None
        self._path = npc_io.from_pathlike(path)
        logger.debug(f"opening file {self._path}")
        self._accessor = _open_file(self._path)
        self._hdmf_backend = self.get_hdmf_backend()
        logger.debug(f"initialized with backend {self._hdmf_backend}")

    def get_hdmf_backend(self) -> HDMFBackend:
        if isinstance(self._accessor, (h5py.File, h5py.Group)):
            return self.HDMFBackend.HDF5
        elif isinstance(self._accessor, zarr.Group):
            return self.HDMFBackend.ZARR
        raise NotImplementedError(f"Unknown backend for {self._accessor!r}")

    @classmethod
    def _clear_cache(cls) -> None:
        """
        Clear the FileAccessor cache.
        This is useful to reset the state of cached accessors.
        """
        global _accessor_cache
        logger.debug("Closing all accessors in FileAccessor cache")
        for accessor in _accessor_cache.values():
            if accessor._hdmf_backend == cls.HDMFBackend.HDF5:
                accessor._accessor.close()
            elif accessor._hdmf_backend == cls.HDMFBackend.ZARR:
                accessor._accessor.store.close()
        logger.debug("Clearing FileAccessor cache")
        _accessor_cache.clear()

    def __getstate__(self) -> dict[str, Any]:
        """
        Custom pickle state for multiprocessing compatibility.
        Only serialize the path, not the accessor or cache references.
        """
        return {
            "path": self._path,
            "hdmf_backend": self._hdmf_backend,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Custom unpickle state for multiprocessing compatibility.
        Recreate the accessor from the path in the new process.
        """
        self._path = state["path"]
        self._hdmf_backend = state["hdmf_backend"]

        # Check if already cached in new process
        u_path = upath.UPath(self._path, **config.fsspec_storage_options)
        key = u_path.as_posix()

        if key in _accessor_cache:
            # Reuse existing accessor from cache
            self._accessor = _accessor_cache[key]._accessor
        else:
            # Create new accessor and cache this instance
            self._accessor = _open_file(self._path)
            _accessor_cache[key] = self

    def __getattr__(self, name) -> Any:
        if name == "_accessor":
            raise AttributeError(
                f"{self.__class__.__name__} has no attribute '_accessor'"
            )
            # this is correct behavior, as _accessor should be provided by __getattribute__ before __getattr__ is called
            # - if we reach this point it means _accessor has not been set yet
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
            if self._hdmf_backend == self.HDMFBackend.HDF5:
                self._accessor.close()
            elif self._hdmf_backend == self.HDMFBackend.ZARR:
                self._accessor.store.close()


if __name__ == "__main__":
    from npc_io import testmod

    testmod()
