"""
Unified module for file handling operations:
- Opening files (local or remote)
- Path handling and extraction
- Automatic file opening and closing via decorators
- Thread pool support for parallel file operations
"""

from __future__ import annotations

import contextlib
import enum
import logging
from collections.abc import Collection
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
)

import h5py
import npc_io
import remfile
import upath
import zarr

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable)


# Create a Protocol for objects that support __getitem__
class HasGetItem(Protocol):
    def __getitem__(self, key: str) -> Any: ...


# Type hint for arguments that can be either a path or file object
FileArgument = Union[
    str, Path, upath.UPath, h5py.File, h5py.Group, zarr.Group, "FileAccessor"
]


#
# ---- File Opening ----
#


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
        >>> nwb = open('https://dandiarchive.s3.amazonaws.com/blobs/f78/fe2/f78fe2a6-3dc9-4c12-a288-fbf31ce6fc1c') # doctest: +SKIP
        >>> nwb = open('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/00865745-db58-495d-9c5e-e28424bb4b97/nwb/ecephys_721536_2024-05-16_12-32-31_experiment1_recording1.nwb') # doctest: +SKIP
    """
    path_str = str(npc_io.from_pathlike(path))

    # Check for S3 path early to avoid creating UPath unless needed
    if anon_s3 and path_str.startswith("s3://"):
        fsspec_storage_options.setdefault("anon", True)

    # Check for Zarr file by extension in the path string
    if "zarr" in path_str:
        is_zarr = True

    # For local paths, use direct approach without UPath
    if not _is_remote_path(path_str):
        if not is_zarr:
            try:
                return h5py.File(path_str, mode="r")
            except OSError:
                pass
        # Try zarr for local file
        try:
            return zarr.open(store=path_str, mode="r")
        except Exception:
            pass
    else:
        # For remote paths, we need UPath functionality
        upath_obj = upath.UPath(path_str, **fsspec_storage_options)

        # Try hdf5 first unless explicitly zarr
        if not is_zarr:
            with contextlib.suppress(Exception):
                return _open_hdf5(upath_obj, use_remfile=use_remfile)

        # Try zarr for remote file
        with contextlib.suppress(Exception):
            return zarr.open(store=upath_obj, mode="r")

    raise ValueError(
        f"Failed to open {path_str} as hdf5 or zarr. Is this the correct path to an NWB file?"
    )


def _is_remote_path(path_str: str) -> bool:
    """Determine if a path string points to a remote location."""
    return (
        path_str.startswith(("http://", "https://", "s3://", "gs://", "azure://"))
        or "://" in path_str
    )


def _s3_to_http(url: str) -> str:
    if url.startswith("s3://"):
        s3_path = url
        bucket = s3_path[5:].split("/")[0]
        object_name = "/".join(s3_path[5:].split("/")[1:])
        return f"https://{bucket}.s3.amazonaws.com/{object_name}"
    else:
        return url


def _open_file(path: Union[str, Path], open_func: Callable) -> Any:
    """Open a file and return the accessor."""
    return open_func(path)


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


#
# ---- File Access Wrappers ----
#


class HDMFBackend(enum.Enum):
    """Enum for file-type backend used by FileAccessor (e.g. HDF5, ZARR)"""

    HDF5 = "hdf5"
    ZARR = "zarr"


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
        >>> file = FileAccessor(
            "s3://aind-open-data/ecephys_625749_2022-08-03_15-15-06_nwb_2023-05-16_16-34-55/"
            "ecephys_625749_2022-08-03_15-15-06_nwb/"
            "ecephys_625749_2022-08-03_15-15-06_experiment1_recording1.nwb.zarr/"
        )
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

    _path: str | None  # Now using string instead of UPath
    _accessor: h5py.File | h5py.Group | zarr.Group
    _hdmf_backend: HDMFBackend
    """File-type backend used by this instance (e.g. HDF5, ZARR)"""

    def __init__(
        self,
        path_or_accessor: Union[npc_io.PathLike, h5py.File, h5py.Group, zarr.Group],
        fsspec_storage_options: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(path_or_accessor, (h5py.File, h5py.Group, zarr.Group)):
            self._path = None
            self._accessor = path_or_accessor
        else:
            # Store path as string
            self._path = str(npc_io.from_pathlike(path_or_accessor))
            self._accessor = open(self._path, **(fsspec_storage_options or {}))
        self._hdmf_backend = self.get_hdmf_backend()

    def __enter__(self) -> FileAccessor:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        if self._accessor is not None:
            if isinstance(self._accessor, h5py.File):
                self._accessor.close()
            elif isinstance(self._accessor, zarr.Group):
                if hasattr(self._accessor, "store") and hasattr(
                    self._accessor.store, "close"
                ):
                    self._accessor.store.close()


class FileAccessWrapper(Generic[T]):
    """
    A wrapper that unifies handling of file paths and data accessors.
    When given a file path, it will open and close the file automatically.
    When given an already open data accessor, it will leave it open.
    """

    def __init__(
        self,
        source: Union[str, Path, T],
        open_func: Optional[Callable[[Union[str, Path]], T]] = None,
    ):
        self.source = source
        self.open_func = open_func
        self._is_path = isinstance(source, (str, Path))
        self._accessor = None

    @contextmanager
    def access(self) -> ContextManager[T]:
        """Context manager to handle file access."""
        if self._is_path and self.open_func:
            # Open the file if source is a path
            accessor = self.open_func(self.source)
            try:
                yield accessor
            finally:
                # Close only if we opened it
                if hasattr(accessor, "close") and callable(accessor.close):
                    accessor.close()
        else:
            # Just use the provided accessor
            yield self.source


#
# ---- Path extraction utilities ----
#


def get_file_path(file_arg: FileArgument) -> upath.UPath:
    """
    Extract a UPath object from a file path string, Path object, or FileAccessor.

    This function doesn't open any files - it just extracts the path.
    Returns a UPath which supports cloud paths (s3://, etc.) in addition to local paths.

    Args:
        file_arg: Either a string path, Path/UPath object, FileAccessor, or a file-like object
                 that might have path information

    Returns:
        UPath object representing the file path

    Raises:
        ValueError: If the path can't be extracted from the provided argument
    """
    path_str = None

    if isinstance(file_arg, (str, Path, upath.UPath)):
        path_str = str(file_arg)
    elif isinstance(file_arg, FileAccessor) and file_arg._path is not None:
        path_str = file_arg._path.as_posix()
    elif hasattr(file_arg, "filename"):
        # h5py.File objects have a filename attribute
        path_str = str(file_arg.filename)
    elif hasattr(file_arg, "name"):
        # Some file objects have a name attribute
        path_str = str(file_arg.name)
    elif hasattr(file_arg, "path"):
        # Some file objects have a path attribute
        path_attr = file_arg.path
        if isinstance(path_attr, (str, Path, upath.UPath)):
            path_str = str(path_attr)

    if path_str is not None:
        return upath.UPath(path_str)

    raise ValueError(
        f"Unable to extract file path from object of type {type(file_arg)}"
    )


def is_same_file(file1: FileArgument, file2: FileArgument) -> bool:
    """
    Check if two file arguments refer to the same file.

    Works with both local paths and cloud paths (s3://, etc.).

    Args:
        file1: First file (path, Path/UPath object, or file-like object)
        file2: Second file (path, Path/UPath object, or file-like object)

    Returns:
        True if both arguments refer to the same file, False otherwise

    Raises:
        ValueError: If a path can't be extracted from either argument
    """
    try:
        path1 = get_file_path(file1)
        path2 = get_file_path(file2)

        # For local paths, resolve to handle symlinks, relative paths, etc.
        if not path1.protocol and not path2.protocol:
            return Path(path1).resolve() == Path(path2).resolve()

        # For cloud paths or mixed paths, compare as strings
        return path1.as_posix() == path2.as_posix()
    except ValueError as e:
        raise ValueError(f"Failed to compare files: {e}")


#
# ---- File Access Decorators ----
#


def auto_file_close(
    open_func: Optional[Callable[[Union[str, Path]], Any]] = None,
    use_thread_pool: bool = False,
    max_workers: Optional[int] = None,
):
    """
    Decorator that automatically ensures files are closed after use.

    Behavior:
    - If a path (string/Path) is passed as an argument, it will be opened and automatically closed
    - If an already open file object is passed, it will be used as-is and NOT closed
    - Works with individual file arguments and collections (lists/tuples) of file paths

    When given file paths as arguments, this decorator will:
    1. Open the files with the specified opener
    2. Pass the open file handles to the decorated function
    3. Automatically close the files when the function returns

    Usage example:

    @auto_file_close()  # Uses default opener from file_handlers
    def my_func(file1, file2, other_arg=None):
        # file1 and file2 will be automatically opened and then closed after use
        # if they are paths. If already open, they'll be left untouched.
        ...

    @auto_file_close(h5py.File)  # Specify a specific opener
    def another_func(file1, file2, other_arg=None):
        # file1 and file2 will be automatically opened with h5py.File and closed
        ...

    @auto_file_close(use_thread_pool=True)  # Use parallel opening
    def process_many_files(files, output_file):
        # Files in the list will be opened in parallel and all closed afterward
        ...

    Args:
        open_func: Function to use for opening files if a path is provided.
                Defaults to the open function from this module
        use_thread_pool: Whether to use a thread pool for opening multiple files in parallel
        max_workers: Maximum number of worker threads to use. None means the default
                    ThreadPoolExecutor behavior

    Returns:
        Decorator function
    """
    # Use default open() function if open_func is not provided
    actual_open_func = open_func if open_func is not None else open

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature to map positional args to names
            sig = signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Use ExitStack to manage multiple context managers
            with ExitStack() as stack:
                # Create copies of args and kwargs for modification
                new_kwargs: Dict[str, Any] = {}
                modified = False

                # Keep track of file paths for parallel opening
                file_paths_to_open: List[Dict] = []

                # Process all arguments
                for param_name, value in bound_args.arguments.items():
                    # Handle single file path
                    if isinstance(value, (str, Path)):
                        file_paths_to_open.append(
                            {
                                "param_name": param_name,
                                "path": value,
                                "is_collection": False,
                            }
                        )
                        modified = True

                    # Handle list/collection of file paths
                    elif (
                        isinstance(value, Collection)
                        and not isinstance(value, (str, Path))
                        and any(isinstance(src, (str, Path)) for src in value)
                    ):
                        # Store the collection items for potential parallel opening
                        items_to_open = []
                        for i, src in enumerate(value):
                            if isinstance(src, (str, Path)):
                                items_to_open.append({"index": i, "path": src})

                        if items_to_open:
                            file_paths_to_open.append(
                                {
                                    "param_name": param_name,
                                    "items": items_to_open,
                                    "is_collection": True,
                                    "original_value": value,
                                }
                            )
                            modified = True

                    # Keep original value for non-file parameters
                    if not modified or param_name not in [
                        item["param_name"] for item in file_paths_to_open
                    ]:
                        new_kwargs[param_name] = value

                # Open files, potentially in parallel
                if file_paths_to_open:
                    if use_thread_pool and (
                        len(file_paths_to_open) > 1
                        or any(item.get("is_collection") for item in file_paths_to_open)
                    ):
                        # Open files in parallel using thread pool
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            # Process single file paths
                            single_files = [
                                item
                                for item in file_paths_to_open
                                if not item["is_collection"]
                            ]
                            if single_files:
                                futures = {
                                    executor.submit(
                                        _open_file, item["path"], actual_open_func
                                    ): item["param_name"]
                                    for item in single_files
                                }

                                for future in futures:
                                    param_name = futures[future]
                                    accessor = future.result()
                                    # Wrap in FileAccessor for consistent behavior
                                    new_kwargs[param_name] = stack.enter_context(
                                        FileAccessor(accessor)
                                    )

                            # Process collections of file paths
                            collection_items = [
                                item
                                for item in file_paths_to_open
                                if item["is_collection"]
                            ]
                            for coll in collection_items:
                                param_name = coll["param_name"]
                                original = list(
                                    coll["original_value"]
                                )  # Make a copy of the original collection
                                items_to_open = coll["items"]

                                # Submit jobs to thread pool
                                item_futures = {
                                    executor.submit(
                                        _open_file, item["path"], actual_open_func
                                    ): item["index"]
                                    for item in items_to_open
                                }

                                # Process results as they complete
                                for future in item_futures:
                                    index = item_futures[future]
                                    accessor = future.result()
                                    # Wrap in FileAccessor for consistent behavior
                                    original[index] = stack.enter_context(
                                        FileAccessor(accessor)
                                    )

                                new_kwargs[param_name] = original
                    else:
                        # Process files sequentially (simpler code path)
                        for item in file_paths_to_open:
                            if not item["is_collection"]:
                                # Single file path
                                accessor = actual_open_func(item["path"])
                                # Wrap in FileAccessor for consistent behavior
                                new_kwargs[item["param_name"]] = stack.enter_context(
                                    FileAccessor(accessor)
                                )
                            else:
                                # Collection of file paths
                                accessors = list(item["original_value"])  # Make a copy
                                param_name = item["param_name"]
                                for subitem in item["items"]:
                                    accessor = actual_open_func(subitem["path"])
                                    # Wrap in FileAccessor for consistent behavior
                                    accessors[subitem["index"]] = stack.enter_context(
                                        FileAccessor(accessor)
                                    )
                                new_kwargs[param_name] = accessors

                if modified:
                    return func(**new_kwargs)
                else:
                    # No file paths found, use original arguments
                    return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def auto_file_open(
    open_func: Optional[Callable[[Union[str, Path]], T]] = None,
    use_thread_pool: bool = False,
    max_workers: Optional[int] = None,
):
    """
    Decorator that automatically handles file access for all arguments that are file paths.

    This decorator:
    1. Opens file paths with the specified opener
    2. Passes the open file handles to the decorated function
    3. Automatically closes the files when the function completes

    Usage example:

    @auto_file_open()  # Uses default opener
    def my_func(file1, file2, other_arg=None):
        # file1 and file2 will be automatically opened if they're paths
        ...

    @auto_file_open(h5py.File)  # Specify a specific opener
    def another_func(file1, file2, other_arg=None):
        # file1 and file2 will be automatically opened with h5py.File
        ...

    @auto_file_open(use_thread_pool=True)  # Use default opener with thread pool
    def process_many_files(files, output_file):
        # Files in the list will be opened in parallel
        ...

    Args:
        open_func: Function to use for opening files if a path is provided.
                  Defaults to the open function from this module
        use_thread_pool: Whether to use a thread pool for opening multiple files in parallel
        max_workers: Maximum number of worker threads to use. None means the default
                    ThreadPoolExecutor behavior

    Returns:
        Decorator function
    """
    # Use default open() function if open_func is not provided
    actual_open_func = open_func if open_func is not None else open

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature to map positional args to names
            sig = signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Use ExitStack to manage multiple context managers
            with ExitStack() as stack:
                # Create copies of args and kwargs for modification
                new_kwargs: Dict[str, Any] = {}
                modified = False

                # Keep track of file paths for parallel opening
                file_paths_to_open: List[Dict] = []

                # Process all arguments
                for param_name, value in bound_args.arguments.items():
                    # Handle single file path
                    if isinstance(value, (str, Path)):
                        file_paths_to_open.append(
                            {
                                "param_name": param_name,
                                "path": value,
                                "is_collection": False,
                            }
                        )
                        modified = True
                    # Handle already open file objects - pass through as is
                    elif isinstance(
                        value, (h5py.File, h5py.Group, zarr.Group, FileAccessor)
                    ):
                        new_kwargs[param_name] = value
                        modified = True
                    # Handle list/collection of file paths
                    elif (
                        isinstance(value, Collection)
                        and not isinstance(value, (str, Path))
                        and any(isinstance(src, (str, Path)) for src in value)
                    ):
                        # Store the collection items for potential parallel opening
                        items_to_open = []
                        for i, src in enumerate(value):
                            if isinstance(src, (str, Path)):
                                items_to_open.append({"index": i, "path": src})

                        if items_to_open:
                            file_paths_to_open.append(
                                {
                                    "param_name": param_name,
                                    "items": items_to_open,
                                    "is_collection": True,
                                    "original_value": value,
                                }
                            )
                            modified = True

                    # Keep original value for non-file parameters
                    if not modified or (
                        param_name
                        not in [item["param_name"] for item in file_paths_to_open]
                        and param_name not in new_kwargs
                    ):
                        new_kwargs[param_name] = value

                # Open files, potentially in parallel
                if file_paths_to_open:
                    if use_thread_pool and len(file_paths_to_open) > 1:
                        # Open files in parallel using thread pool
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            # Process single file paths
                            single_files = [
                                item
                                for item in file_paths_to_open
                                if not item["is_collection"]
                            ]
                            if single_files:
                                futures = {
                                    executor.submit(
                                        _open_file, item["path"], actual_open_func
                                    ): item["param_name"]
                                    for item in single_files
                                }

                                for future in futures:
                                    param_name = futures[future]
                                    accessor = future.result()
                                    wrapper = FileAccessWrapper(accessor)
                                    new_kwargs[param_name] = stack.enter_context(
                                        wrapper.access()
                                    )

                            # Process collections of file paths
                            collection_items = [
                                item
                                for item in file_paths_to_open
                                if item["is_collection"]
                            ]
                            for coll in collection_items:
                                param_name = coll["param_name"]
                                original = list(
                                    coll["original_value"]
                                )  # Make a copy of the original collection
                                items_to_open = coll["items"]

                                # Submit jobs to thread pool
                                item_futures = {
                                    executor.submit(
                                        _open_file, item["path"], actual_open_func
                                    ): item["index"]
                                    for item in items_to_open
                                }

                                # Process results as they complete
                                for future in item_futures:
                                    index = item_futures[future]
                                    accessor = future.result()
                                    wrapper = FileAccessWrapper(accessor)
                                    original[index] = stack.enter_context(
                                        wrapper.access()
                                    )

                                new_kwargs[param_name] = original
                    else:
                        # Process files sequentially (simpler code path)
                        for item in file_paths_to_open:
                            if not item["is_collection"]:
                                # Single file path
                                wrapper = FileAccessWrapper(
                                    item["path"], actual_open_func
                                )
                                new_kwargs[item["param_name"]] = stack.enter_context(
                                    wrapper.access()
                                )
                            else:
                                # Collection of file paths
                                accessors = list(item["original_value"])  # Make a copy
                                for subitem in item["items"]:
                                    wrapper = FileAccessWrapper(
                                        subitem["path"], actual_open_func
                                    )
                                    accessors[subitem["index"]] = stack.enter_context(
                                        wrapper.access()
                                    )
                                new_kwargs[param_name] = accessors

                if modified:
                    return func(**new_kwargs)
                else:
                    # No file paths found, use original arguments
                    return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


if __name__ == "__main__":
    from npc_io import testmod

    testmod()
