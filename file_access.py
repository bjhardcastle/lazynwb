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
    TypeVar,
    Union,
    cast,
)

from src.lazynwb.file_io import open as default_opener

T = TypeVar("T")
F = TypeVar("F", bound=Callable)


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


def _open_file(path: Union[str, Path], open_func: Callable) -> Any:
    """Open a file and return the accessor."""
    return open_func(path)


def auto_file_open(
    open_func: Optional[Callable[[Union[str, Path]], T]] = None,
    use_thread_pool: bool = False,
    max_workers: Optional[int] = None,
):
    """
    Decorator that automatically handles file access for all arguments that are file paths.

    Usage example:

    @auto_file_open()  # Uses default opener from file_io.py
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
                  Defaults to the open function from file_io.py
        use_thread_pool: Whether to use a thread pool for opening multiple files in parallel
        max_workers: Maximum number of worker threads to use. None means the default
                    ThreadPoolExecutor behavior (typically min(32, os.cpu_count() + 4))

    Returns:
        Decorator function
    """
    # Use default_opener if open_func is not provided
    actual_open_func = open_func if open_func is not None else default_opener

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
