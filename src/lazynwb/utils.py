from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing
import os

import h5py
import npc_io
import zarr

import lazynwb.file_io

logger = logging.getLogger(__name__)

thread_pool_executor: concurrent.futures.ThreadPoolExecutor | None = None
process_pool_executor: concurrent.futures.ProcessPoolExecutor | None = None


def get_threadpool_executor() -> concurrent.futures.ThreadPoolExecutor:
    global thread_pool_executor
    if thread_pool_executor is None:
        thread_pool_executor = concurrent.futures.ThreadPoolExecutor()
    return thread_pool_executor


def get_processpool_executor() -> concurrent.futures.ProcessPoolExecutor:
    global process_pool_executor
    if process_pool_executor is None:
        process_pool_executor = concurrent.futures.ProcessPoolExecutor(
            mp_context=(
                multiprocessing.get_context("spawn") if os.name == "posix" else None
            )
        )
    return process_pool_executor


def normalize_internal_file_path(path: str) -> str:
    """
    Normalize the internal file path for an NWB file.

    - add leading '/' if not present
    """
    return path if path.startswith("/") else f"/{path}"


def get_nwb_file_structure(
    nwb_path: npc_io.PathLike,
    exclude_specifications: bool = True,
    exclude_table_columns: bool = True,
    exclude_metadata: bool = True,
) -> dict[str, h5py.Dataset | zarr.Array]:
    """
    Get a summary of the internal structure of an NWB file.

    This function provides a quick overview of the contents of a single NWB file,
    showing all internal paths and their corresponding datasets or groups.

    Parameters
    ----------
    nwb_path : PathLike
        Path to the NWB file (local file path, S3 URL, or other supported path types).
    exclude_specifications : bool, default True
        Whether to exclude specification-related paths from the output.
    exclude_table_columns : bool, default True
        Whether to exclude individual table columns from the output.
    exclude_metadata : bool, default True
        Whether to exclude top-level metadata paths from the output.

    Returns
    -------
    dict[str, h5py.Dataset | zarr.Array]
        Dictionary mapping internal file paths to their corresponding datasets or arrays.
        Keys are internal paths (e.g., '/units/spike_times'), values are the actual
        dataset/array objects that can be inspected for shape, dtype, etc.

    Examples
    --------
    >>> import lazynwb
    >>> structure = lazynwb.get_nwb_file_structure("path/to/file.nwb")
    >>> for path, dataset in structure.items():
    ...     print(f"{path}: {dataset}")
    /acquisition/lick_sensor_events/data: <HDF5 dataset "data": shape (2734,), type "<f8">
    /intervals/trials: <HDF5 group "/intervals/trials" (48 members)>
    /units/spike_times: <HDF5 dataset "/units/spike_times" ...>
    """
    file_accessor = lazynwb.file_io._get_accessor(nwb_path)
    return _traverse_internal_paths(
        file_accessor._accessor,
        exclude_specifications=exclude_specifications,
        exclude_table_columns=exclude_table_columns,
        exclude_metadata=exclude_metadata,
    )


def _traverse_internal_paths(
    group: h5py.Group | zarr.Group | zarr.Array,
    exclude_specifications: bool = True,
    exclude_table_columns: bool = True,
    exclude_metadata: bool = True,
) -> dict[str, h5py.Dataset | zarr.Array]:
    results: dict[str, h5py.Dataset | zarr.Array] = {}
    if exclude_specifications and group.name == "/specifications":
        return results
    if not hasattr(group, "keys") or (
        exclude_table_columns and "colnames" in getattr(group, "attrs", {})
    ):
        if exclude_metadata and (
            group.name.count("/") == 1 or group.name.startswith("/general")
        ):
            return {}
        else:
            results[group.name] = group
            return results
    for subpath in group.keys():
        try:
            results = {
                **results,
                **_traverse_internal_paths(
                    group[subpath],
                    exclude_specifications=exclude_specifications,
                    exclude_table_columns=exclude_table_columns,
                    exclude_metadata=exclude_metadata,
                ),
            }
        except (AttributeError, IndexError, TypeError):
            results[group.name] = group
    return results


if __name__ == "__main__":
    from npc_io import testmod

    testmod()
