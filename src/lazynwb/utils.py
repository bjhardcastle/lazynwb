from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing
import os

import h5py
import zarr

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


def get_internal_file_paths(
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
                **get_internal_file_paths(
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
