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


def get_internal_paths(
    nwb_path: npc_io.PathLike,
    include_arrays: bool = True,
    include_table_columns: bool = False,
    include_metadata: bool = False,
    include_specifications: bool = False,
    parents: bool = False,
) -> dict[str, h5py.Dataset | zarr.Array]:
    """
    Traverse the internal structure of an NWB file and return a mapping of paths to data accessors.

    Parameters
    ----------
    nwb_path : PathLike
        Path to the NWB file (local file path, S3 URL, or other supported path type).
    include_table_columns : bool, default False
        Include individual table columns (which are actually arrays) in the output.
    include_arrays : bool, default False
        Include arrays like 'data' or 'timestamps' in a TimeSeries object.
    include_metadata : bool, default False
        Include top-level metadata paths (like /session_start_time or /general/subject) in the output.
    include_specifications : bool, default False
        Include NWB schema-related paths in the output - rarely needed.
    parents : bool, default False
        If True, include paths that have children paths in the output, even if it is not a table
        column or array itself, e.g. the path to a table (parent) as well as its columns (children).

    Returns
    -------
    dict[str, h5py.Dataset | zarr.Array]
        Dictionary mapping internal file paths to their corresponding datasets or arrays.
        Keys are internal paths (e.g., '/units/spike_times'), values are the actual
        dataset/array objects that can be inspected for shape, dtype, etc.
    """
    file_accessor = lazynwb.file_io._get_accessor(nwb_path)
    paths_to_accessors = _traverse_internal_paths(
        file_accessor._accessor,
        include_table_columns=include_table_columns,
        include_arrays=include_arrays,
        include_metadata=include_metadata,
        include_specifications=include_specifications,
    )
    if not parents:
        paths = list(paths_to_accessors.keys())
        # remove paths that have children
        for path in paths:
            if any(p.startswith(path + "/") for p in paths):
                del paths_to_accessors[path]
    return paths_to_accessors


def _traverse_internal_paths(
    group: h5py.Group | zarr.Group | zarr.Array,
    include_arrays: bool = False,
    include_table_columns: bool = False,
    include_metadata: bool = False,
    include_specifications: bool = False,
) -> dict[str, h5py.Dataset | zarr.Array]:
    """https://nwb-overview.readthedocs.io/en/latest/intro_to_nwb/2_file_structure.html"""
    results: dict[str, h5py.Dataset | zarr.Array] = {}
    if "/specifications" in group.name:
        if include_specifications:
            results[group.name] = group
        else:
            return {}
    is_dataset = hasattr(group, "keys") and len(group) > 0
    shape = getattr(group, "shape", None)
    is_scalar = shape == () or shape == (1,)
    is_array = shape is not None and not is_scalar
    if is_scalar:
        return {}
    attrs = dict(getattr(group, "attrs", {}))
    neurodata_type = attrs.get("neurodata_type", None)
    is_neurodata = neurodata_type is not None
    is_table = "colnames" in attrs
    is_metadata = is_scalar or group.name.startswith(
        "/general"
    )  # other metadata like /general/lab
    if is_metadata and not include_metadata:
        return {}
    elif is_metadata and include_metadata:
        results[group.name] = group
    elif is_neurodata and neurodata_type not in (
        "/",
        "NWBFile",
        "ProcessingModule",
    ):
        results[group.name] = group
    elif is_array and include_arrays:  # has no neurodata_type
        results[group.name] = group
    else:
        pass
    if is_table and not include_table_columns:
        return results
    if not is_dataset:
        return results
    for subpath in group.keys():
        try:
            results = {
                **results,
                **_traverse_internal_paths(
                    group[subpath],
                    include_table_columns=include_table_columns,
                    include_arrays=include_arrays,
                    include_metadata=include_metadata,
                    include_specifications=include_specifications,
                ),
            }
        except (AttributeError, IndexError, TypeError):
            results[group.name] = group
    return results


if __name__ == "__main__":
    from npc_io import testmod

    testmod()
