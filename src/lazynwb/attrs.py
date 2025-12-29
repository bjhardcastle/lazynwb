"""
Functions for retrieving and consolidating attributes from NWB files.

Attributes in HDF5/Zarr files provide metadata about datasets and groups, including
user-defined descriptions, neurodata types, units information, and other useful metadata.
These functions expose this metadata in a structured way for summarizing NWB file contents.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from typing import Any

import h5py
import zarr

import lazynwb.file_io
import lazynwb.types_
import lazynwb.utils

logger = logging.getLogger(__name__)


# Per-FileAccessor attrs caches: {FileAccessor._path: {internal_path: {attr_name: {...}}}}
_attrs_cache: dict[str, dict[str, dict[str, dict[str, Any]] | None]] = {}
_attrs_cache_lock = threading.RLock()


def _get_cache_key(file_accessor: lazynwb.file_io.FileAccessor) -> str:
    """Get normalized cache key for a FileAccessor."""
    return file_accessor._path.as_posix()


def _get_attrs_from_accessor(
    obj: h5py.Dataset | h5py.Group | zarr.Array | zarr.Group,
) -> dict[str, Any] | None:
    """Extract all attributes from an h5py or zarr object, converting to JSON-compatible types."""
    attrs = getattr(obj, "attrs", None)
    if attrs is None:
        return None
    attrs = dict(attrs)

    # Convert to JSON-compatible types
    converted: dict[str, Any] = {}
    for key, value in attrs.items():
        converted[key] = _to_json_compatible(value)

    return converted


def _to_json_compatible(value: object) -> object:
    """Convert a value to JSON-compatible type."""
    import numpy as np

    # Handle numpy arrays
    if isinstance(value, np.ndarray):
        return value.tolist()

    # Handle numpy scalar types
    if isinstance(value, (np.integer, np.floating)):
        return value.item()

    # Handle bytes
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()

    # Handle h5py references
    if hasattr(value, "id") or str(type(value)).startswith("<class 'h5py"):
        return str(value)

    # Handle zarr references
    if str(type(value)).startswith("<class 'zarr"):
        return str(value)

    # Basic JSON-compatible types are fine
    if isinstance(value, (str, int, float, bool, type(None), list, dict)):
        return value

    # Fallback to string representation
    return str(value)


def _filter_attrs(
    attrs: dict[str, Any] | None,
    exclude_private: bool = True,
    exclude_empty_fields: bool = False,
) -> dict[str, Any]:
    if attrs is None:
        return {}
    filtered = attrs.copy()
    if exclude_private:
        filtered = {
            k: v
            for k, v in filtered.items()
            if not k.startswith("_") and not k.startswith(".")
        }
        filtered.pop("id", None)
        filtered.pop("object_id", None)
        filtered.pop("namespace", None)
        filtered.pop("target", None)
    if exclude_empty_fields:
        filtered = {
            k: v
            for k, v in filtered.items()
            if v not in (None, {}, [], "", "no description")
        }
    return filtered


def get_attrs(
    nwb_path: lazynwb.types_.PathLike,
    internal_path: str,
    exclude_private: bool = True,
    exclude_empty_fields: bool = False,
) -> dict[str, Any]:
    """
    Retrieve all attributes for a single internal path in an NWB file (cached).

    Parameters
    ----------
    nwb_path : PathLike
        Path to the NWB file.
    internal_path : str
        Internal path within the NWB file (e.g., '/units', '/units/spike_times',
        '/general/subject').
    exclude_private : bool, default True
        If True, exclude attributes starting with underscore or period, and the 'id' attr.
    exclude_empty_fields : bool, default False
        If True, exclude attributes with empty values (None, empty dict, empty list, empty string, "no description").

    Returns
    -------
    dict[str, Any]
        Dictionary mapping attribute names to their values.
        path has no attributes.

    Examples
    --------
    >>> attrs = get_attrs('data.nwb', '/units')
    >>> print(attrs)  # doctest: +SKIP
    {'neurodata_type': 'DynamicTable', 'help': 'Data...', ...}

    >>> # Get column-specific attributes
    >>> spike_times_attrs = get_attrs('data.nwb', '/units/spike_times')  # doctest: +SKIP
    """
    file_accessor = lazynwb.file_io._get_accessor(nwb_path)
    cache_key = _get_cache_key(file_accessor)
    internal_path = lazynwb.utils.normalize_internal_file_path(internal_path)

    with _attrs_cache_lock:
        # Check cache
        if cache_key in _attrs_cache:
            if internal_path in _attrs_cache[cache_key]:
                cached_attrs = _attrs_cache[cache_key][internal_path]
                # Return filtered view of cached attrs
                return _filter_attrs(
                    cached_attrs,
                    exclude_private=exclude_private,
                    exclude_empty_fields=exclude_empty_fields,
                )

        # Cache miss: retrieve from file
        attrs: dict | None = _get_attrs_from_accessor(file_accessor[internal_path])

        # Store in cache (even if None, to avoid repeat lookup)
        if cache_key not in _attrs_cache:
            _attrs_cache[cache_key] = {}
        _attrs_cache[cache_key][internal_path] = attrs

        return _filter_attrs(
            attrs,
            exclude_private=exclude_private,
            exclude_empty_fields=exclude_empty_fields,
        )


def _post_process_attrs(
    result: dict[str, dict[str, Any]],
    exclude_private: bool = True,
    exclude_empty: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Post-process collected attributes to exclude _index counterparts, empty dicts,
    specifications paths, and filter neurodata_type for table columns.

    Parameters
    ----------
    result : dict[str, dict[str, Any]]
        Raw attributes collected from traversal.
    exclude_private : bool, default True
        If True, filter out neurodata_type for VectorData/VectorIndex columns.
    exclude_empty : bool, default True
        If True, exclude paths with empty attribute dicts.

    Returns
    -------
    dict[str, dict[str, Any]]
        Processed attributes with filtering applied.
    """
    filtered_result: dict[str, dict[str, Any]] = {}
    all_paths = set(result.keys())

    for path, attrs in result.items():
        # Skip /specifications paths
        if path.startswith("/specifications"):
            continue

        # Skip _index counterparts if their base counterpart exists
        if path.endswith("_index"):
            base_path = path[:-6]  # Remove "_index"
            if base_path in all_paths:
                continue

        # Filter out neurodata_type for table columns if exclude_private
        filtered_attrs = attrs.copy()
        if exclude_private and "neurodata_type" in filtered_attrs:
            ntype = filtered_attrs["neurodata_type"]
            if ntype in ("VectorData", "VectorIndex", "ElementIdentifiers"):
                filtered_attrs.pop("neurodata_type")

        # Skip empty dicts if exclude_empty
        if exclude_empty and not filtered_attrs:
            continue

        filtered_result[path] = filtered_attrs

    return filtered_result


def get_sub_attrs(
    nwb_path: lazynwb.types_.PathLike,
    parent_path: str = "/",
    exclude_private: bool = True,
    exclude_empty: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Retrieve all attributes for all objects under a parent path (optimized).

    Recursively traverses the file structure from the parent path and collects attributes for each group and dataset encountered. Results are cached.

    Parameters
    ----------
    nwb_path : PathLike
        Path to the NWB file.
    parent_path : str, default "/"
        Parent path to start traversal from (e.g., '/units', '/general').
    exclude_private : bool, default True
        If True, exclude attributes starting with underscore or period, and the 'id' attr.
    exclude_empty : bool, default True
        If True, exclude paths with empty attribute dicts.

    Returns
    -------
    dict[str, dict[str, Any]]
        Dictionary mapping internal paths to their attributes. Each value is itself a dict
        mapping attribute names to their values.
        Example: {'/units': {...}, '/units/spike_times': {...}, '/general/subject': {...}}

    Examples
    --------
    >>> all_units_attrs = get_sub_attrs('data.nwb', parent_path='/units')
    >>> print(list(all_units_attrs.keys()))  # doctest: +SKIP
    ['/units', '/units/spike_times', '/units/spike_times_index', '/units/waveform_mean', ...]
    """
    file_accessor = lazynwb.file_io._get_accessor(nwb_path)
    cache_key = _get_cache_key(file_accessor)
    parent_path = lazynwb.utils.normalize_internal_file_path(parent_path)

    result: dict[str, dict[str, Any]] = {}

    def _traverse(current_path: str) -> None:
        """Recursively traverse and collect attrs from all sub-objects."""
        # Get attrs for current path
        try:
            if current_path != "/":
                obj = file_accessor.get(current_path)
            else:
                obj = file_accessor._accessor
            if obj is None:
                return
        except (KeyError, AttributeError, TypeError):
            return

        # Cache and collect attrs for this path
        with _attrs_cache_lock:
            if cache_key not in _attrs_cache:
                _attrs_cache[cache_key] = {}

            if current_path not in _attrs_cache[cache_key]:
                attrs = _get_attrs_from_accessor(obj)
                _attrs_cache[cache_key][current_path] = attrs
            else:
                attrs = _attrs_cache[cache_key][current_path]

        result[current_path] = _filter_attrs(
            attrs, exclude_private=exclude_private, exclude_empty_fields=exclude_empty
        )

        # Recurse into children
        if lazynwb.file_io.is_group(obj):
            try:
                for key in obj.keys():
                    child_path = f"{current_path.rstrip('/')}/{key}"
                    _traverse(child_path)
            except (AttributeError, TypeError, ValueError):
                pass

    _traverse(parent_path)
    return _post_process_attrs(
        result, exclude_private=exclude_private, exclude_empty=exclude_empty
    )


def _to_hashable_key(value: object) -> str:
    """Convert a value to a hashable string key for comparison."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return str(value)
    return str(value)


def _consolidate_single_attr(
    attr_name: str,
    values_by_file: dict[str, Any],
) -> dict[str, Any]:
    """
    Consolidate a single attribute across files.

    Parameters
    ----------
    attr_name : str
        Name of the attribute (for logging).
    values_by_file : dict[str, Any]
        Mapping of file paths to attribute values.

    Returns
    -------
    dict[str, Any]
        Entry with 'common' value and divergent file paths.
    """
    if not values_by_file:
        return {}

    # Count value occurrences
    value_counts: dict[str, tuple[int, Any]] = {}
    for _, value in values_by_file.items():
        key = _to_hashable_key(value)
        if key not in value_counts:
            value_counts[key] = (0, value)
        count, _ = value_counts[key]
        value_counts[key] = (count + 1, value)

    # Get the most common value
    common_key, (_, common_value) = max(
        value_counts.items(),
        key=lambda x: x[1][0],
    )

    # Build the consolidated entry
    attr_entry: dict[str, Any] = {"common": common_value}

    # Add divergent values
    for path, value in values_by_file.items():
        value_key = _to_hashable_key(value)
        if value_key != common_key:
            attr_entry[path] = value

    return attr_entry


def consolidate_attrs(
    nwb_paths: lazynwb.types_.PathLike | Iterable[lazynwb.types_.PathLike],
    internal_path: str,
    exclude_private: bool = True,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Consolidate attributes across multiple NWB files.

    Finds attributes for a specific path across multiple files, identifying which attributes
    are common across all files and which vary.

    Parameters
    ----------
    nwb_paths : PathLike or iterable of PathLike
        Path(s) to NWB file(s). If a single path is provided, it will be wrapped in a list.
    internal_path : str
        Internal path to get attributes for (e.g., '/units', '/units/spike_times').
    exclude_private : bool, default True
        If True, exclude attributes starting with underscore or period, and the 'id' attr.

    Returns
    -------
    dict[str, dict[str, dict[str, Any]]]
        Consolidated attributes with structure:
        {
            internal_path: {
                attr_name: {
                    'common': <value>,
                    'nwb_path_1': <value>,  # only if differs from common
                    'nwb_path_2': <value>,  # only if differs from common
                    ...
                },
                ...
            }
        }

        The 'common' key always exists and contains the most frequent value for that attribute
        across all files. For files with different values, their path is added as a key with
        that file's specific value. If all files have the same value, only 'common' is present.

    Examples
    --------
    >>> consolidated = consolidate_attrs(
    ...     ['file1.nwb', 'file2.nwb', 'file3.nwb'],
    ...     '/units/spike_times'
    ... )
    >>> print(consolidated)  # doctest: +SKIP
    {
        '/units/spike_times': {
            'neurodata_type': {'common': 'VectorData'},
            'units': {'common': 'seconds'},
            'help': {
                'common': 'Spike times for unit 0',
                '/path/to/file2.nwb': 'Spike times for unit X'
            }
        }
    }
    """
    # Normalize input to list of paths
    if isinstance(nwb_paths, (str, bytes)) or not isinstance(nwb_paths, Iterable):
        paths: list[lazynwb.types_.PathLike] = [nwb_paths]
    else:
        paths = list(nwb_paths)  # type: ignore[arg-type]

    if not paths:
        raise ValueError("At least one NWB path must be provided")

    internal_path = lazynwb.utils.normalize_internal_file_path(internal_path)

    # Collect attrs from all files
    all_attrs: dict[str, dict[str, Any]] = {}
    for nwb_path in paths:
        file_attrs = get_attrs(
            nwb_path=nwb_path,
            internal_path=internal_path,
            exclude_private=exclude_private,
        )
        all_attrs[str(nwb_path)] = file_attrs

    # Find all unique attribute names
    all_attr_names: set[str] = set()
    for file_attrs in all_attrs.values():
        all_attr_names.update(file_attrs.keys())

    # Consolidate each attribute
    consolidated: dict[str, dict[str, dict[str, Any]]] = {internal_path: {}}

    for attr_name in sorted(all_attr_names):
        # Collect values from all files
        values_by_file: dict[str, Any] = {}
        for path, file_attrs in all_attrs.items():
            if attr_name in file_attrs:
                values_by_file[path] = file_attrs[attr_name]

        if values_by_file:
            attr_entry = _consolidate_single_attr(attr_name, values_by_file)
            consolidated[internal_path][attr_name] = attr_entry

    return consolidated


def clear_attrs_cache(
    nwb_path: lazynwb.types_.PathLike | None = None,
) -> None:
    """
    Clear cached attributes for a specific file or all files.

    Parameters
    ----------
    nwb_path : PathLike, optional
        If provided, only clear the cache for this specific file.
        If None, clear the entire attrs cache.

    Examples
    --------
    >>> clear_attrs_cache()  # Clear all cached attrs
    >>> clear_attrs_cache('data.nwb')  # Clear cache for specific file
    """
    global _attrs_cache
    with _attrs_cache_lock:
        if nwb_path is None:
            _attrs_cache.clear()
            logger.debug("Cleared all attrs caches")
        else:
            file_accessor = lazynwb.file_io._get_accessor(nwb_path)
            cache_key = _get_cache_key(file_accessor)
            _attrs_cache.pop(cache_key, None)
            logger.debug(f"Cleared attrs cache for {cache_key}")


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
