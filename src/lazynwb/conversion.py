"""Convert NWB files to various output formats."""

from __future__ import annotations

import concurrent.futures
import logging
import pathlib
from collections import Counter
from collections.abc import Iterable
from typing import Any, Literal

import npc_io
import polars as pl
import tqdm

import lazynwb.file_io
import lazynwb.lazyframe
import lazynwb.utils

logger = logging.getLogger(__name__)

# Supported output formats mapped to their write methods and file extensions
_OUTPUT_FORMATS = {
    "parquet": ("write_parquet", ".parquet"),
    "csv": ("write_csv", ".csv"),
    "json": ("write_json", ".json"),
    "excel": ("write_excel", ".xlsx"),
    "feather": ("write_ipc", ".feather"),
    "arrow": ("write_ipc", ".arrow"),
    "avro": ("write_avro", ".avro"),
    "delta": ("write_delta", ""),  # Delta uses directory structure
}

OutputFormat = Literal[
    "parquet", "csv", "json", "excel", "feather", "arrow", "avro", "delta"
]


def convert_nwb_tables(
    nwb_sources: npc_io.PathLike | Iterable[npc_io.PathLike],
    output_dir: pathlib.Path | str,
    *,
    output_format: OutputFormat = "parquet",
    full_path: bool = False,
    min_file_count: int = 1,
    exclude_array_columns: bool = False,
    ignore_errors: bool = True,
    disable_progress: bool = False,
    **write_kwargs: Any,
) -> dict[str, pathlib.Path]:
    """
    Convert NWB files to specified output format, creating one file per common table.

    Uses `get_internal_paths` in a threadpool to discover tables across all NWB files,
    then `scan_nwb` to efficiently read and export each common table.

    Parameters
    ----------
    nwb_sources : PathLike or iterable of PathLike
        Paths to NWB files to convert. May be local paths, S3 URLs, or other supported formats.
    output_dir : Path or str
        Directory where output files will be written. Will be created if it doesn't exist.
    output_format : str, default "parquet"
        Output format for files. Supported formats: "parquet", "csv", "json", "excel",
        "feather", "arrow", "avro", "delta".
    full_path : bool, default False
        If False, table names are assumed to be unique and the full path will be truncated,
        e.g. 'trials.parquet' instead of 'intervals_trials.parquet'.
    min_file_count : int, default 1
        Minimum number of files that must contain a table path for it to be exported.
        Use 1 to export all tables found in any file, or len(nwb_sources) to export
        only tables present in all files.
    exclude_array_columns : bool, default True
        If True, columns containing array/list data will be excluded from exported tables.
        Array columns can significantly increase file size and may not be suitable for
        all analytical workflows.
    ignore_errors : bool, default True
        If True, continue processing other tables when errors occur reading specific tables.
    disable_progress : bool, default False
        If True, progress bars will be disabled.
    **write_kwargs : Any
        Additional keyword arguments passed to the polars DataFrame write method.
        For parquet: compression="snappy" or compression="zstd"
        For csv: separator=",", has_header=True
        For json: pretty=True, row_oriented=False
        See polars documentation for format-specific options.

    Returns
    -------
    dict[str, pathlib.Path]
        Dictionary mapping table paths to their corresponding output file paths.

    Raises
    ------
    ValueError
        If output_format is not supported.

    Examples
    --------
    Convert all NWB files in a directory to Parquet:

    >>> import lazynwb
    >>> nwb_files = list(pathlib.Path("/data/nwb").glob("*.nwb"))
    >>> output_paths = lazynwb.convert_nwb_tables(
    ...     nwb_files,
    ...     output_dir="/data/parquet",
    ...     output_format="parquet",
    ...     compression="snappy"
    ... )
    >>> output_paths
    {'/intervals/trials': PosixPath('/data/parquet/intervals_trials.parquet'),
     '/units': PosixPath('/data/parquet/units.parquet')}

    Convert to CSV format:

    >>> output_paths = lazynwb.convert_nwb_tables(
    ...     nwb_files,
    ...     output_dir="/data/csv",
    ...     output_format="csv",
    ...     separator=",",
    ...     has_header=True
    ... )

    Export only tables present in all files to JSON:

    >>> output_paths = lazynwb.convert_nwb_tables(
    ...     nwb_files,
    ...     output_dir="/data/json",
    ...     output_format="json",
    ...     min_file_count=len(nwb_files),
    ...     exclude_array_columns=False,
    ...     pretty=True
    ... )
    """
    output_format = output_format.lower().strip(".")
    if output_format not in _OUTPUT_FORMATS:
        raise ValueError(
            f"Unsupported output format '{output_format}'. "
            f"Supported formats: {list(_OUTPUT_FORMATS.keys())}"
        )

    if isinstance(nwb_sources, (str, pathlib.Path)) or not isinstance(
        nwb_sources, Iterable
    ):
        nwb_sources = (nwb_sources,)
    nwb_sources = tuple(nwb_sources)

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Discovering tables in {len(nwb_sources)} NWB files...")

    # Find common table paths across all files using threadpool
    common_table_paths = _find_common_paths(
        nwb_sources=nwb_sources,
        min_file_count=min_file_count,
        disable_progress=disable_progress,
    )

    if not common_table_paths:
        logger.warning("No common table paths found across NWB files")
        return {}

    logger.info(
        f"Found {len(common_table_paths)} common table paths: {sorted(common_table_paths)}"
    )

    # Convert each table to specified format
    output_paths: dict[str, pathlib.Path] = {}
    write_method, file_extension = _OUTPUT_FORMATS[output_format]

    for table_path in common_table_paths:
        output_path = _table_path_to_output_path(
            output_dir,
            table_path,
            file_extension,
            full_path=full_path,
        )
        logger.info(f"Converting {table_path} -> {output_path.name}")

        # Read table across all files
        df = lazynwb.lazyframe.scan_nwb(
            source=nwb_sources,
            table_path=table_path,
            exclude_array_columns=exclude_array_columns,
            ignore_errors=ignore_errors,
            disable_progress=disable_progress,
        ).collect()

        if df.is_empty():
            logger.warning(f"Table {table_path} is empty, skipping")
            continue

        # Write using the appropriate method
        write_func = getattr(df, write_method)
        write_func(output_path, **write_kwargs)
        output_paths[table_path] = output_path

        logger.info(f"Wrote {df.height} rows, {df.width} columns to {output_path.name}")

    logger.info(f"Successfully converted {len(output_paths)} tables to {output_format}")
    return output_paths


def get_sql_context(
    nwb_sources: npc_io.PathLike | Iterable[npc_io.PathLike],
    *,
    full_path: bool = False,
    min_file_count: int = 1,
    exclude_array_columns: bool = False,
    ignore_errors: bool = True,
    disable_progress: bool = False,
    infer_schema_length: int | None = None,
    table_names: Iterable[str] | None = None,
    **sqlcontext_kwargs: Any,
) -> pl.SQLContext:

    if isinstance(nwb_sources, (str, pathlib.Path)) or not isinstance(
        nwb_sources, Iterable
    ):
        nwb_sources = (nwb_sources,)
    nwb_sources = tuple(nwb_sources)

    logger.info(
        f"Discovering tables in {infer_schema_length or len(nwb_sources)} NWB files..."
    )

    # Find common table paths across all files using threadpool
    common_table_paths = _find_common_paths(
        nwb_sources=(
            nwb_sources[:infer_schema_length] if infer_schema_length else nwb_sources
        ),
        min_file_count=min_file_count,
        disable_progress=disable_progress,
        include_arrays=True,  # Include arrays to be able to query them
    )

    if not common_table_paths:
        logger.warning("No common table paths found across NWB files")
        return {}

    logger.info(
        f"Found {len(common_table_paths)} common table paths: {sorted(common_table_paths)}"
    )
    if not full_path:
        # Normalize paths to just the last part if full_path is False
        common_table_paths = {
            lazynwb.utils.normalize_internal_file_path(path)
            for path in common_table_paths
        }

    if table_names is not None:
        # Filter to only include specified table names
        norm_table_names = {
            (
                lazynwb.utils.normalize_internal_file_path(name)
                if full_path
                else name.split("/")[-1]
            )
            for name in table_names
        }
        if not set(table_names).issubset(norm_table_names):
            raise ValueError(
                f"{table_names=} do not all match paths in NWB files: {norm_table_names}"
                f" ({full_path=} can be toggled to use just the last part of the path)"
            )
        common_table_paths = sorted(set(common_table_paths) & set(table_names))

    sql_context = pl.SQLContext(**sqlcontext_kwargs)
    for table_path in sorted(common_table_paths):
        table_name = (
            lazynwb.utils.normalize_internal_file_path(table_path)
            if full_path
            else table_path.split("/")[-1]
        )

        logger.info(f"Adding {table_path} as {table_name}")

        sql_context.register(
            table_name,
            lazynwb.lazyframe.scan_nwb(
                source=nwb_sources,
                table_path=table_path,
                exclude_array_columns=exclude_array_columns,
                ignore_errors=ignore_errors,
                disable_progress=disable_progress,
                infer_schema_length=infer_schema_length,
            ),
        )

    return sql_context


def _find_common_paths(
    nwb_sources: tuple[npc_io.PathLike, ...],
    min_file_count: int,
    disable_progress: bool,
    include_arrays: bool = False,
) -> set[str]:
    """Find table paths that appear in at least min_file_count files."""

    # Use threadpool to get internal paths from all files in parallel
    future_to_path = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for nwb_path in nwb_sources:
            future = executor.submit(
                lazynwb.file_io.get_internal_paths,
                nwb_path=nwb_path,
                include_arrays=include_arrays,  # We only want table-like structures
                include_table_columns=False,
                include_metadata=False,
                include_specifications=False,
                parents=True,  # Include table groups themselves
            )
            future_to_path[future] = nwb_path

        # Collect results with progress tracking
        futures = concurrent.futures.as_completed(future_to_path)
        if not disable_progress:
            futures = tqdm.tqdm(
                futures,
                total=len(future_to_path),
                desc="Scanning NWB files",
                unit="file",
                ncols=80,
            )

        all_table_paths: list[str] = []
        for future in futures:
            nwb_path = future_to_path[future]
            try:
                internal_paths = future.result()
            except Exception as exc:
                logger.warning(f"Error scanning {nwb_path}: {exc}")
                continue
            else:
                # Filter for table-like paths (groups with attributes indicating they're tables)
                table_paths = _filter_table_paths(internal_paths)
                all_table_paths.extend(table_paths)
                logger.debug(f"Found {len(table_paths)} table paths in {nwb_path}")
                if include_arrays:
                    array_paths = _filter_array_paths(
                        {
                            k: v
                            for k, v in internal_paths.items()
                            if k not in table_paths
                        }
                    )
                    all_table_paths.extend(array_paths)
                    logger.debug(f"Found {len(array_paths)} array paths in {nwb_path}")

    # Count occurrences and filter by min_file_count
    path_counts = Counter(all_table_paths)
    common_paths = {
        path for path, count in path_counts.items() if count >= min_file_count
    }

    return common_paths


def _filter_table_paths(internal_paths: dict[str, Any]) -> list[str]:
    """Filter internal paths to identify table-like structures."""
    table_paths = []

    for path, accessor in internal_paths.items():
        # Look for known table patterns
        if any(
            table_pattern in path
            for table_pattern in [
                "/intervals/",
                "/units",
                "/electrodes",
                "/trials",
                "/epochs",
            ]
        ):
            table_paths.append(path)
            continue

        # Check if the accessor has table-like attributes
        attrs = getattr(accessor, "attrs", {})
        if "colnames" in attrs:  # Standard NWB table indicator
            table_paths.append(path)
            continue

    return table_paths


def _filter_array_paths(internal_paths: dict[str, Any]) -> list[str]:
    """Filter internal paths to identify array-like structures."""
    array_paths = []

    for path, accessor in internal_paths.items():
        # Check if the accessor has array-like attributes
        if path.endswith("/data") or path.endswith("/timestamps"):
            continue
        attrs = getattr(accessor, "attrs", {})
        try:
            if (
                # required attributes for TimeSeries objects
                (
                    (has_timestamps := "timestamps" in accessor)
                    or "rate" in getattr(accessor.get("starting_time", {}), "attrs", {})
                )
                or
                # try to accommodate possible variants
                (
                    "series" in attrs.get("neurodata_type", "").lower()
                    and "data" in accessor
                )
            ):
                if not has_timestamps:
                    # TODO
                    # without timestamps, the default TimeSeries object has two keys: 'data' and
                    # 'starting_time' which is another Group.
                    # get_df() interprets them as 'data': List[float], 'starting_time': float
                    # it needs to be aware of this possibility and generate a timestamps column
                    logger.warning(
                        f"TimeSeries without timestamps are not currently supported. Skipping {path}"
                    )
                    continue
                array_paths.append(path)
        except AttributeError:
            continue

    return array_paths


def _table_path_to_output_path(
    output_dir: pathlib.Path,
    table_path: str,
    file_extension: str,
    full_path: bool = True,
) -> pathlib.Path:
    """Convert internal NWB table path to an output filename."""
    # Remove leading slash and replace path separators with underscores
    if full_path:
        path = table_path
    else:
        path = table_path.split("/")[-1]
    clean_path = path.lstrip("/").replace("/", "_").replace(" ", "_").strip()
    if file_extension:
        return output_dir / f"{clean_path}{file_extension}"
    else:
        # For formats like delta that use directories
        return output_dir / clean_path
