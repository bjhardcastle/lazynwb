"""Convert NWB files to various output formats."""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import pathlib
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any, Literal, cast

import polars as pl
import tqdm

import lazynwb._catalog.models as catalog_models
import lazynwb._hdf5.reader as hdf5_reader
import lazynwb.base
import lazynwb.file_io
import lazynwb.lazyframe
import lazynwb.tables
import lazynwb.types_
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
    nwb_sources: lazynwb.types_.PathLike | Iterable[lazynwb.types_.PathLike],
    output_dir: pathlib.Path | str,
    *,
    output_format: OutputFormat = "parquet",
    full_path: bool = False,
    min_file_count: int = 1,
    exclude_array_columns: bool = False,
    ignore_errors: bool = True,
    disable_progress: bool = False,
    session_subdirs: bool = False,
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
    session_subdirs : bool, default False
        If True, write each session's tables into a subdirectory named by /general/session_id
        (falling back to /identifier). The common schema across all files is computed once per
        table path and applied consistently to every session file. Output dict keys become
        "{session_name}{table_path}" (e.g. "abc123/intervals/trials").
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
    sources = cast("tuple[lazynwb.types_.PathLike, ...]", tuple(nwb_sources))

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Discovering tables in {len(sources)} NWB files...")

    # Find common table paths across all files using threadpool
    common_table_paths = _find_common_paths(
        nwb_sources=sources,
        min_file_count=min_file_count,
        disable_progress=disable_progress,
    )

    if not common_table_paths:
        logger.warning("No common table paths found across NWB files")
        return {}

    _warm_hdf5_schema_snapshots_for_sql_context(
        nwb_sources=nwb_sources,
        table_paths=common_table_paths,
    )

    logger.info(
        f"Found {len(common_table_paths)} common table paths: {sorted(common_table_paths)}"
    )

    # Convert each table to specified format
    output_paths: dict[str, pathlib.Path] = {}
    write_method, file_extension = _OUTPUT_FORMATS[output_format]

    if session_subdirs:
        output_paths = _convert_nwb_tables_session_subdir(
            nwb_sources=sources,
            output_dir=output_dir,
            common_table_paths=common_table_paths,
            write_method=write_method,
            file_extension=file_extension,
            full_path=full_path,
            exclude_array_columns=exclude_array_columns,
            ignore_errors=ignore_errors,
            disable_progress=disable_progress,
            write_kwargs=write_kwargs,
        )
    else:
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
                source=sources,
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

            logger.info(
                f"Wrote {df.height} rows, {df.width} columns to {output_path.name}"
            )

    logger.info(f"Successfully converted {len(output_paths)} tables to {output_format}")
    return output_paths


def _convert_nwb_tables_session_subdir(
    nwb_sources: Sequence[lazynwb.types_.PathLike],
    output_dir: pathlib.Path,
    common_table_paths: set[str],
    write_method: str,
    file_extension: str,
    full_path: bool,
    exclude_array_columns: bool,
    ignore_errors: bool,
    disable_progress: bool,
    write_kwargs: dict[str, Any],
) -> dict[str, pathlib.Path]:
    """Write each NWB file's tables into a per-session subdirectory.

    The common schema across all files is computed once per table path and applied
    consistently to every session file to guarantee uniform column sets and types.
    Dict keys use the form "{session_name}{table_path}" (e.g. "abc123/intervals/trials").
    """
    # Cache common schema for each table path across ALL files up front
    table_schemas: dict[str, pl.Schema] = {}
    for table_path in common_table_paths:
        try:
            table_schemas[table_path] = lazynwb.tables.get_table_schema(
                file_paths=nwb_sources,
                table_path=table_path,
                exclude_array_columns=exclude_array_columns,
            )
            logger.debug(
                f"Cached schema for {table_path!r}: {list(table_schemas[table_path].keys())}"
            )
        except Exception as exc:
            if not ignore_errors:
                raise
            logger.warning(f"Could not compute schema for {table_path!r}: {exc}")

    output_paths: dict[str, pathlib.Path] = {}

    sources_iter: Iterable[lazynwb.types_.PathLike] = nwb_sources
    if not disable_progress:
        sources_iter = tqdm.tqdm(
            sources_iter,
            total=len(nwb_sources),
            desc="Converting sessions",
            unit="session",
            ncols=80,
        )

    for nwb_source in sources_iter:
        session_name = _get_session_name(nwb_source)
        session_dir = output_dir / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing session {session_name!r} -> {session_dir}")

        for table_path, schema in table_schemas.items():
            output_path = _table_path_to_output_path(
                session_dir, table_path, file_extension, full_path=full_path
            )
            if _write_session_table(
                nwb_source=nwb_source,
                table_path=table_path,
                schema=schema,
                output_path=output_path,
                write_method=write_method,
                exclude_array_columns=exclude_array_columns,
                ignore_errors=ignore_errors,
                write_kwargs=write_kwargs,
            ):
                output_paths[f"{session_name}{table_path}"] = output_path

    return output_paths


def _write_session_table(
    nwb_source: lazynwb.types_.PathLike,
    table_path: str,
    schema: pl.Schema,
    output_path: pathlib.Path,
    write_method: str,
    exclude_array_columns: bool,
    ignore_errors: bool,
    write_kwargs: dict[str, Any],
) -> bool:
    """Read one table from one NWB file and write it. Returns True if written."""
    try:
        df = lazynwb.lazyframe.scan_nwb(
            source=[nwb_source],
            table_path=table_path,
            schema=schema,
            exclude_array_columns=exclude_array_columns,
            ignore_errors=ignore_errors,
            disable_progress=True,
        ).collect()
    except Exception as exc:
        if not ignore_errors:
            raise
        logger.warning(f"Error reading {table_path!r} from {nwb_source!r}: {exc}")
        return False

    if df.is_empty():
        logger.warning(f"Table {table_path!r} is empty for {nwb_source!r}, skipping")
        return False

    getattr(df, write_method)(output_path, **write_kwargs)
    logger.info(f"Wrote {df.height} rows to {output_path}")
    return True


def _get_session_name(nwb_source: lazynwb.types_.PathLike) -> str:
    """Get a session directory name from an NWB file.

    Prefers /general/session_id, falls back to /identifier, then the file stem.
    The result is sanitized to be safe as a directory name.
    """
    accessor = lazynwb.file_io._get_accessor(nwb_source)
    for internal_path in ("/general/session_id", "identifier"):
        value = lazynwb.base._cast(accessor, internal_path)
        if value and isinstance(value, str) and value.strip():
            return re.sub(r'[\\/:*?"<>|]', "_", value).strip()
    logger.warning(
        f"No session_id or identifier found in {nwb_source!r}, using file stem"
    )
    return pathlib.Path(str(nwb_source)).stem


def get_sql_context(
    nwb_sources: lazynwb.types_.PathLike | Iterable[lazynwb.types_.PathLike],
    *,
    full_path: bool = False,
    min_file_count: int = 1,
    exclude_array_columns: bool = False,
    exclude_timeseries: bool = False,
    ignore_errors: bool = True,
    disable_progress: bool = False,
    infer_schema_length: int | None = None,
    table_names: Iterable[str] | None = None,
    rename_general_metadata: bool = True,
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
        include_timeseries=not exclude_timeseries,
        include_metadata=True,
    )

    if not common_table_paths:
        logger.warning("No common table paths found across NWB files")
        return {}

    _warm_hdf5_schema_snapshots_for_sql_context(
        nwb_sources=nwb_sources,
        table_paths=common_table_paths,
    )

    logger.info(
        f"Found {len(common_table_paths)} common table paths: {sorted(common_table_paths)}"
    )
    if table_names is not None:
        # Filter to only include specified table names
        requested_table_names = set(table_names)
        registered_table_names = {
            _sql_table_name(
                path,
                full_path=full_path,
                rename_general_metadata=rename_general_metadata,
            )
            for path in common_table_paths
        }
        if not requested_table_names.issubset(registered_table_names):
            raise ValueError(
                f"{table_names=} do not all match paths in NWB files: {registered_table_names}"
                f" ({full_path=} can be toggled to use just the last part of the path)"
            )
        common_table_paths = {
            path
            for path in common_table_paths
            if _sql_table_name(
                path,
                full_path=full_path,
                rename_general_metadata=rename_general_metadata,
            )
            in requested_table_names
        }

    sql_context = pl.SQLContext(**sqlcontext_kwargs)
    for table_path in sorted(common_table_paths):
        table_name = _sql_table_name(
            table_path,
            full_path=full_path,
            rename_general_metadata=rename_general_metadata,
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


def _sql_table_name(
    table_path: str,
    *,
    full_path: bool,
    rename_general_metadata: bool,
) -> str:
    normalized_path = lazynwb.utils.normalize_internal_file_path(table_path)
    table_name = normalized_path if full_path else normalized_path.split("/")[-1]
    if rename_general_metadata and normalized_path == "general":
        return "session"
    return table_name


def _find_common_paths(
    nwb_sources: tuple[lazynwb.types_.PathLike, ...],
    min_file_count: int,
    disable_progress: bool,
    include_timeseries: bool = False,
    include_metadata: bool = True,
) -> set[str]:
    """Find table paths that appear in at least min_file_count files."""

    # Use threadpool to get internal paths from all files in parallel
    future_to_path = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for nwb_path in nwb_sources:
            future = executor.submit(
                _get_internal_paths_for_discovery,
                nwb_path,
                include_timeseries,
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
                if include_timeseries:
                    array_paths = _filter_timeseries_paths(
                        {
                            k: v
                            for k, v in internal_paths.items()
                            if k not in table_paths
                        }
                    )
                    all_table_paths.extend(array_paths)
                    logger.debug(f"Found {len(array_paths)} array paths in {nwb_path}")
                if include_metadata:
                    # Include metadata paths as well
                    all_table_paths.extend(
                        [
                            k
                            for k in ["/general", "/general/subject"]
                            if k in internal_paths
                        ]
                    )
    # Count occurrences and filter by min_file_count
    path_counts = Counter(all_table_paths)
    common_paths = {
        path for path, count in path_counts.items() if count >= min_file_count
    }

    return common_paths


def _warm_hdf5_schema_snapshots_for_sql_context(
    nwb_sources: tuple[lazynwb.types_.PathLike, ...],
    table_paths: Iterable[str],
) -> None:
    exact_table_paths = tuple(
        dict.fromkeys(
            lazynwb.utils.normalize_internal_file_path(table_path)
            for table_path in table_paths
        )
    )
    if not exact_table_paths:
        return
    for nwb_source in nwb_sources:
        if not hdf5_reader._is_fast_hdf5_candidate(nwb_source):
            continue
        reader = hdf5_reader._default_hdf5_backend_reader(nwb_source)
        request_count_before = int(getattr(reader._range_reader, "request_count", 0))
        fetched_bytes_before = int(getattr(reader._range_reader, "bytes_fetched", 0))
        try:
            results = lazynwb.tables._run_async_value(
                reader._read_table_schema_snapshots(exact_table_paths)
            )
        except hdf5_reader._NotHDF5Error:
            logger.debug("SQL context HDF5 multi-table scan skipped non-HDF5 %r", nwb_source)
            continue
        except Exception as exc:
            logger.debug(
                "SQL context HDF5 multi-table scan failed for %r: %r",
                nwb_source,
                exc,
            )
            continue
        finally:
            with contextlib.suppress(Exception):
                lazynwb.tables._run_async_value(reader.close())
        ok_count = sum(result.ok for result in results.values())
        failure_count = len(results) - ok_count
        request_count = int(getattr(reader._range_reader, "request_count", 0))
        fetched_bytes = int(getattr(reader._range_reader, "bytes_fetched", 0))
        logger.debug(
            "SQL context HDF5 multi-table scan for %r: tables=%d ok=%d "
            "failures=%d requests=%d bytes=%d cache_writes=%d",
            nwb_source,
            len(exact_table_paths),
            ok_count,
            failure_count,
            request_count - request_count_before,
            fetched_bytes - fetched_bytes_before,
            ok_count,
        )
        for table_path, result in results.items():
            if result.ok:
                logger.debug(
                    "SQL context HDF5 multi-table scan cached %r/%s "
                    "(requests=%d bytes=%d)",
                    nwb_source,
                    table_path,
                    result.request_count,
                    result.fetched_bytes,
                )
            else:
                logger.debug(
                    "SQL context HDF5 multi-table scan table failure %r/%s: %r",
                    nwb_source,
                    table_path,
                    result.error,
                )


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
        ) and _is_group_path_entry(accessor):
            table_paths.append(path)
            continue

        # Check if the accessor has table-like attributes
        attrs = _path_entry_attrs(accessor)
        if "colnames" in attrs:  # Standard NWB table indicator
            table_paths.append(path)
            continue

    return table_paths


def _filter_timeseries_paths(internal_paths: dict[str, Any]) -> list[str]:
    """Filter internal paths to identify TimeSeries-like structures."""
    timeseries_paths = []

    for path, accessor in internal_paths.items():
        # Check if the accessor has TimeSeries-like attributes
        if path.endswith("/data") or path.endswith("/timestamps"):
            continue
        if not _is_group_path_entry(accessor):
            continue
        attrs = _path_entry_attrs(accessor)

        try:
            if (
                # required attributes for TimeSeries objects
                (
                    f"{path}/timestamps" in internal_paths
                    or (
                        "timestamps" in accessor
                        if not isinstance(accessor, catalog_models._PathSummaryEntry)
                        else False
                    )
                    or _path_entry_has_rate_starting_time(path, accessor, internal_paths)
                )
                or
                # try to accommodate possible variants
                (
                    "series" in attrs.get("neurodata_type", "").lower()
                    and (
                        f"{path}/data" in internal_paths
                        or (
                            "data" in accessor
                            if not isinstance(
                                accessor,
                                catalog_models._PathSummaryEntry,
                            )
                            else False
                        )
                    )
                )
            ):
                timeseries_paths.append(path)
        except AttributeError:
            continue

    return timeseries_paths


def _get_internal_paths_for_discovery(
    nwb_path: lazynwb.types_.PathLike,
    include_timeseries: bool,
) -> dict[str, Any]:
    summary = lazynwb.file_io._get_catalog_path_summary_if_available(
        nwb_path=nwb_path,
        include_arrays=include_timeseries,
        include_table_columns=False,
        include_metadata=True,
        include_specifications=False,
        parents=True,
    )
    if summary is not None:
        logger.debug("path discovery used catalog summary for %r", nwb_path)
        return summary
    logger.debug("path discovery falling back to accessor traversal for %r", nwb_path)
    return lazynwb.file_io.get_internal_paths(
        nwb_path=nwb_path,
        include_arrays=include_timeseries,
        include_table_columns=False,
        include_metadata=True,
        include_specifications=False,
        parents=True,
    )


def _is_group_path_entry(accessor: Any) -> bool:
    if isinstance(accessor, catalog_models._PathSummaryEntry):
        return accessor.is_group
    if isinstance(accessor, dict):
        return bool(accessor.get("is_group", False))
    return lazynwb.file_io.is_group(accessor)


def _path_entry_attrs(accessor: Any) -> dict[str, Any]:
    if isinstance(accessor, dict):
        return dict(accessor.get("attrs", {}))
    return dict(getattr(accessor, "attrs", {}))


def _path_entry_has_rate_starting_time(
    path: str,
    accessor: Any,
    internal_paths: dict[str, Any],
) -> bool:
    starting_time = internal_paths.get(f"{path}/starting_time")
    if starting_time is not None:
        return "rate" in _path_entry_attrs(starting_time)
    if isinstance(accessor, catalog_models._PathSummaryEntry) or isinstance(
        accessor,
        dict,
    ):
        return False
    return "rate" in getattr(accessor.get("starting_time", {}), "attrs", {})


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
