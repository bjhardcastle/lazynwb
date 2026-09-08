from __future__ import annotations

import functools
import json
import logging
import operator
from collections.abc import Iterable, Iterator, Mapping, Sequence

import polars as pl
import polars._typing
import polars.io.plugins

import lazynwb._catalog.models as catalog_models
import lazynwb.file_io
import lazynwb.tables
import lazynwb.types_

logger = logging.getLogger(__name__)

_POLARS_DYNAMIC_PREDICATE_TOKEN = "dynamic_pred"


def _align_frame_to_schema(
    df: pl.DataFrame,
    schema: polars._typing.SchemaDict,
    output_columns: Iterable[str],
    table_path: str,
) -> pl.DataFrame:
    """Align requested DataFrame columns to the merged scan schema."""
    output_columns = tuple(output_columns)
    missing_columns = tuple(
        column for column in output_columns if column not in df.columns
    )
    cast_columns = tuple(
        column
        for column in output_columns
        if column in df.columns
        and column in schema
        and df.schema[column] != schema[column]
    )
    if missing_columns:
        logger.debug(
            "Null-filling %d columns absent from the materialized %r frame: %s",
            len(missing_columns),
            table_path,
            missing_columns,
        )
    if cast_columns:
        logger.debug(
            "Casting %d materialized %r columns to the merged scan schema: %s",
            len(cast_columns),
            table_path,
            cast_columns,
        )
    expressions = (
        *(pl.lit(None, dtype=schema[column]).alias(column) for column in missing_columns),
        *(
            pl.col(column).cast(schema[column], strict=False)
            for column in cast_columns
        ),
    )
    return df.with_columns(expressions) if expressions else df


def scan_nwb(  # noqa: C901
    source: lazynwb.types_.PathLike | Iterable[lazynwb.types_.PathLike],
    table_path: str,
    raise_on_missing: bool = False,
    ignore_errors: bool = False,
    infer_schema_length: int | None = None,
    exclude_array_columns: bool = False,
    low_memory: bool = False,
    single_file_batches: bool = False,
    schema: polars._typing.SchemaDict | None = None,
    schema_overrides: polars._typing.SchemaDict | None = None,
    disable_progress: bool = False,
) -> pl.LazyFrame:
    """
    Lazily read from a common table in one or more local or cloud-hosted NWB files.

    This function allows the query optimizer to push down predicates and projections to the scan
    level, typically increasing performance and reducing memory overhead.

    See https://docs.pola.rs/user-guide/lazy/using/#using-the-lazy-api-from-a-file for LazyFrame
    usage.

    Parameters
    ----------
    source : str or PathLike, or iterable of these
        Paths to the NWB file(s) to read from. May be hdf5 or zarr.
    table_path : str
        The internal path to the table in the NWB file, e.g. '/intervals/trials' or '/units'
        It is expected that the table path is the same for all files.
    raise_on_missing : bool, default False
        If True, a KeyError will be raised if the table is not found in every file. Otherwise, a
        KeyError is raised only if the table is not found in any file.
    ignore_errors : bool, default False
        If True, other errors will be ignored when reading files (missing table path errors are
        toggled via `raise_on_missing`).
    infer_schema_length : int, None, default None
        The number of files to read to infer the table schema. If None, all files will be read.
    exclude_array_columns : bool, default False
        If True, columns containing list or array-like data will be excluded from the schema
        and any resulting DataFrame.
    low_memory : bool, default False
        If True, the data will be read in smaller chunks to reduce memory usage, at the cost
        of speed.
    single_file_batches : bool, default False
        If True, each materialization batch contains rows from only one NWB file. This can
        reduce peak memory for large list or array columns, at the cost of cross-file
        parallelism.
    schema : dict[str, pl.DataType], default None
        User-defined schema for the table. If None, the schema will be generated using the stored
        dtypes for columns in each file. Conflicts are signalled to the user via a warning.
    schema_overrides : dict[str, pl.DataType], default None
        User-defined schema for a subset of columns, overriding the inferred schema.
    disable_progress : bool, default False
        If True, progress bars will be disabled.

    Returns
    -------
    pl.LazyFrame
    """
    if not isinstance(source, Iterable) or isinstance(source, str):
        source = (source,)

    source = tuple(source)  # type: ignore[arg-type]
    if not source:
        raise ValueError("No NWB source files provided.")

    scan_catalog_snapshots = {}
    if not schema:
        schema_result = lazynwb.tables._get_table_schema_with_catalog_snapshots(
            file_paths=source,
            table_path=table_path,
            first_n_files_to_infer_schema=infer_schema_length,
            exclude_array_columns=exclude_array_columns,
            exclude_internal_columns=False,
            raise_on_missing=raise_on_missing,
        )
        schema = schema_result.schema
        scan_catalog_snapshots = schema_result.catalog_snapshots
    schema = pl.Schema(schema) | pl.Schema(
        schema_overrides or {}
    )  # create new object to avoid mutating the original schema

    def source_generator(  # noqa: C901
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        """
        Generator function that creates the source, following the example in polars.io.plugins.
        Note: the signature of this function is pre-determined, to fulfill the requirements of the
        register_io_source function.

        Work is split into multiple parts if we have a predicate:
        1) fetch all data for columns in the predicate,
        2) filter the data with the predicate,
        3) join with values from the remaining columns in with_columns, reading only the
           relevant files and rows.

        Without a predicate, we fetch all data for all columns.
        """
        if batch_size is None:
            batch_size = 1_000
            logger.debug(
                "Batch size not specified: using default of %d rows per batch",
                batch_size,
            )
        else:
            logger.debug("Batch size set to %d rows per batch", batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        predicate, dynamic_predicate_count = _remove_polars_dynamic_predicates(
            predicate
        )
        if dynamic_predicate_count:
            logger.debug(
                "Removed %d Polars dynamic predicate conjunct(s) from %r scan "
                "predicate; Python IO plugins do not receive the dynamic TopK state",
                dynamic_predicate_count,
                table_path,
            )

        predicate_filtered_source = _prune_sources_with_nwb_path_predicates(
            source,
            predicate,
        )

        if predicate is not None:
            # Fetch predicate columns, apply the predicate, then fetch remaining columns.
            initial_columns = predicate.meta.root_names()
            logger.debug(
                "Predicate specified: fetching initial columns in %r: %s",
                table_path,
                initial_columns,
            )
        else:
            # - if we don't have a predicate, we'll fetch all required columns in the initial df
            initial_columns = with_columns or []
            logger.debug(
                "Predicate not specified: fetching all requested columns in %r: %s",
                table_path,
                initial_columns,
            )

        if not predicate_filtered_source:
            logger.debug(
                "Skipping %r table materialization because the pushed %s predicate "
                "matched no NWB source files",
                table_path,
                lazynwb.NWB_PATH_COLUMN_NAME,
            )
            yield pl.DataFrame(schema=schema).select(with_columns or schema.keys())
            return

        output_columns = tuple(with_columns or schema.keys())
        if predicate is None:
            include_column_names: set[str] = set()
        else:
            include_column_names = set(output_columns) - set(initial_columns)
            logger.debug(
                "Fetching additional columns from %r after predicate filtering: %s",
                table_path,
                sorted(include_column_names),
            )

        input_row_limit = n_rows if predicate is None else None
        output_rows_yielded = 0
        for batch_index, nwb_path_to_row_indices in enumerate(
            _iter_path_to_row_index_batches(
                source=predicate_filtered_source,
                table_path=table_path,
                batch_size=batch_size,
                n_rows=input_row_limit,
                catalog_snapshots=scan_catalog_snapshots,
                ignore_errors=ignore_errors,
                single_file_batches=single_file_batches,
            ),
            start=1,
        ):
            requested_row_count = sum(
                len(row_indices) for row_indices in nwb_path_to_row_indices.values()
            )
            logger.debug(
                "Fetching %r input batch %d with %d rows from %d NWB sources",
                table_path,
                batch_index,
                requested_row_count,
                len(nwb_path_to_row_indices),
            )
            initial_df = lazynwb.tables.get_df(
                nwb_data_sources=nwb_path_to_row_indices.keys(),
                search_term=table_path,
                exact_path=True,
                include_column_names=initial_columns or None,
                nwb_path_to_row_indices=nwb_path_to_row_indices,
                disable_progress=disable_progress,
                ignore_errors=ignore_errors,
                as_polars=True,
                exclude_array_columns=(
                    False
                    if initial_columns
                    else exclude_array_columns
                    # Explicitly requested array columns override the exclusion setting.
                ),
                low_memory=low_memory,
                _catalog_snapshots=scan_catalog_snapshots,
                _allow_missing_columns=True,
            )
            initial_df = _order_frame_by_source(
                initial_df,
                source_paths=nwb_path_to_row_indices,
            )

            if predicate is None:
                result_df = initial_df
            else:
                result_df = initial_df.filter(predicate)
                logger.debug(
                    "Filtered %r input batch %d from %d to %d rows",
                    table_path,
                    batch_index,
                    initial_df.height,
                    result_df.height,
                )
                if result_df.is_empty():
                    continue

                if n_rows is not None:
                    remaining_output_rows = n_rows - output_rows_yielded
                    if remaining_output_rows <= 0:
                        break
                    result_df = result_df.head(remaining_output_rows)

                if include_column_names:
                    filtered_path_to_row_indices = (
                        lazynwb.tables._get_path_to_row_indices(result_df)
                    )
                    logger.debug(
                        "Fetching %d projected columns for %d rows in %r output batch %d",
                        len(include_column_names),
                        result_df.height,
                        table_path,
                        batch_index,
                    )
                    additional_df = lazynwb.tables.get_df(
                        nwb_data_sources=filtered_path_to_row_indices.keys(),
                        search_term=table_path,
                        exact_path=True,
                        include_column_names=include_column_names,
                        nwb_path_to_row_indices=filtered_path_to_row_indices,
                        disable_progress=disable_progress,
                        use_process_pool=False,
                        as_polars=True,
                        ignore_errors=ignore_errors,
                        low_memory=low_memory,
                        _catalog_snapshots=scan_catalog_snapshots,
                        _allow_missing_columns=True,
                    )
                    result_df = result_df.join(
                        other=additional_df,
                        on=[
                            lazynwb.NWB_PATH_COLUMN_NAME,
                            lazynwb.TABLE_PATH_COLUMN_NAME,
                            lazynwb.TABLE_INDEX_COLUMN_NAME,
                        ],
                        how="inner",
                    )

            result_df = _align_frame_to_schema(
                result_df,
                schema=schema,
                output_columns=output_columns,
                table_path=table_path,
            ).select(output_columns)
            if result_df.is_empty():
                continue

            logger.debug(
                "Yielding %r output batch %d with %d rows and %d columns",
                table_path,
                batch_index,
                result_df.height,
                result_df.width,
            )
            output_rows_yielded += result_df.height
            yield result_df

            if n_rows is not None and output_rows_yielded >= n_rows:
                logger.debug(
                    "Stopped %r materialization after satisfying n_rows=%d",
                    table_path,
                    n_rows,
                )
                break

    return polars.io.plugins.register_io_source(
        io_source=source_generator, schema=schema
    )


def _remove_polars_dynamic_predicates(
    predicate: pl.Expr | None,
) -> tuple[pl.Expr | None, int]:
    if predicate is None or not _predicate_contains_polars_dynamic_predicate(
        predicate
    ):
        return predicate, 0

    conjuncts = _split_conjunctive_predicate(predicate)
    retained_conjuncts: list[pl.Expr] = []
    dropped_conjunct_count = 0
    for conjunct in conjuncts:
        if not _predicate_contains_polars_dynamic_predicate(conjunct):
            retained_conjuncts.append(conjunct)
            continue
        if _is_standalone_polars_dynamic_predicate(conjunct):
            dropped_conjunct_count += 1
            continue
        logger.debug(
            "Cannot isolate Polars dynamic predicate from pushed predicate %s; "
            "leaving predicate unchanged",
            conjunct,
        )
        return predicate, 0

    if dropped_conjunct_count == 0:
        return predicate, 0
    if not retained_conjuncts:
        return None, dropped_conjunct_count
    return functools.reduce(operator.and_, retained_conjuncts), dropped_conjunct_count


def _prune_sources_with_nwb_path_predicates(
    source: tuple[lazynwb.types_.PathLike, ...],
    predicate: pl.Expr | None,
) -> tuple[lazynwb.types_.PathLike, ...]:
    if (
        predicate is None
        or lazynwb.NWB_PATH_COLUMN_NAME not in predicate.meta.root_names()
    ):
        return source

    path_predicates: list[pl.Expr] = []
    for conjunct in _split_conjunctive_predicate(predicate):
        # Polars only passes predicates that it has accepted for scan pushdown. After removing
        # optimizer-only dynamic predicates above, a conjunct rooted solely in the synthetic path
        # column can be evaluated once per source without reading table data.
        if set(conjunct.meta.root_names()) != {lazynwb.NWB_PATH_COLUMN_NAME}:
            continue
        path_predicates.append(conjunct)

    if not path_predicates:
        return source

    path_predicate = functools.reduce(operator.and_, path_predicates)
    source_paths = tuple(
        lazynwb.tables._source_path_strings(file)[0] for file in source
    )
    source_path_frame = pl.DataFrame(
        {lazynwb.NWB_PATH_COLUMN_NAME: source_paths},
        schema={lazynwb.NWB_PATH_COLUMN_NAME: pl.String},
    )
    try:
        selected_source_paths = set(
            source_path_frame.filter(path_predicate)[
                lazynwb.NWB_PATH_COLUMN_NAME
            ].to_list()
        )
    except Exception as exc:
        logger.debug(
            "Could not evaluate pushed %s predicate against NWB source files; "
            "continuing without file pruning: %r",
            lazynwb.NWB_PATH_COLUMN_NAME,
            exc,
            exc_info=True,
        )
        return source

    filtered_source = tuple(
        file
        for file, source_path in zip(source, source_paths, strict=True)
        if source_path in selected_source_paths
    )
    logger.debug(
        "Pruned %d of %d NWB source files using %d pushed %s predicate "
        "conjunct(s); %d file(s) remain",
        len(source) - len(filtered_source),
        len(source),
        len(path_predicates),
        lazynwb.NWB_PATH_COLUMN_NAME,
        len(filtered_source),
    )
    return filtered_source


def _split_conjunctive_predicate(predicate: pl.Expr) -> list[pl.Expr]:
    if not _is_binary_and_predicate(predicate):
        return [predicate]

    conjuncts: list[pl.Expr] = []
    try:
        children = predicate.meta.pop()
    except BaseException as exc:
        if not _is_polars_panic_exception(exc):
            raise
        logger.debug(
            "Polars panicked while splitting pushed predicate %s; leaving it intact",
            predicate,
        )
        return [predicate]

    for child in children:
        conjuncts.extend(_split_conjunctive_predicate(child))
    return conjuncts


def _is_binary_and_predicate(predicate: pl.Expr) -> bool:
    payload = _predicate_json_payload(predicate)
    if not isinstance(payload, dict):
        return False
    binary_expr = payload.get("BinaryExpr")
    return isinstance(binary_expr, dict) and binary_expr.get("op") == "And"


def _is_standalone_polars_dynamic_predicate(predicate: pl.Expr) -> bool:
    payload = _predicate_json_payload(predicate)
    if not isinstance(payload, dict):
        return False
    display_expr = payload.get("Display")
    if not isinstance(display_expr, dict):
        return False
    fmt_str = display_expr.get("fmt_str")
    return isinstance(fmt_str, str) and fmt_str.startswith(
        _POLARS_DYNAMIC_PREDICATE_TOKEN
    )


def _predicate_contains_polars_dynamic_predicate(predicate: pl.Expr) -> bool:
    predicate_text = str(predicate)
    if _POLARS_DYNAMIC_PREDICATE_TOKEN not in predicate_text:
        return False

    payload = _predicate_json_payload(predicate)
    if payload is None:
        return True
    return _json_payload_contains_polars_dynamic_predicate(payload)


def _predicate_json_payload(predicate: pl.Expr) -> object | None:
    try:
        serialized = predicate.meta.serialize(format="json")
    except BaseException as exc:
        if not _is_polars_panic_exception(exc):
            raise
        logger.debug(
            "Polars panicked while serializing pushed predicate %s to JSON",
            predicate,
        )
        return None

    if not isinstance(serialized, str):
        return None
    try:
        return json.loads(serialized)
    except json.JSONDecodeError:
        logger.debug("Could not decode pushed predicate JSON: %s", serialized)
        return None


def _json_payload_contains_polars_dynamic_predicate(payload: object) -> bool:
    if isinstance(payload, dict):
        display_expr = payload.get("Display")
        if isinstance(display_expr, dict):
            fmt_str = display_expr.get("fmt_str")
            if isinstance(fmt_str, str) and fmt_str.startswith(
                _POLARS_DYNAMIC_PREDICATE_TOKEN
            ):
                return True
        return any(
            _json_payload_contains_polars_dynamic_predicate(value)
            for value in payload.values()
        )
    if isinstance(payload, list):
        return any(
            _json_payload_contains_polars_dynamic_predicate(value) for value in payload
        )
    return False


def _is_polars_panic_exception(exc: BaseException) -> bool:
    return type(exc).__name__ == "PanicException"


def _iter_path_to_row_index_batches(  # noqa: C901
    source: Iterable[lazynwb.types_.PathLike],
    table_path: str,
    batch_size: int,
    n_rows: int | None,
    catalog_snapshots: (
        Mapping[
            str,
            catalog_models._TableSchemaSnapshot,
        ]
        | None
    ) = None,
    ignore_errors: bool = False,
    single_file_batches: bool = False,
) -> Iterator[dict[str, Sequence[int]]]:
    """Yield bounded, source-ordered table row selections."""
    remaining_rows = n_rows
    path_to_row_indices: dict[str, Sequence[int]] = {}
    rows_in_batch = 0
    table_found = False
    missing_table_error: KeyError | None = None
    for file in source:
        if remaining_rows is not None and remaining_rows <= 0:
            break
        try:
            table_length = lazynwb.tables._get_table_length(
                file,
                table_path,
                catalog_snapshot=(
                    catalog_snapshots.get(lazynwb.tables._catalog_snapshot_key(file))
                    if catalog_snapshots is not None
                    else None
                ),
            )
        except KeyError as exc:
            logger.debug("Skipping %r because table %r is missing", file, table_path)
            missing_table_error = exc
            continue
        except Exception:
            if not ignore_errors:
                raise
            logger.debug(
                "Skipping %r after table-length lookup failed for %r",
                file,
                table_path,
                exc_info=True,
            )
            continue

        table_found = True
        source_path = lazynwb.file_io.from_pathlike(file).as_posix()
        if source_path in path_to_row_indices:
            logger.debug(
                "Yielding an early %r batch to preserve duplicate source %r",
                table_path,
                source_path,
            )
            yield path_to_row_indices
            path_to_row_indices = {}
            rows_in_batch = 0

        row_start = 0
        while row_start < table_length:
            if remaining_rows is not None and remaining_rows <= 0:
                break
            available_rows = table_length - row_start
            available_batch_rows = batch_size - rows_in_batch
            row_count = min(available_rows, available_batch_rows)
            if remaining_rows is not None:
                row_count = min(row_count, remaining_rows)
            row_stop = row_start + row_count
            path_to_row_indices[source_path] = list(range(row_start, row_stop))
            rows_in_batch += row_count
            row_start = row_stop
            if remaining_rows is not None:
                remaining_rows -= row_count

            if rows_in_batch == batch_size:
                yield path_to_row_indices
                path_to_row_indices = {}
                rows_in_batch = 0

        if single_file_batches and path_to_row_indices:
            yield path_to_row_indices
            path_to_row_indices = {}
            rows_in_batch = 0

    if path_to_row_indices:
        yield path_to_row_indices
    if not table_found and missing_table_error is not None:
        raise missing_table_error


def _order_frame_by_source(
    df: pl.DataFrame,
    source_paths: Iterable[str],
) -> pl.DataFrame:
    """Restore source order after parallel multi-file materialization."""
    source_paths = tuple(source_paths)
    if len(source_paths) < 2 or df.is_empty():
        return df
    source_order = {
        lazynwb.tables._source_path_strings(path)[0]: index
        for index, path in enumerate(source_paths)
    }
    return df.sort(
        pl.col(lazynwb.NWB_PATH_COLUMN_NAME).replace_strict(
            source_order,
            return_dtype=pl.UInt32,
        ),
        pl.col(lazynwb.TABLE_INDEX_COLUMN_NAME),
    )


def read_nwb(
    source: lazynwb.types_.PathLike | Iterable[lazynwb.types_.PathLike],
    table_path: str,
    raise_on_missing: bool = False,
    ignore_errors: bool = False,
    infer_schema_length: int | None = None,
    exclude_array_columns: bool = False,
    low_memory: bool = False,
    single_file_batches: bool = False,
    schema: polars._typing.SchemaDict | None = None,
    schema_overrides: polars._typing.SchemaDict | None = None,
    disable_progress: bool = False,
) -> pl.DataFrame:
    """
    Read from a common table in one or more local or cloud-hosted NWB files into a DataFrame.

    This function is a wrapper around `scan_nwb` that calls `collect()` on the resulting LazyFrame.

    Parameters
    ----------
    source : str, PathLike, or iterable of these
        Paths to the NWB file(s) to read from. May be hdf5 or zarr.
    table_path : str
        The internal path to the table in the NWB file, e.g. '/intervals/trials' or '/units'
        It is expected that the table path is the same for all files.
    raise_on_missing : bool, default False
        If True, a KeyError will be raised if the table is not found in every file. Otherwise, a
        KeyError is raised only if the table is not found in any file.
    ignore_errors : bool, default False
        If True, other errors will be ignored when reading files (missing table path errors are
        toggled via `raise_on_missing`).
    infer_schema_length : int, None, default None
        The number of files to read to infer the table schema. If None, all files will be read.
    exclude_array_columns : bool, default False
        If True, columns containing list or array-like data will be excluded from the schema
        and any resulting DataFrame.
    low_memory : bool, default False
        If True, the data will be read in smaller chunks to reduce memory usage, at the cost
        of speed.
    single_file_batches : bool, default False
        If True, each materialization batch contains rows from only one NWB file. This can
        reduce peak memory for large list or array columns, at the cost of cross-file
        parallelism.
    schema : dict[str, pl.DataType], default None
        User-defined schema for the table. If None, the schema will be generated using the stored
        dtypes for columns in each file. Conflicts are signalled to the user via a warning.
    schema_overrides : dict[str, pl.DataType], default None
        User-defined schema for a subset of columns, overriding the inferred schema.
    disable_progress : bool, default False
        If True, progress bars will be disabled.

    Returns
    -------
    pl.DataFrame
    """
    return scan_nwb(
        source=source,
        table_path=table_path,
        raise_on_missing=raise_on_missing,
        ignore_errors=ignore_errors,
        infer_schema_length=infer_schema_length,
        exclude_array_columns=exclude_array_columns,
        low_memory=low_memory,
        single_file_batches=single_file_batches,
        schema=schema,
        schema_overrides=schema_overrides,
        disable_progress=disable_progress,
    ).collect()
