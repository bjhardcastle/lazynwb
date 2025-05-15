from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator

import npc_io
import polars as pl
import polars._typing
import polars.io.plugins

import lazynwb.file_io
import lazynwb.tables

logger = logging.getLogger(__name__)


def scan_nwb(
    source: (
        npc_io.PathLike
        | lazynwb.file_io.FileAccessor
        | Iterable[npc_io.PathLike | lazynwb.file_io.FileAccessor]
    ),
    table_path: str,
    raise_on_missing: bool = False,
    ignore_errors: bool = False,
    infer_schema_length: int | None = None,
    exclude_array_columns: bool = False,
    low_memory: bool = False,
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
    source : str, PathLike, lazynwb.file_io.FileAccessor, or iterable of these
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
        If True, columns containing list or array-like data will be excluded from the schema and any
        resulting DataFrame.
    low_memory : bool, default False
        If True, the data will be read in smaller chunks to reduce memory usage, at the cost of speed.
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
        source = [source]

    files: list[lazynwb.file_io.FileAccessor] = []
    for f in source:  # type: ignore[union-attr]
        if isinstance(f, lazynwb.file_io.FileAccessor):
            files.append(f)
        else:
            files.append(lazynwb.file_io.FileAccessor(f))  # type: ignore[arg-type]

    if not schema:
        schema = lazynwb.tables._get_table_schema(
            files=files,
            table_path=table_path,
            first_n_files_to_infer_schema=infer_schema_length,
            exclude_array_columns=exclude_array_columns,
            exclude_internal_columns=False,
            raise_on_missing=raise_on_missing,
        )
    schema = pl.Schema(schema) | pl.Schema(
        schema_overrides or {}
    )  # create new object to avoid mutating the original schema

    def _apply_schema(
        df: pl.DataFrame,
        schema: polars._typing.SchemaDict,
    ) -> pl.DataFrame:
        """
        Apply the schema to the DataFrame, converting columns to the specified types.
        """
        return pl.DataFrame(
            df, schema={column: schema[column] for column in df.columns}
        )

    def source_generator(
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
        3) join with values from the remaining columns in with_columns, by reading only the relevant files/rows.

        Without a predicate, we fetch all data for all columns.
        """
        if batch_size is None:
            batch_size = 1_000
            logger.debug(
                f"Batch size not specified: using default of {batch_size} rows per batch"
            )
        else:
            logger.debug(f"Batch size set to {batch_size} rows per batch")

        if predicate is not None:
            # - if we have a predicate, we'll fetch the minimal df, apply predicate, then fetch remaining columns in with_columns
            initial_columns = predicate.meta.root_names()
            logger.debug(
                f"Predicate specified: fetching initial columns in {table_path!r}: {initial_columns}"
            )
        else:
            # - if we don't have a predicate, we'll fetch all required columns in the initial df
            initial_columns = with_columns or []
            logger.debug(
                f"Predicate not specified: fetching all requested columns in {table_path!r} ({initial_columns})"
            )

        # TODO use batch_size
        if n_rows and len(files) > 1:
            sum_rows = 0
            for idx, file in enumerate(files):
                try:
                    sum_rows += lazynwb.tables._get_table_length(file, table_path)
                except KeyError:
                    continue
                if sum_rows >= n_rows:
                    break
            filtered_files = files[: idx + 1]
            logger.debug(f"Limiting files to {len(files)} based on n_rows={n_rows}")
        else:
            filtered_files = files
        df = lazynwb.tables.get_df(
            nwb_data_sources=filtered_files,
            search_term=table_path,
            exact_path=True,
            include_column_names=initial_columns or None,
            disable_progress=disable_progress,
            ignore_errors=ignore_errors,
            as_polars=True,
            exclude_array_columns=(
                False
                if initial_columns
                else exclude_array_columns
                # if array columns were requested specifically, they will be returned regardless of
                # this setting. Otherwise, use the user setting.
            ),
            low_memory=low_memory,
        )

        if predicate is None:
            logger.debug(
                f"Yielding {table_path!r} df with {df.height} rows and {df.width} columns"
            )

            df = _apply_schema(df, schema=schema)
            yield df[:n_rows] if n_rows is not None and n_rows < df.height else df

        else:
            filtered_df = df.filter(predicate)
            logger.debug(
                f"Initial {table_path!r} df filtered with predicate: {df.height} rows reduced to {filtered_df.height}"
            )
            if with_columns:
                include_column_names = set(with_columns) - set(initial_columns)
            else:
                include_column_names = set(schema.keys()) - set(initial_columns)
            logger.debug(
                f"Fetching additional columns from {table_path!r}: {sorted(include_column_names)}"
            )
            if not n_rows:
                n_rows = len(filtered_df)
            i = 0
            while i < n_rows:
                nwb_path_to_row_indices = lazynwb.tables._get_path_to_row_indices(
                    filtered_df[i : min(i + batch_size, n_rows)]
                )
                yield (
                    _apply_schema(
                        filtered_df.join(
                            other=(
                                lazynwb.tables.get_df(
                                    nwb_data_sources=nwb_path_to_row_indices.keys(),
                                    search_term=table_path,
                                    exact_path=True,
                                    include_column_names=include_column_names,
                                    nwb_path_to_row_indices=nwb_path_to_row_indices,
                                    disable_progress=disable_progress,
                                    use_process_pool=False,  # no speed gain, cannot use from top-level of scripts
                                    as_polars=True,
                                    ignore_errors=ignore_errors,
                                    low_memory=low_memory,
                                )
                            ),
                            on=[
                                lazynwb.NWB_PATH_COLUMN_NAME,
                                lazynwb.TABLE_PATH_COLUMN_NAME,
                                lazynwb.TABLE_INDEX_COLUMN_NAME,
                            ],
                            how="inner",
                        ),
                        schema=schema,
                    )
                )
                i += batch_size

    return polars.io.plugins.register_io_source(
        io_source=source_generator, schema=schema
    )
