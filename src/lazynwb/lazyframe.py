from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence

import polars as pl
from polars.io.plugins import register_io_source

import lazynwb.file_io
import lazynwb.tables

logger = logging.getLogger(__name__)


def scan_nwb(
    files: lazynwb.file_io.FileAccessor | Sequence[lazynwb.file_io.FileAccessor],
    table_path: str,
    first_n_files_to_infer_schema: int | None = 1,
    include_array_columns: bool = True,
) -> pl.LazyFrame:
    if not isinstance(files, Sequence):
        files = [files]
    if not isinstance(files[0], lazynwb.file_io.FileAccessor):
        files = [lazynwb.file_io.FileAccessor(file) for file in files]

    logger.debug(f"Fetching schema for {table_path!r} from {len(files)} files")
    schema = lazynwb.tables._get_table_schema(
        files,
        table_path,
        first_n_files_to_read=first_n_files_to_infer_schema,
        include_array_columns=include_array_columns,
        include_internal_columns=True,
    )

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        """
        Generator function that creates the source.
        This function will be registered as IO source.
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
                sum_rows += lazynwb.tables._get_table_length(file, table_path)
                if sum_rows >= n_rows:
                    break
            filtered_files = files[: idx + 1]
            logger.debug(f"Limiting files to {len(files)} based on n_rows={n_rows}")
        else:
            filtered_files = files
        df = lazynwb.tables.get_df(
            filtered_files,
            search_term=table_path,
            exact_path=True,
            include_column_names=initial_columns or None,
            disable_progress=False,
            as_polars=True,
            exclude_array_columns=(
                False
                if initial_columns
                else (not include_array_columns)
                # if specific columns were requested, they will be returned regardless of whether or
                # not they're array columns. Otherwise, use the user setting for include_array_columns
            ),
        )

        if predicate is None:
            logger.debug(
                f"Yielding {table_path!r} df with {df.height} rows and {df.width} columns"
            )
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
                    filtered_df.join(
                        other=(
                            lazynwb.tables.get_df(
                                nwb_data_sources=nwb_path_to_row_indices.keys(),
                                search_term=table_path,
                                exact_path=True,
                                include_column_names=include_column_names,
                                exclude_array_columns=False,
                                nwb_path_to_row_indices=nwb_path_to_row_indices,
                                disable_progress=False,
                                use_process_pool=any(
                                    isinstance(schema[name], pl.List)
                                    for name in include_column_names
                                ),
                                as_polars=True,
                            )
                        ),
                        on=[
                            lazynwb.NWB_PATH_COLUMN_NAME,
                            lazynwb.TABLE_INDEX_COLUMN_NAME,
                        ],
                        how="inner",
                    )
                )
                i += batch_size

    return register_io_source(io_source=source_generator, schema=schema)
