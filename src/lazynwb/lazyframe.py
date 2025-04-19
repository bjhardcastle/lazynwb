# Use python for csv parsing.
from collections.abc import Iterator, Sequence

import numpy as np
import polars as pl

# Used to register a new generator on every instantiation.
from polars.io.plugins import register_io_source

import lazynwb.file_io
import lazynwb.tables


def scan_nwb(
    files: lazynwb.file_io.FileAccessor | Sequence[lazynwb.file_io.FileAccessor],
    table_path: str,
    first_n_files_to_infer_schema: int | None = None,
    include_array_columns: bool = False,
) -> pl.LazyFrame:
    if not isinstance(files, Sequence):
        files = [files]

    if not isinstance(files, Sequence):
        files = [files]
    if not isinstance(files[0], lazynwb.file_io.FileAccessor):
        files = [lazynwb.file_io.FileAccessor(file) for file in files]
    schema = lazynwb.tables._get_table_schema(
        files,
        table_path,
        first_n_files_to_read=first_n_files_to_infer_schema,
        include_array_columns=include_array_columns,
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

        if predicate is not None:
            # - if we have a predicate, we'll fetch the minimal df, apply predicate, then fetch remaining columns in with_columns
            initial_columns = set(predicate.meta.root_names())
        else:
            # - if we don't have a predicate, we'll fetch the full df
            initial_columns = set()
        #TODO if n_rows is not None, don't use all files, or do one file at a time until fulfilled
        #TODO also use batch_size
        df = lazynwb.tables.get_df(
            files,
            search_term=table_path,
            include_column_names=initial_columns or None,
            disable_progress=False,
            as_polars=True,
        )
        if predicate is None:
            yield df[:n_rows] if n_rows is not None and n_rows < df.height else df
        else:
            filtered_df = df.filter(predicate)
            table_row_indices = filtered_df[lazynwb.TABLE_INDEX_COLUMN_NAME]
            if n_rows is not None:
                table_row_indices = table_row_indices[:n_rows]
            i = 0
            while i < len(table_row_indices):
                yield (
                    filtered_df
                    .join(
                        other=(
                            lazynwb.tables.get_df(
                                filtered_df[lazynwb.NWB_PATH_COLUMN_NAME],
                                search_term=table_path,
                                exact_path=True,
                                include_column_names=(set(with_columns) - initial_columns) if with_columns is not None else None,
                                table_row_indices=table_row_indices[
                                    i : min(i + batch_size, len(table_row_indices))
                                ].to_list(),
                                disable_progress=False,
                                as_polars=True,
                            )
                        ),
                        on=[lazynwb.NWB_PATH_COLUMN_NAME, lazynwb.TABLE_INDEX_COLUMN_NAME],
                        how="inner",
                    )
                )
                i += batch_size
    return register_io_source(io_source=source_generator, schema=schema)
