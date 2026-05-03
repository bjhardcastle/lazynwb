from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import polars as pl
import polars._typing
import polars.datatypes.convert

import lazynwb._catalog.models as catalog_models
import lazynwb.table_metadata

logger = logging.getLogger(__name__)


def _snapshot_to_polars_schema(
    snapshot: catalog_models._TableSchemaSnapshot,
) -> pl.Schema:
    """Derive the public per-file Polars schema from a catalog snapshot."""
    file_schema = pl.Schema()
    for column in snapshot.columns:
        if column.is_index_column:
            continue
        if column.is_group:
            continue
        if column.name == "starting_time" and column.is_timeseries_with_rate:
            file_schema["timestamps"] = pl.Float64
            continue
        if column.is_timeseries and not column.is_timeseries_length_aligned:
            logger.debug(
                "skipping column %r with shape %s from catalog TimeSeries table: "
                "length does not match data length",
                column.name,
                column.shape,
            )
            continue
        file_schema[column.name] = _get_polars_dtype(column, snapshot.columns)
    logger.debug(
        "derived Polars schema from catalog snapshot %s/%s: %s",
        snapshot.source_identity.source_url,
        snapshot.table_path,
        file_schema,
    )
    return file_schema


def _get_polars_dtype(
    column: catalog_models._TableColumnSchema,
    all_columns: Iterable[catalog_models._TableColumnSchema],
) -> polars._typing.PolarsDataType:
    dtype = _neutral_dtype_to_polars_base(column.dtype)
    if column.is_metadata_table and column.shape:
        return pl.List(dtype)
    if column.ndim is not None and column.ndim > 1 and column.shape is not None:
        dtype = pl.Array(dtype, shape=column.shape[1:])
    if column.is_nominally_indexed:
        all_column_names = [raw_column.name for raw_column in all_columns]
        index_cols = [
            column_name
            for column_name in lazynwb.table_metadata._get_indexed_column_names(
                all_column_names
            )
            if column_name.startswith(column.name) and column_name.endswith("_index")
        ]
        for _ in index_cols:
            dtype = pl.List(dtype)
    return dtype


def _neutral_dtype_to_polars_base(
    dtype: catalog_models._NeutralDType,
) -> polars._typing.PolarsDataType:
    if dtype.kind in {"object", "reference", "string", "vlen_string"}:
        return pl.String
    if dtype.kind == "bool":
        return pl.Boolean
    if dtype.kind in {"compound", "opaque", "unknown"}:
        return pl.Object
    if dtype.kind == "array" and dtype.element_numpy_dtype is not None:
        np_dtype = np.dtype(dtype.element_numpy_dtype)
        return polars.datatypes.convert.numpy_char_code_to_dtype(np_dtype)
    if dtype.numpy_dtype is None:
        return pl.Object
    np_dtype = np.dtype(dtype.numpy_dtype)
    return polars.datatypes.convert.numpy_char_code_to_dtype(np_dtype)
