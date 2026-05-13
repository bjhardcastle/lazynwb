from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterable
from typing import Protocol

import lazynwb.utils

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class _ColumnFacts:
    """Backend-neutral DynamicTable facts needed by shared schema policy."""

    name: str
    shape: tuple[int, ...] | None
    ndim: int | None
    is_scalar_metadata: bool | None = None

    @property
    def metadata_scalar(self) -> bool:
        if self.is_scalar_metadata is not None:
            return self.is_scalar_metadata
        return self.ndim == 0


@dataclasses.dataclass(frozen=True, slots=True)
class _TableSchemaRules:
    """Resolved table-level DynamicTable rules shared by all schema readers."""

    table_path: str
    column_names: tuple[str, ...]
    column_name_set: frozenset[str]
    is_metadata_table: bool
    is_timeseries: bool
    is_timeseries_with_rate: bool
    timeseries_len: int | None

    def column_rules(
        self,
        name: str,
        shape: tuple[int, ...] | None,
        ndim: int | None,
    ) -> _ColumnSchemaRules:
        return _get_column_schema_rules(
            table_rules=self,
            name=name,
            shape=shape,
            ndim=ndim,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _ColumnSchemaRules:
    """Resolved column-level DynamicTable schema flags."""

    is_metadata_table: bool
    is_timeseries: bool
    is_timeseries_with_rate: bool
    is_timeseries_length_aligned: bool
    is_nominally_indexed: bool
    is_index_column: bool
    is_multidimensional: bool
    index_column_name: str | None
    data_column_name: str | None
    row_element_shape: tuple[int, ...] | None


class _TableLengthColumn(Protocol):
    name: str
    shape: tuple[int, ...] | None
    ndim: int | None
    source_path: str
    table_path: str
    is_metadata_table: bool
    is_nominally_indexed: bool
    index_column_name: str | None


def _get_table_schema_rules(
    table_path: str,
    columns: Iterable[_ColumnFacts],
) -> _TableSchemaRules:
    columns = tuple(columns)
    column_names = tuple(column.name for column in columns)
    column_name_set = frozenset(column_names)
    normalized_table_path = lazynwb.utils.normalize_internal_file_path(table_path)
    is_timeseries = _is_timeseries(column_name_set)
    is_timeseries_with_rate = _is_timeseries_with_rate(column_name_set)
    is_metadata_table = normalized_table_path == "general" or _is_metadata(
        columns,
        is_timeseries_with_rate=is_timeseries_with_rate,
    )
    timeseries_len = _get_timeseries_data_length(
        columns=columns,
        is_timeseries=is_timeseries,
    )
    logger.debug(
        "resolved DynamicTable schema rules for %s: columns=%d metadata=%s "
        "timeseries=%s timeseries_with_rate=%s timeseries_len=%s",
        normalized_table_path,
        len(columns),
        is_metadata_table,
        is_timeseries,
        is_timeseries_with_rate,
        timeseries_len,
    )
    return _TableSchemaRules(
        table_path=normalized_table_path,
        column_names=column_names,
        column_name_set=column_name_set,
        is_metadata_table=is_metadata_table,
        is_timeseries=is_timeseries,
        is_timeseries_with_rate=is_timeseries_with_rate,
        timeseries_len=timeseries_len,
    )


def _get_column_schema_rules(
    *,
    table_rules: _TableSchemaRules,
    name: str,
    shape: tuple[int, ...] | None,
    ndim: int | None,
) -> _ColumnSchemaRules:
    is_nominally_indexed = _is_nominally_indexed_column(
        name,
        table_rules.column_name_set,
    )
    is_index_column = is_nominally_indexed and name.endswith("_index")
    index_column_name = _get_index_column_name(name, table_rules.column_name_set)
    data_column_name = _get_data_column_name(name, table_rules.column_name_set)
    is_timeseries_length_aligned = _is_timeseries_length_aligned(
        is_timeseries=table_rules.is_timeseries,
        shape=shape,
        timeseries_len=table_rules.timeseries_len,
    )
    rules = _ColumnSchemaRules(
        is_metadata_table=table_rules.is_metadata_table,
        is_timeseries=table_rules.is_timeseries,
        is_timeseries_with_rate=table_rules.is_timeseries_with_rate,
        is_timeseries_length_aligned=is_timeseries_length_aligned,
        is_nominally_indexed=is_nominally_indexed,
        is_index_column=is_index_column,
        is_multidimensional=ndim is not None and ndim > 1,
        index_column_name=index_column_name,
        data_column_name=data_column_name,
        row_element_shape=_get_row_element_shape(
            shape=shape,
            ndim=ndim,
            is_index_column=is_index_column,
        ),
    )
    logger.debug(
        "resolved DynamicTable column rules for %s/%s: metadata=%s timeseries=%s "
        "aligned=%s indexed=%s index_column=%s row_shape=%s",
        table_rules.table_path,
        name,
        rules.is_metadata_table,
        rules.is_timeseries,
        rules.is_timeseries_length_aligned,
        rules.is_nominally_indexed,
        rules.index_column_name,
        rules.row_element_shape,
    )
    return rules


def _get_table_length(
    columns: Iterable[_TableLengthColumn],
) -> int | None:
    columns = tuple(columns)
    columns_by_name = {column.name: column for column in columns}

    if columns and all(column.is_metadata_table for column in columns):
        logger.debug(
            "table length for %s/%s resolved as one metadata row",
            columns[0].source_path,
            columns[0].table_path,
        )
        return 1

    for column in columns:
        if column.is_nominally_indexed:
            index_column_name = column.index_column_name or column.name
            index_column = columns_by_name.get(index_column_name)
            if index_column is not None and index_column.shape is not None:
                logger.debug(
                    "table length for %s/%s resolved from indexed column %r: %d",
                    column.source_path,
                    column.table_path,
                    index_column.name,
                    index_column.shape[0],
                )
                return index_column.shape[0]
        if column.ndim == 1 and column.shape is not None:
            logger.debug(
                "table length for %s/%s resolved from regular column %r: %d",
                column.source_path,
                column.table_path,
                column.name,
                column.shape[0],
            )
            return column.shape[0]
        if column.ndim == 0:
            logger.debug(
                "table length for %s/%s resolved as metadata row",
                column.source_path,
                column.table_path,
            )
            return 1

    for column in columns:
        if column.shape:
            logger.debug(
                "table length for %s/%s resolved from multidimensional column %r: %d",
                column.source_path,
                column.table_path,
                column.name,
                column.shape[0],
            )
            return column.shape[0]

    logger.debug("could not resolve table length from %d columns", len(columns))
    return None


def _is_timeseries(column_names: Iterable[str]) -> bool:
    column_name_set = frozenset(column_names)
    return "data" in column_name_set and (
        "timestamps" in column_name_set or "starting_time" in column_name_set
    )


def _is_timeseries_with_rate(column_names: Iterable[str]) -> bool:
    column_name_set = frozenset(column_names)
    return (
        "data" in column_name_set
        and "starting_time" in column_name_set
        and "timestamps" not in column_name_set
    )


def _is_nominally_indexed_column(
    column_name: str,
    all_column_names: Iterable[str],
) -> bool:
    all_column_names = frozenset(all_column_names)
    if column_name not in all_column_names:
        return False
    if column_name.endswith("_index"):
        return column_name.split("_index")[0] in all_column_names
    return f"{column_name}_index" in all_column_names


def _get_indexed_column_names(column_names: Iterable[str]) -> set[str]:
    column_names = tuple(column_names)
    return {
        column_name
        for column_name in column_names
        if _is_nominally_indexed_column(column_name, column_names)
    }


def _get_index_column_name(
    column_name: str,
    all_column_names: Iterable[str],
) -> str | None:
    all_column_names = frozenset(all_column_names)
    if column_name.endswith("_index"):
        return (
            column_name
            if _is_nominally_indexed_column(column_name, all_column_names)
            else None
        )
    index_column_name = f"{column_name}_index"
    if index_column_name in all_column_names:
        return index_column_name
    return None


def _get_data_column_name(
    column_name: str,
    all_column_names: Iterable[str],
) -> str | None:
    all_column_names = frozenset(all_column_names)
    if not column_name.endswith("_index"):
        return column_name
    data_column_name = column_name.split("_index")[0]
    if data_column_name in all_column_names:
        return data_column_name
    return None


def _is_metadata(
    columns: Iterable[_ColumnFacts],
    *,
    is_timeseries_with_rate: bool,
) -> bool:
    columns = tuple(columns)
    no_multi_dim_columns = all(
        column.ndim is None or column.ndim <= 1 for column in columns
    )
    some_scalar_columns = any(column.metadata_scalar for column in columns)
    return no_multi_dim_columns and some_scalar_columns and not is_timeseries_with_rate


def _get_timeseries_data_length(
    *,
    columns: Iterable[_ColumnFacts],
    is_timeseries: bool,
) -> int | None:
    if not is_timeseries:
        return None
    for column in columns:
        if column.name == "data" and column.shape:
            return column.shape[0]
    return None


def _is_timeseries_length_aligned(
    *,
    is_timeseries: bool,
    shape: tuple[int, ...] | None,
    timeseries_len: int | None,
) -> bool:
    if not is_timeseries or timeseries_len is None or not shape:
        return True
    return shape[0] == timeseries_len


def _get_row_element_shape(
    *,
    shape: tuple[int, ...] | None,
    ndim: int | None,
    is_index_column: bool,
) -> tuple[int, ...] | None:
    if shape is None or ndim is None or is_index_column:
        return None
    if ndim <= 1:
        return ()
    return shape[1:]
