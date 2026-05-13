from __future__ import annotations

import dataclasses

import lazynwb._catalog._schema as catalog_schema


@dataclasses.dataclass(frozen=True, slots=True)
class _ResolvedColumn:
    name: str
    shape: tuple[int, ...] | None
    ndim: int | None
    source_path: str
    table_path: str
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


def test_shared_schema_rules_mark_metadata_tables_and_length() -> None:
    columns = _resolve_columns(
        "general/subject",
        (
            catalog_schema._ColumnFacts("age", (), 0),
            catalog_schema._ColumnFacts("description", (4,), 1),
        ),
    )

    assert all(column.is_metadata_table for column in columns)
    assert catalog_schema._get_table_length(columns) == 1


def test_shared_schema_rules_mark_timeseries_with_rate_and_length() -> None:
    columns = _resolve_columns(
        "processing/behavior/running_speed_with_rate",
        (
            catalog_schema._ColumnFacts("data", (4,), 1),
            catalog_schema._ColumnFacts("starting_time", (), 0),
        ),
    )

    assert all(column.is_timeseries for column in columns)
    assert all(column.is_timeseries_with_rate for column in columns)
    assert all(column.is_timeseries_length_aligned for column in columns)
    assert catalog_schema._get_table_length(columns) == 4


def test_shared_schema_rules_mark_unaligned_timeseries_columns() -> None:
    columns = _resolve_columns(
        "acquisition/run",
        (
            catalog_schema._ColumnFacts("data", (4,), 1),
            catalog_schema._ColumnFacts("timestamps", (4,), 1),
            catalog_schema._ColumnFacts("short_sidecar", (3,), 1),
        ),
    )
    columns_by_name = {column.name: column for column in columns}

    assert columns_by_name["data"].is_timeseries_length_aligned
    assert columns_by_name["timestamps"].is_timeseries_length_aligned
    assert not columns_by_name["short_sidecar"].is_timeseries_length_aligned
    assert catalog_schema._get_table_length(columns) == 4


def test_shared_schema_rules_mark_indexed_columns_and_length() -> None:
    columns = _resolve_columns(
        "units",
        (
            catalog_schema._ColumnFacts("spike_times", (5,), 1),
            catalog_schema._ColumnFacts("spike_times_index", (3,), 1),
            catalog_schema._ColumnFacts("obs_intervals", (6, 2), 2),
            catalog_schema._ColumnFacts("obs_intervals_index", (3,), 1),
        ),
    )
    columns_by_name = {column.name: column for column in columns}

    assert columns_by_name["spike_times"].is_nominally_indexed
    assert columns_by_name["spike_times"].index_column_name == "spike_times_index"
    assert columns_by_name["spike_times"].row_element_shape == ()
    assert columns_by_name["spike_times_index"].is_index_column
    assert columns_by_name["spike_times_index"].data_column_name == "spike_times"
    assert columns_by_name["obs_intervals"].row_element_shape == (2,)
    assert catalog_schema._get_table_length(columns) == 3


def _resolve_columns(
    table_path: str,
    facts: tuple[catalog_schema._ColumnFacts, ...],
) -> tuple[_ResolvedColumn, ...]:
    table_rules = catalog_schema._get_table_schema_rules(table_path, facts)
    return tuple(_resolve_column(table_rules=table_rules, fact=fact) for fact in facts)


def _resolve_column(
    *,
    table_rules: catalog_schema._TableSchemaRules,
    fact: catalog_schema._ColumnFacts,
) -> _ResolvedColumn:
    column_rules = table_rules.column_rules(
        fact.name,
        shape=fact.shape,
        ndim=fact.ndim,
    )
    return _ResolvedColumn(
        name=fact.name,
        shape=fact.shape,
        ndim=fact.ndim,
        source_path="memory://schema-test",
        table_path=table_rules.table_path,
        is_metadata_table=column_rules.is_metadata_table,
        is_timeseries=column_rules.is_timeseries,
        is_timeseries_with_rate=column_rules.is_timeseries_with_rate,
        is_timeseries_length_aligned=column_rules.is_timeseries_length_aligned,
        is_nominally_indexed=column_rules.is_nominally_indexed,
        is_index_column=column_rules.is_index_column,
        is_multidimensional=column_rules.is_multidimensional,
        index_column_name=column_rules.index_column_name,
        data_column_name=column_rules.data_column_name,
        row_element_shape=column_rules.row_element_shape,
    )
