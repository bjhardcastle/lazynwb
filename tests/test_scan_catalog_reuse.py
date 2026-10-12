"""Regression coverage for catalog work repeated by bounded scan reads."""

import pathlib

import polars as pl
import pytest

import lazynwb._catalog.models as catalog_models
import lazynwb.lazyframe as lazyframe
import lazynwb.tables as tables
import lazynwb.types_ as types_


def test_scan_retains_zarr_snapshot_between_length_and_materialization(
    local_zarr_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Length planning and column reads must share one catalog snapshot."""
    reads = []
    original = tables._get_fast_catalog_snapshot_if_available

    def _counted_snapshot(
        file_path: types_.PathLike, exact_table_path: str
    ) -> catalog_models._TableSchemaSnapshot | None:
        """Count real catalog reads made while collecting the lazy scan."""
        reads.append((file_path, exact_table_path))
        return original(file_path, exact_table_path)

    monkeypatch.setattr(
        tables, "_get_fast_catalog_snapshot_if_available", _counted_snapshot
    )
    query = lazyframe.scan_nwb(
        local_zarr_path, "/intervals/trials", disable_progress=True
    )
    result = query.select(pl.col("stop_time") - pl.col("start_time")).collect()
    assert result.height > 0
    assert len(reads) == 1
