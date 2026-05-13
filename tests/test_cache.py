from __future__ import annotations

import asyncio
import pathlib
import sqlite3

import lazynwb._cache.sqlite as cache_sqlite
import lazynwb._catalog.models as catalog_models


def test_source_identity_validator_order() -> None:
    identity = catalog_models._SourceIdentity(
        source_url="s3://bucket/file.nwb",
        version_id="version-1",
        etag="etag-1",
        last_modified="2026-01-01T00:00:00+00:00",
        content_length=10,
        in_process_token="token-1",
    )

    assert identity.validator_kind == "version_id"
    assert identity.validator_value == "version-1"

    identity = catalog_models._SourceIdentity(
        source_url="s3://bucket/file.nwb",
        etag="etag-1",
        last_modified="2026-01-01T00:00:00+00:00",
        content_length=10,
    )
    assert identity.validator_kind == "etag"
    assert identity.validator_value == "etag-1"

    identity = catalog_models._SourceIdentity(
        source_url="s3://bucket/file.nwb",
        etag="W/weak-etag",
        last_modified="2026-01-01T00:00:00+00:00",
        content_length=10,
    )
    assert identity.validator_kind == "last_modified_content_length"
    assert identity.validator_value == "2026-01-01T00:00:00+00:00:10"

    identity = catalog_models._SourceIdentity(
        source_url="memory://file.nwb",
        in_process_token="token-1",
    )
    assert identity.validator_kind == "in_process"


def test_sqlite_table_schema_cache_hit(tmp_path: pathlib.Path) -> None:
    cache = cache_sqlite._SQLiteSnapshotCache(tmp_path / "catalog.sqlite")
    snapshot = _snapshot(source_url="s3://bucket/file.nwb", version_id="v1")

    asyncio.run(cache.put_table_schema_snapshot(snapshot))
    result = asyncio.run(
        cache.get_table_schema_snapshot(snapshot.source_identity, snapshot.table_path)
    )

    assert result.hit
    assert result.reason == "hit"
    assert result.snapshot == snapshot


def test_sqlite_table_schema_cache_miss(tmp_path: pathlib.Path) -> None:
    cache = cache_sqlite._SQLiteSnapshotCache(tmp_path / "catalog.sqlite")
    identity = catalog_models._SourceIdentity(
        source_url="s3://bucket/missing.nwb",
        version_id="v1",
    )

    result = asyncio.run(cache.get_table_schema_snapshot(identity, "intervals/trials"))

    assert not result.hit
    assert result.reason == "source_miss"


def test_sqlite_table_schema_cache_identity_mismatch(tmp_path: pathlib.Path) -> None:
    cache = cache_sqlite._SQLiteSnapshotCache(tmp_path / "catalog.sqlite")
    snapshot = _snapshot(source_url="s3://bucket/file.nwb", version_id="v1")
    mismatched_identity = catalog_models._SourceIdentity(
        source_url="s3://bucket/file.nwb",
        version_id="v2",
    )

    asyncio.run(cache.put_table_schema_snapshot(snapshot))
    result = asyncio.run(
        cache.get_table_schema_snapshot(mismatched_identity, snapshot.table_path)
    )

    assert not result.hit
    assert result.reason == "identity_mismatch"


def test_sqlite_table_schema_cache_payload_version_mismatch(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "catalog.sqlite"
    cache = cache_sqlite._SQLiteSnapshotCache(path)
    snapshot = _snapshot(source_url="s3://bucket/file.nwb", version_id="v1")
    asyncio.run(cache.put_table_schema_snapshot(snapshot))

    with sqlite3.connect(path) as db:
        db.execute("UPDATE table_schema_snapshots SET payload_version = 999")
        db.commit()

    result = asyncio.run(
        cache.get_table_schema_snapshot(snapshot.source_identity, snapshot.table_path)
    )

    assert not result.hit
    assert result.reason == "payload_version_mismatch"


def test_sqlite_table_schema_cache_immediate_cold_then_warm(
    tmp_path: pathlib.Path,
) -> None:
    cache = cache_sqlite._SQLiteSnapshotCache(tmp_path / "catalog.sqlite")
    snapshot = _snapshot(source_url="memory://file.nwb", in_process_token="token-1")

    cold = asyncio.run(
        cache.get_table_schema_snapshot(snapshot.source_identity, snapshot.table_path)
    )
    asyncio.run(cache.put_table_schema_snapshot(snapshot))
    warm = asyncio.run(
        cache.get_table_schema_snapshot(snapshot.source_identity, snapshot.table_path)
    )

    assert not cold.hit
    assert cold.reason == "source_miss"
    assert warm.hit
    assert warm.snapshot == snapshot


def test_sqlite_table_schema_cache_skips_unvalidated_sources(
    tmp_path: pathlib.Path,
) -> None:
    cache = cache_sqlite._SQLiteSnapshotCache(tmp_path / "catalog.sqlite")
    snapshot = _snapshot(source_url="memory://file.nwb")

    asyncio.run(cache.put_table_schema_snapshot(snapshot))
    result = asyncio.run(
        cache.get_table_schema_snapshot(snapshot.source_identity, snapshot.table_path)
    )

    assert not result.hit
    assert result.reason == "unreliable_identity"


def _snapshot(
    source_url: str,
    version_id: str | None = None,
    in_process_token: str | None = None,
) -> catalog_models._TableSchemaSnapshot:
    identity = catalog_models._SourceIdentity(
        source_url=source_url,
        version_id=version_id,
        in_process_token=in_process_token,
    )
    return catalog_models._TableSchemaSnapshot(
        source_identity=identity,
        table_path="intervals/trials",
        backend="hdf5",
        table_length=3,
        columns=(
            catalog_models._TableColumnSchema(
                name="start_time",
                table_path="intervals/trials",
                source_path=source_url,
                backend="hdf5",
                dataset=catalog_models._DatasetSchema(
                    path="intervals/trials/start_time",
                    dtype=catalog_models._NeutralDType(
                        kind="numeric",
                        numpy_dtype="<f8",
                        byte_order="<",
                        itemsize=8,
                        detail="float64",
                    ),
                    shape=(3,),
                    ndim=1,
                    is_dataset=True,
                ),
            ),
        ),
    )
