from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
import pathlib
import threading

import aiosqlite

import lazynwb._catalog.models as catalog_models
import lazynwb.types_

logger = logging.getLogger(__name__)

_SQLITE_LOCK = threading.RLock()
_SQLITE_TIMEOUT_SECONDS = 30


@dataclasses.dataclass(frozen=True, slots=True)
class _CacheLookupResult:
    """Result of a snapshot cache lookup."""

    snapshot: catalog_models._TableSchemaSnapshot | None
    reason: str

    @property
    def hit(self) -> bool:
        return self.snapshot is not None


@dataclasses.dataclass(frozen=True, slots=True)
class _ParsedMetadataLookupResult:
    """Result of a parsed metadata cache lookup."""

    payload: dict[str, object] | None
    reason: str

    @property
    def hit(self) -> bool:
        return self.payload is not None


class _SQLiteSnapshotCache:
    """Async SQLite cache for catalog snapshots."""

    def __init__(self, path: lazynwb.types_.PathLike) -> None:
        self._path = pathlib.Path(os.fsdecode(path))

    async def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with _SQLITE_LOCK:
            async with aiosqlite.connect(
                self._path,
                timeout=_SQLITE_TIMEOUT_SECONDS,
            ) as db:
                await self._initialize_connection(db)
                await db.commit()

    async def get_table_schema_snapshot(
        self,
        source_identity: catalog_models._SourceIdentity,
        table_path: str,
    ) -> _CacheLookupResult:
        await self.initialize()
        if source_identity.validator_kind == "none":
            logger.debug(
                "table schema cache miss for %s/%s: source has no reliable validator",
                source_identity.source_url,
                table_path,
            )
            return _CacheLookupResult(snapshot=None, reason="unreliable_identity")
        with _SQLITE_LOCK:
            async with aiosqlite.connect(
                self._path,
                timeout=_SQLITE_TIMEOUT_SECONDS,
            ) as db:
                db.row_factory = aiosqlite.Row
                await self._initialize_connection(db)
                source_row = await self._get_source_row(db, source_identity)
                if source_row is None:
                    reason = await self._classify_missing_source(db, source_identity)
                    logger.debug(
                        "table schema cache miss for %s/%s: %s",
                        source_identity.source_url,
                        table_path,
                        reason,
                    )
                    return _CacheLookupResult(snapshot=None, reason=reason)
                snapshot_row = await self._get_table_schema_row(
                    db,
                    source_id=int(source_row["source_id"]),
                    table_path=table_path,
                )
                if snapshot_row is None:
                    logger.debug(
                        "table schema cache miss for %s/%s: no snapshot row",
                        source_identity.source_url,
                        table_path,
                    )
                    return _CacheLookupResult(snapshot=None, reason="snapshot_miss")
                payload_version = int(snapshot_row["payload_version"])
                if (
                    payload_version
                    != catalog_models._TableSchemaSnapshot.PAYLOAD_VERSION
                ):
                    logger.debug(
                        "table schema cache miss for %s/%s: payload version %d != %d",
                        source_identity.source_url,
                        table_path,
                        payload_version,
                        catalog_models._TableSchemaSnapshot.PAYLOAD_VERSION,
                    )
                    return _CacheLookupResult(
                        snapshot=None,
                        reason="payload_version_mismatch",
                    )
                payload = json.loads(str(snapshot_row["payload_json"]))
                snapshot = catalog_models._TableSchemaSnapshot.from_json_dict(payload)
                logger.debug(
                    "table schema cache hit for %s/%s with %d columns",
                    source_identity.source_url,
                    table_path,
                    len(snapshot.columns),
                )
                return _CacheLookupResult(snapshot=snapshot, reason="hit")

    async def put_table_schema_snapshot(
        self,
        snapshot: catalog_models._TableSchemaSnapshot,
    ) -> None:
        await self.initialize()
        source_identity = snapshot.source_identity
        if source_identity.validator_kind == "none":
            logger.debug(
                "skipping table schema cache write for %s/%s: source has no validator",
                source_identity.source_url,
                snapshot.table_path,
            )
            return
        with _SQLITE_LOCK:
            async with aiosqlite.connect(
                self._path,
                timeout=_SQLITE_TIMEOUT_SECONDS,
            ) as db:
                await self._initialize_connection(db)
                source_id = await self._get_or_create_source_id(db, source_identity)
                payload_json = json.dumps(
                    snapshot.to_json_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                now = _utc_now()
                await db.execute(
                    """
                    INSERT INTO table_schema_snapshots (
                        source_id,
                        table_path,
                        payload_version,
                        payload_json,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_id, table_path) DO UPDATE SET
                        payload_version = excluded.payload_version,
                        payload_json = excluded.payload_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        source_id,
                        snapshot.table_path,
                        snapshot.payload_version,
                        payload_json,
                        now,
                        now,
                    ),
                )
                await db.commit()
                logger.debug(
                    "wrote table schema cache snapshot for %s/%s with %d columns",
                    source_identity.source_url,
                    snapshot.table_path,
                    len(snapshot.columns),
                )

    async def get_parsed_hdf5_metadata(
        self,
        source_identity: catalog_models._SourceIdentity,
        *,
        payload_version: int,
        options_key: str,
    ) -> _ParsedMetadataLookupResult:
        await self.initialize()
        if source_identity.validator_kind == "none":
            logger.debug(
                "parsed HDF5 metadata cache miss for %s: source has no reliable validator",
                source_identity.source_url,
            )
            return _ParsedMetadataLookupResult(
                payload=None,
                reason="unreliable_identity",
            )
        with _SQLITE_LOCK:
            async with aiosqlite.connect(
                self._path,
                timeout=_SQLITE_TIMEOUT_SECONDS,
            ) as db:
                db.row_factory = aiosqlite.Row
                await self._initialize_connection(db)
                source_row = await self._get_source_row(db, source_identity)
                if source_row is None:
                    reason = await self._classify_missing_source(db, source_identity)
                    logger.debug(
                        "parsed HDF5 metadata cache miss for %s: %s",
                        source_identity.source_url,
                        reason,
                    )
                    return _ParsedMetadataLookupResult(payload=None, reason=reason)
                metadata_row = await self._get_parsed_hdf5_metadata_row(
                    db,
                    source_id=int(source_row["source_id"]),
                    options_key=options_key,
                )
                if metadata_row is None:
                    logger.debug(
                        "parsed HDF5 metadata cache miss for %s: no payload row",
                        source_identity.source_url,
                    )
                    return _ParsedMetadataLookupResult(
                        payload=None,
                        reason="metadata_miss",
                    )
                cached_version = int(metadata_row["payload_version"])
                if cached_version != payload_version:
                    logger.debug(
                        "parsed HDF5 metadata cache miss for %s: payload version %d != %d",
                        source_identity.source_url,
                        cached_version,
                        payload_version,
                    )
                    return _ParsedMetadataLookupResult(
                        payload=None,
                        reason="payload_version_mismatch",
                    )
                payload = json.loads(str(metadata_row["payload_json"]))
                if not isinstance(payload, dict):
                    logger.debug(
                        "parsed HDF5 metadata cache miss for %s: payload was not object",
                        source_identity.source_url,
                    )
                    return _ParsedMetadataLookupResult(
                        payload=None,
                        reason="invalid_payload",
                    )
                logger.debug(
                    "parsed HDF5 metadata cache hit for %s",
                    source_identity.source_url,
                )
                return _ParsedMetadataLookupResult(payload=payload, reason="hit")

    async def put_parsed_hdf5_metadata(
        self,
        source_identity: catalog_models._SourceIdentity,
        *,
        payload_version: int,
        options_key: str,
        payload: dict[str, object],
    ) -> None:
        await self.initialize()
        if source_identity.validator_kind == "none":
            logger.debug(
                "skipping parsed HDF5 metadata cache write for %s: source has no validator",
                source_identity.source_url,
            )
            return
        with _SQLITE_LOCK:
            async with aiosqlite.connect(
                self._path,
                timeout=_SQLITE_TIMEOUT_SECONDS,
            ) as db:
                await self._initialize_connection(db)
                source_id = await self._get_or_create_source_id(db, source_identity)
                payload_json = json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                now = _utc_now()
                await db.execute(
                    """
                    INSERT INTO parsed_hdf5_metadata (
                        source_id,
                        options_key,
                        payload_version,
                        payload_json,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_id, options_key) DO UPDATE SET
                        payload_version = excluded.payload_version,
                        payload_json = excluded.payload_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        source_id,
                        options_key,
                        payload_version,
                        payload_json,
                        now,
                        now,
                    ),
                )
                await db.commit()
                logger.debug(
                    "wrote parsed HDF5 metadata cache for %s",
                    source_identity.source_url,
                )

    async def _initialize_connection(self, db: aiosqlite.Connection) -> None:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS source_identities (
                source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_url TEXT NOT NULL,
                resolved_url TEXT,
                validator_kind TEXT NOT NULL,
                validator_value TEXT NOT NULL,
                content_length INTEGER,
                version_id TEXT,
                etag TEXT,
                last_modified TEXT,
                in_process_token TEXT,
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                UNIQUE(source_url, resolved_url, validator_kind, validator_value)
            )
            """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS table_schema_snapshots (
                source_id INTEGER NOT NULL,
                table_path TEXT NOT NULL,
                payload_version INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (source_id, table_path),
                FOREIGN KEY (source_id)
                    REFERENCES source_identities(source_id)
                    ON DELETE CASCADE
            )
            """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS parsed_hdf5_metadata (
                source_id INTEGER NOT NULL,
                options_key TEXT NOT NULL,
                payload_version INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (source_id, options_key),
                FOREIGN KEY (source_id)
                    REFERENCES source_identities(source_id)
                    ON DELETE CASCADE
            )
            """)

    async def _get_source_row(
        self,
        db: aiosqlite.Connection,
        source_identity: catalog_models._SourceIdentity,
    ) -> aiosqlite.Row | None:
        cursor = await db.execute(
            """
            SELECT * FROM source_identities
            WHERE source_url = ?
              AND resolved_url IS ?
              AND validator_kind = ?
              AND validator_value = ?
            """,
            (
                source_identity.source_url,
                source_identity.resolved_url,
                source_identity.validator_kind,
                source_identity.validator_value,
            ),
        )
        return await cursor.fetchone()

    async def _get_table_schema_row(
        self,
        db: aiosqlite.Connection,
        source_id: int,
        table_path: str,
    ) -> aiosqlite.Row | None:
        cursor = await db.execute(
            """
            SELECT * FROM table_schema_snapshots
            WHERE source_id = ? AND table_path = ?
            """,
            (source_id, table_path),
        )
        return await cursor.fetchone()

    async def _get_parsed_hdf5_metadata_row(
        self,
        db: aiosqlite.Connection,
        source_id: int,
        options_key: str,
    ) -> aiosqlite.Row | None:
        cursor = await db.execute(
            """
            SELECT * FROM parsed_hdf5_metadata
            WHERE source_id = ? AND options_key = ?
            """,
            (source_id, options_key),
        )
        return await cursor.fetchone()

    async def _classify_missing_source(
        self,
        db: aiosqlite.Connection,
        source_identity: catalog_models._SourceIdentity,
    ) -> str:
        cursor = await db.execute(
            """
            SELECT 1 FROM source_identities
            WHERE source_url = ? AND resolved_url IS ?
            LIMIT 1
            """,
            (source_identity.source_url, source_identity.resolved_url),
        )
        if await cursor.fetchone():
            return "identity_mismatch"
        return "source_miss"

    async def _get_or_create_source_id(
        self,
        db: aiosqlite.Connection,
        source_identity: catalog_models._SourceIdentity,
    ) -> int:
        db.row_factory = aiosqlite.Row
        existing = await self._get_source_row(db, source_identity)
        now = _utc_now()
        if existing is not None:
            source_id = int(existing["source_id"])
            await db.execute(
                "UPDATE source_identities SET last_seen_at = ? WHERE source_id = ?",
                (now, source_id),
            )
            return source_id
        cursor = await db.execute(
            """
            INSERT INTO source_identities (
                source_url,
                resolved_url,
                validator_kind,
                validator_value,
                content_length,
                version_id,
                etag,
                last_modified,
                in_process_token,
                created_at,
                last_seen_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_identity.source_url,
                source_identity.resolved_url,
                source_identity.validator_kind,
                source_identity.validator_value,
                source_identity.content_length,
                source_identity.version_id,
                source_identity.etag,
                source_identity.last_modified,
                source_identity.in_process_token,
                now,
                now,
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return source identity row id")
        return int(cursor.lastrowid)


def _utc_now() -> str:
    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat()


def _default_cache_path() -> pathlib.Path:
    override = os.environ.get("LAZYNWB_CATALOG_CACHE_PATH")
    if override:
        return pathlib.Path(override).expanduser()
    cache_root = pathlib.Path(
        os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")
    )
    return cache_root / "lazynwb" / "catalog.sqlite"
