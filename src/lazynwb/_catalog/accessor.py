from __future__ import annotations

import logging
import pathlib
import time

import lazynwb._catalog.backend as catalog_backend
import lazynwb._catalog.models as catalog_models
import lazynwb.file_io
import lazynwb.table_metadata
import lazynwb.types_

logger = logging.getLogger(__name__)


class _AccessorBackendReader:
    """Reference catalog reader backed by the existing FileAccessor behavior."""

    def __init__(self, source: lazynwb.types_.PathLike) -> None:
        self._file = lazynwb.file_io._get_accessor(source)
        self._source_identity = self._build_source_identity()
        logger.debug(
            "initialized accessor backend reader for %s with backend %s",
            self._source_identity.source_url,
            self._file._hdmf_backend.value,
        )

    async def get_source_identity(self) -> catalog_models._SourceIdentity:
        return self._source_identity

    async def read_table_schema_snapshot(
        self,
        exact_table_path: str,
    ) -> catalog_models._TableSchemaSnapshot:
        catalog_backend._require_exact_normalized_path(exact_table_path)
        t0 = time.time()
        columns = lazynwb.table_metadata.get_table_column_metadata(
            self._file,
            exact_table_path,
        )
        column_schemas = tuple(
            catalog_models._column_from_raw_metadata(column) for column in columns
        )
        try:
            table_length = lazynwb.table_metadata.get_table_length_from_metadata(
                columns
            )
        except Exception:
            logger.debug(
                "could not determine table length for accessor catalog snapshot %s/%s",
                self._source_identity.source_url,
                exact_table_path,
                exc_info=True,
            )
            table_length = None
        snapshot = catalog_models._TableSchemaSnapshot(
            source_identity=self._source_identity,
            table_path=exact_table_path,
            backend=self._file._hdmf_backend.value,
            columns=column_schemas,
            table_length=table_length,
        )
        logger.debug(
            "built accessor catalog snapshot for %s/%s with %d columns in %.2f s",
            self._source_identity.source_url,
            exact_table_path,
            len(snapshot.columns),
            time.time() - t0,
        )
        return snapshot

    async def close(self) -> None:
        logger.debug(
            "accessor backend reader close requested for %s; FileAccessor cache owns lifetime",
            self._source_identity.source_url,
        )

    def _build_source_identity(self) -> catalog_models._SourceIdentity:
        path = pathlib.Path(self._file._path.as_posix())
        if not self._file._path.protocol and path.exists():
            return catalog_models._source_identity_from_local_path(path)
        return catalog_models._SourceIdentity(
            source_url=self._file._path.as_posix(),
            in_process_token=f"accessor:{id(self._file._accessor)}",
        )
