from __future__ import annotations

import logging

import lazynwb._catalog.backend as catalog_backend
import lazynwb._catalog.models as catalog_models
import lazynwb.types_

logger = logging.getLogger(__name__)


class _ZarrBackendReader:
    """Future explicit Zarr catalog reader.

    TODO: implement targeted Zarr metadata loading with consolidated `.zmetadata`
    preference, then non-consolidated metadata fallback. Until that slice lands,
    local Zarr behavior remains covered by `_AccessorBackendReader`.
    """

    def __init__(self, source: lazynwb.types_.PathLike) -> None:
        self._source = source

    async def get_source_identity(self) -> catalog_models._SourceIdentity:
        return catalog_models._SourceIdentity(
            source_url=str(self._source),
            in_process_token=f"zarr-skeleton:{id(self)}",
        )

    async def read_table_schema_snapshot(
        self,
        exact_table_path: str,
    ) -> catalog_models._TableSchemaSnapshot:
        catalog_backend._require_exact_normalized_path(exact_table_path)
        logger.debug(
            "explicit Zarr backend reader is not implemented yet for %s/%s",
            self._source,
            exact_table_path,
        )
        raise NotImplementedError(
            "_ZarrBackendReader is a private skeleton; use _AccessorBackendReader "
            "until the Zarr catalog slice is implemented"
        )

    async def close(self) -> None:
        logger.debug("closing Zarr backend reader skeleton for %s", self._source)
