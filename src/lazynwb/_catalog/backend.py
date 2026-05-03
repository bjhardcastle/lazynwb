from __future__ import annotations

import typing

import lazynwb._catalog.models as catalog_models


@typing.runtime_checkable
class _BackendReader(typing.Protocol):
    """Async private reader interface for exact-path catalog access."""

    async def get_source_identity(self) -> catalog_models._SourceIdentity:
        """Return storage identity facts for this reader's source."""

    async def read_table_schema_snapshot(
        self,
        exact_table_path: str,
    ) -> catalog_models._TableSchemaSnapshot:
        """Return catalog facts for one exact, normalized table path."""

    async def close(self) -> None:
        """Release backend resources, if any."""


def _require_exact_normalized_path(exact_table_path: str) -> None:
    if exact_table_path.startswith("/") or exact_table_path in {"", "."}:
        raise ValueError(
            "backend readers require exact normalized internal paths without a leading slash"
        )
