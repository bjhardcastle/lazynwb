from __future__ import annotations

import logging
from typing import Any

import pydantic
import pydantic_settings

logger = logging.getLogger(__name__)


class Config(pydantic_settings.BaseSettings):
    """
    Global configuration for lazynwb behavior.
    """

    model_config = pydantic_settings.SettingsConfigDict(
        env_prefix="",
        populate_by_name=True,
    )

    use_polars: bool = pydantic.Field(
        default=False,
        validation_alias=pydantic.AliasChoices(
            "LAZYNWB_USE_POLARS",
            "LAZYNWB_FILE_IO_USE_POLARS",
        ),
    )
    use_remfile: bool = pydantic.Field(
        default=True,
        validation_alias=pydantic.AliasChoices(
            "LAZYNWB_USE_REMFILE",
            "LAZYNWB_FILE_IO_USE_REMFILE",
        ),
    )
    use_obstore: bool = pydantic.Field(
        default=False,
        validation_alias=pydantic.AliasChoices(
            "LAZYNWB_USE_OBSTORE",
            "LAZYNWB_FILE_IO_USE_OBSTORE",
        ),
    )
    anon: bool | None = pydantic.Field(
        default=None,
        validation_alias=pydantic.AliasChoices(
            "LAZYNWB_ANON",
            "LAZYNWB_FILE_IO_ANON",
        ),
    )
    fsspec_storage_options: dict[str, Any] = pydantic.Field(
        default_factory=lambda: {"anon": False},
        validation_alias=pydantic.AliasChoices(
            "LAZYNWB_FSSPEC_STORAGE_OPTIONS",
            "LAZYNWB_FILE_IO_FSSPEC_STORAGE_OPTIONS",
        ),
    )
    disable_cache: bool = pydantic.Field(
        default=False,
        validation_alias=pydantic.AliasChoices(
            "LAZYNWB_DISABLE_CACHE",
            "LAZYNWB_FILE_IO_DISABLE_CACHE",
        ),
    )


config = Config()


def _resolve_as_polars(as_polars: bool | None) -> bool:
    if as_polars is not None:
        logger.debug(
            "using explicit DataFrame backend setting: as_polars=%s", as_polars
        )
        return as_polars
    logger.debug(
        "using configured DataFrame backend setting: use_polars=%s",
        config.use_polars,
    )
    return config.use_polars


__all__ = ["Config", "config"]
