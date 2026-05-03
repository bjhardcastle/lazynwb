from __future__ import annotations

import enum
import typing


class _ExitCode(enum.IntEnum):
    OK = 0
    INTERNAL_ERROR = 1
    USAGE_ERROR = 2
    VALIDATION_ERROR = 3


class _ErrorCode(str, enum.Enum):
    CONFIG_ALREADY_EXISTS = "config_already_exists"
    CONFIG_INVALID = "config_invalid"
    CONFIG_NOT_FOUND = "config_not_found"
    CONFIG_PARSE_ERROR = "config_parse_error"
    INTERNAL_ERROR = "internal_error"
    SOURCE_CONFIG_CONFLICT = "source_config_conflict"
    SOURCE_CONFIG_INCOMPLETE = "source_config_incomplete"
    SOURCE_DANDI_NO_ASSETS = "source_dandi_no_assets"
    SOURCE_DANDI_URL_UNAVAILABLE = "source_dandi_url_unavailable"
    SOURCE_NOT_CONFIGURED = "source_not_configured"
    SOURCE_PATHS_UNAVAILABLE = "source_paths_unavailable"
    SOURCE_PATH_NOT_FOUND = "source_path_not_found"
    SOURCE_ROOT_NOT_FOUND = "source_root_not_found"
    TABLES_NOT_FOUND = "tables_not_found"
    USAGE_ERROR = "usage_error"


class _CLIError(Exception):
    def __init__(
        self,
        *,
        code: _ErrorCode,
        message: str,
        exit_code: _ExitCode,
        details: dict[str, typing.Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code
        self.details = details or {}

    def _to_json_object(self) -> dict[str, typing.Any]:
        return {
            "error": {
                "code": self.code.value,
                "details": self.details,
                "message": self.message,
            }
        }
