from __future__ import annotations

import enum
import typing


class _ExitCode(enum.IntEnum):
    OK = 0
    INTERNAL_ERROR = 1
    USAGE_ERROR = 2
    VALIDATION_ERROR = 3


class _ErrorCode(str, enum.Enum):
    INTERNAL_ERROR = "internal_error"
    SOURCE_PATH_NOT_FOUND = "source_path_not_found"
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
