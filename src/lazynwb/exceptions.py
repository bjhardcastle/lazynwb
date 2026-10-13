from __future__ import annotations

import dataclasses


class ColumnError(KeyError):
    """Requested column name is not in table group"""

    pass


class InternalPathError(KeyError):
    """Requested internal path is not in file"""

    pass


@dataclasses.dataclass(slots=True)
class UnsupportedHDF5LayoutError(RuntimeError):
    """A range-backed HDF5 read cannot safely decode the requested layout."""

    source_url: str
    table_path: str
    columns: tuple[str, ...]
    reasons: tuple[str, ...]

    def __str__(self) -> str:
        details = "; ".join(
            f"{column}: {reason}"
            for column, reason in zip(self.columns, self.reasons, strict=True)
        )
        return (
            f"Unsupported range-backed HDF5 layout for {self.source_url!r} at "
            f"{self.table_path!r}: {details}"
        )
