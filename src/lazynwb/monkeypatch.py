from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

import polars as pl

import lazynwb.file_io
import lazynwb.lazyframe

logger = logging.getLogger(__name__)

_PATCH_MARKER = "_lazynwb_monkeypatch"


def patch() -> None:
    """Add ``to_lazyframe()`` helpers to PyNWB/HDMF classes."""
    logger.debug("applying lazynwb PyNWB/HDMF monkeypatches")

    try:
        import hdmf.common.table as hdmf_table
        import pynwb
    except ImportError as exc:
        msg = "lazynwb.monkeypatch requires the optional PyNWB dependencies"
        logger.debug("%s: %s", msg, exc)
        raise ImportError(f"{msg}; install lazynwb[pynwb]") from exc

    _set_method(pynwb.NWBFile, "to_lazyframe", _nwbfile_to_lazyframe)
    _set_method(hdmf_table.DynamicTable, "to_lazyframe", _dynamic_table_to_lazyframe)


def _set_method(cls: type[object], name: str, method: Callable[..., object]) -> None:
    existing = getattr(cls, name, None)
    if getattr(existing, _PATCH_MARKER, False):
        logger.debug("%s.%s is already patched by lazynwb", cls.__name__, name)
        return
    if existing is not None:
        logger.debug("overwriting existing %s.%s with lazynwb monkeypatch", cls.__name__, name)

    setattr(method, _PATCH_MARKER, True)
    setattr(cls, name, method)
    logger.debug("added %s.%s lazynwb monkeypatch", cls.__name__, name)


def _nwbfile_to_lazyframe(
    self: object,
    table_path: str,
    **scan_kwargs: object,
) -> pl.LazyFrame:
    source = _get_container_source(self)
    logger.debug(
        "scanning NWBFile source=%r table_path=%r scan_kwargs=%r",
        source,
        table_path,
        scan_kwargs,
    )
    return lazynwb.lazyframe.scan_nwb(source=source, table_path=table_path, **scan_kwargs)


def _dynamic_table_to_lazyframe(
    self: object,
    table_path: str | None = None,
    **scan_kwargs: object,
) -> pl.LazyFrame:
    source = _get_container_source(self)
    resolved_table_path = table_path or _get_dynamic_table_path(self)
    logger.debug(
        "scanning DynamicTable source=%r table_path=%r scan_kwargs=%r",
        source,
        resolved_table_path,
        scan_kwargs,
    )
    return lazynwb.lazyframe.scan_nwb(
        source=source,
        table_path=resolved_table_path,
        **scan_kwargs,
    )


def _get_container_source(container: object) -> str:
    source_from_io = _get_container_source_from_read_io(container)
    if source_from_io:
        logger.debug(
            "resolved container source for %s from read IO: %r",
            type(container).__name__,
            source_from_io,
        )
        return source_from_io

    source = getattr(container, "container_source", None)
    if source and not _looks_like_object_repr(str(source)):
        logger.debug(
            "resolved container source for %s: %r",
            type(container).__name__,
            source,
        )
        return str(source)

    logger.debug(
        "could not resolve container source for %s; container_source=%r",
        type(container).__name__,
        source,
    )
    raise ValueError(
        "Could not find the NWB file path on this PyNWB/HDMF object. "
        "Expected a lazynwb source path on the read IO, or a usable "
        "'container_source' attribute."
    )


def _get_container_source_from_read_io(container: object) -> str | None:
    for candidate in _iter_container_source_candidates(container):
        source = getattr(candidate, lazynwb.file_io._LAZYNWB_SOURCE_PATH_ATTR, None)
        if source:
            return str(source)
    return None


def _iter_container_source_candidates(container: object) -> Iterable[object]:
    root = _get_root_container(container)
    for candidate_container in (container, root):
        if candidate_container is None:
            continue
        yield candidate_container
        read_io = _get_read_io(candidate_container)
        if read_io is None:
            continue
        yield read_io
        for attr_name in ("_file", "file"):
            file_obj = getattr(read_io, attr_name, None)
            if file_obj is not None:
                yield file_obj


def _get_read_io(container: object) -> object | None:
    get_read_io = getattr(container, "get_read_io", None)
    if callable(get_read_io):
        try:
            read_io = get_read_io()
        except Exception as exc:
            logger.debug(
                "could not call get_read_io() on %s: %r",
                type(container).__name__,
                exc,
                exc_info=True,
            )
        else:
            if read_io is not None:
                return read_io
    return getattr(container, "read_io", None)


def _looks_like_object_repr(source: str) -> bool:
    return source.startswith("<") and " object at 0x" in source and source.endswith(">")


def _get_dynamic_table_path(table: object) -> str:
    root = _get_root_container(table)
    if root is not None:
        candidates = tuple(_iter_container_paths(root, target=table))
        if candidates:
            path = max(candidates, key=lambda value: (value.count("/"), len(value)))
            logger.debug(
                "resolved DynamicTable path by root traversal: table=%r candidates=%r chosen=%r",
                getattr(table, "name", None),
                candidates,
                path,
            )
            return path

    path = _get_parent_chain_path(table)
    if path is not None:
        logger.debug(
            "resolved DynamicTable path by parent chain: table=%r path=%r",
            getattr(table, "name", None),
            path,
        )
        return path

    logger.debug("could not resolve DynamicTable path for table=%r", getattr(table, "name", None))
    raise ValueError(
        "Could not find the DynamicTable path on this HDMF object. "
        "Pass table_path=... to to_lazyframe()."
    )


def _get_root_container(container: object) -> object | None:
    current = container
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parent = getattr(current, "parent", None)
        if parent is None:
            return current
        current = parent
    return None


def _get_parent_chain_path(container: object) -> str | None:
    names: list[str] = []
    current = container
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = getattr(current, "name", None)
        if name and name != "root":
            names.append(str(name))
        current = getattr(current, "parent", None)

    if not names:
        return None
    return "/" + "/".join(reversed(names))


def _iter_container_paths(root: object, target: object) -> Iterable[str]:
    yield from _iter_value_paths(root, target=target, prefix="", seen=set())


def _iter_value_paths(
    value: object,
    target: object,
    prefix: str,
    seen: set[int],
) -> Iterable[str]:
    if value is target and prefix:
        yield _normalize_internal_path(prefix)
        return

    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)

    if isinstance(value, dict):
        for key, child in value.items():
            yield from _iter_value_paths(
                child,
                target=target,
                prefix=_join_internal_path(prefix, str(key)),
                seen=seen.copy(),
            )
        return

    fields = getattr(value, "fields", None)
    if isinstance(fields, dict):
        for key, child in fields.items():
            if _is_metadata_field(key):
                continue
            yield from _iter_value_paths(
                child,
                target=target,
                prefix=_join_internal_path(prefix, str(key)),
                seen=seen.copy(),
            )


def _is_metadata_field(key: str) -> bool:
    return key in {
        "colnames",
        "columns",
        "description",
        "file_create_date",
        "identifier",
        "id",
        "session_description",
        "session_start_time",
        "timestamps_reference_time",
    }


def _join_internal_path(prefix: str, child: str) -> str:
    return f"{prefix.rstrip('/')}/{child.strip('/')}"


def _normalize_internal_path(path: str) -> str:
    return "/" + path.strip("/")


patch()
