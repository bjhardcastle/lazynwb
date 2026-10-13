"""Private DANDI:001637 sample registry for opt-in integration tests.

The assets below are fixed to the two smallest NWB assets currently present in
DANDI:001637 draft. They are still multi-GB remote HDF5 files, so framework
tests should limit themselves to URL resolution or explicitly bounded reads.
Issue-specific table, TimeSeries, or metadata verification belongs in later
integration tests that opt into this registry.
"""

from __future__ import annotations

import dataclasses
import logging
import time
import urllib.parse

import lazynwb.dandi as dandi

_LOGGER_NAME = "tests.dandi_sample"
_logger = logging.getLogger(_LOGGER_NAME)

_DANDI_SAMPLE_ENV_VAR = "LAZYNWB_DANDI_INTEGRATION_TESTS"
_DANDI_SAMPLE_OPTION = "--run-dandi-integration"
_DANDI_SAMPLE_DANDISET_ID = "001637"
_DANDI_SAMPLE_VERSION = "draft"
_DANDI_SAMPLE_MAX_BOUNDED_READ_BYTES = 1024 * 1024


@dataclasses.dataclass(frozen=True, slots=True)
class _DandiSampleAsset:
    asset_id: str
    path: str
    size_bytes: int
    description: str
    max_bounded_read_bytes: int = _DANDI_SAMPLE_MAX_BOUNDED_READ_BYTES


@dataclasses.dataclass(frozen=True, slots=True)
class _DandiSampleResolvedAsset:
    asset: _DandiSampleAsset
    source_url: str


_DANDI_SAMPLE_ASSETS = (
    _DandiSampleAsset(
        asset_id="ca248278-e1b2-4896-ad1c-900e4506cd04",
        path=(
            "sub-830849/"
            "sub-830849_ses-ecephys-830849-2026-03-07-09-48-16_ecephys.nwb"
        ),
        size_bytes=10_313_437_437,
        description="Smallest fixed NWB asset in the DANDI:001637 draft sample.",
    ),
    _DandiSampleAsset(
        asset_id="1e37bc82-fd23-4cb5-a253-e794cea932ba",
        path=(
            "sub-830795/"
            "sub-830795_ses-ecephys-830795-2026-02-25-16-03-31_ecephys.nwb"
        ),
        size_bytes=10_314_961_958,
        description="Second-smallest fixed NWB asset in the DANDI:001637 draft sample.",
    ),
)


def _resolve_sample_asset_urls() -> tuple[_DandiSampleResolvedAsset, ...]:
    start_time = time.perf_counter()
    _logger.debug(
        "starting DANDI sample URL resolution: dandiset_id=%s version=%s "
        "asset_count=%d asset_ids=%s asset_paths=%s max_bounded_read_bytes=%d",
        _DANDI_SAMPLE_DANDISET_ID,
        _DANDI_SAMPLE_VERSION,
        len(_DANDI_SAMPLE_ASSETS),
        _sample_asset_ids(),
        _sample_asset_paths(),
        _DANDI_SAMPLE_MAX_BOUNDED_READ_BYTES,
    )
    resolved_assets = tuple(_resolve_sample_asset_url(asset) for asset in _DANDI_SAMPLE_ASSETS)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    _logger.debug(
        "finished DANDI sample URL resolution: dandiset_id=%s version=%s "
        "asset_count=%d elapsed_ms=%.3f",
        _DANDI_SAMPLE_DANDISET_ID,
        _DANDI_SAMPLE_VERSION,
        len(resolved_assets),
        elapsed_ms,
    )
    return resolved_assets


def _resolve_sample_asset_url(asset: _DandiSampleAsset) -> _DandiSampleResolvedAsset:
    start_time = time.perf_counter()
    source_url = dandi._get_asset_s3_url(
        _DANDI_SAMPLE_DANDISET_ID,
        asset.asset_id,
        _DANDI_SAMPLE_VERSION,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    _logger.debug(
        "resolved DANDI sample source URL: dandiset_id=%s version=%s "
        "asset_id=%s asset_path=%s asset_size_bytes=%d "
        "max_bounded_read_bytes=%d source_url=%s elapsed_ms=%.3f",
        _DANDI_SAMPLE_DANDISET_ID,
        _DANDI_SAMPLE_VERSION,
        asset.asset_id,
        asset.path,
        asset.size_bytes,
        asset.max_bounded_read_bytes,
        _sanitize_source_url_for_logging(source_url),
        elapsed_ms,
    )
    return _DandiSampleResolvedAsset(asset=asset, source_url=source_url)


def _log_resolved_sample_assets(
    resolved_assets: tuple[_DandiSampleResolvedAsset, ...],
    *,
    cache_scope: str,
) -> None:
    for resolved_asset in resolved_assets:
        asset = resolved_asset.asset
        _logger.debug(
            "using cached DANDI sample source URL: dandiset_id=%s version=%s "
            "asset_id=%s asset_path=%s asset_size_bytes=%d "
            "max_bounded_read_bytes=%d source_url=%s cache_scope=%s",
            _DANDI_SAMPLE_DANDISET_ID,
            _DANDI_SAMPLE_VERSION,
            asset.asset_id,
            asset.path,
            asset.size_bytes,
            asset.max_bounded_read_bytes,
            _sanitize_source_url_for_logging(resolved_asset.source_url),
            cache_scope,
        )


def _sanitize_source_url_for_logging(source_url: str) -> str:
    parsed = urllib.parse.urlsplit(source_url)
    netloc = parsed.netloc
    if parsed.username is not None or parsed.password is not None:
        host = parsed.hostname or ""
        port = "" if parsed.port is None else f":{parsed.port}"
        netloc = f"<redacted>@{host}{port}"
    query = "<redacted>" if parsed.query else ""
    fragment = "<redacted>" if parsed.fragment else ""
    return urllib.parse.urlunsplit(
        (parsed.scheme, netloc, parsed.path, query, fragment)
    )


def _sample_asset_ids() -> tuple[str, ...]:
    return tuple(asset.asset_id for asset in _DANDI_SAMPLE_ASSETS)


def _sample_asset_paths() -> tuple[str, ...]:
    return tuple(asset.path for asset in _DANDI_SAMPLE_ASSETS)
