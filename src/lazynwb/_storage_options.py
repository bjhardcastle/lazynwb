from __future__ import annotations

import logging
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

_S3_REGION_CACHE: dict[str, str] = {}
_S3_REGION_CACHE_LOCK = threading.RLock()
_CUSTOM_S3_ENDPOINT_OPTION_NAMES = frozenset(
    {
        "aws_endpoint",
        "aws_endpoint_url",
        "endpoint",
        "endpoint_url",
    }
)
_CUSTOM_S3_ENDPOINT_NESTED_OPTION_NAMES = frozenset({"config", "client_kwargs"})


def _resolve_anon_setting(
    *,
    anon: bool | None,
    storage_options: Mapping[str, object] | None,
) -> bool:
    """Resolve anonymous-access intent from explicit config and storage options."""
    if anon is not None:
        logger.debug("using top-level anonymous storage setting: %s", anon)
        return anon
    resolved = bool((storage_options or {}).get("anon", False))
    logger.debug("using storage-options anonymous setting: %s", resolved)
    return resolved


def _get_fsspec_storage_options(
    storage_options: Mapping[str, object] | None,
    *,
    anon: bool | None,
) -> dict[str, Any]:
    """Return normalized storage options for UPath/fsspec-backed access."""
    options = dict(storage_options or {})
    resolved_anon = _resolve_anon_setting(anon=anon, storage_options=options)
    options["anon"] = resolved_anon
    logger.debug(
        "normalized fsspec storage options (anon=%s, keys=%s)",
        resolved_anon,
        sorted(str(key) for key in options),
    )
    return options


def _get_obstore_storage_options(
    storage_options: Mapping[str, object] | None,
    *,
    anon: bool | None = None,
    include_aws_region: bool = True,
) -> dict[str, Any]:
    """Return normalized storage options for obstore-backed access."""
    options = dict(storage_options or {})
    resolved_anon = _resolve_anon_setting(anon=anon, storage_options=options)
    options.pop("anon", None)
    if resolved_anon:
        options["skip_signature"] = True
        logger.debug("configured obstore storage for unsigned access")
    if include_aws_region:
        _add_configured_aws_region(options)
    logger.debug(
        "normalized obstore storage options (skip_signature=%s, region=%r, "
        "custom_endpoint=%s, keys=%s)",
        options.get("skip_signature"),
        options.get("region", options.get("aws_region")),
        _has_custom_s3_endpoint(options),
        sorted(str(key) for key in options),
    )
    return options


def _get_obstore_range_reader_storage_options(
    storage_options: Mapping[str, object] | None,
    *,
    s3_bucket: str | None = None,
    discover_bucket_region: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Return normalized obstore options for single-object range reads."""
    options = _get_obstore_storage_options(
        storage_options,
        anon=None,
        include_aws_region=s3_bucket is not None,
    )
    if s3_bucket is not None:
        return _add_discovered_s3_region(
            bucket=s3_bucket,
            storage_options=options,
            discover_bucket_region=discover_bucket_region,
        )
    return options


def _add_configured_aws_region(storage_options: dict[str, Any]) -> None:
    if "region" in storage_options or "aws_region" in storage_options:
        return
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if region:
        storage_options["region"] = region
        logger.debug("using AWS region %s for obstore storage options", region)


def _add_discovered_s3_region(
    bucket: str,
    storage_options: dict[str, Any],
    *,
    discover_bucket_region: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    if not bucket:
        logger.debug("S3 bucket region not discoverable: bucket name is empty")
        return storage_options
    if _has_custom_s3_endpoint(storage_options):
        logger.debug(
            "skipping S3 bucket region discovery for %s because a custom endpoint "
            "is configured",
            bucket,
        )
        return storage_options
    discover = discover_bucket_region or _discover_s3_bucket_region
    region = discover(bucket)
    if region is None:
        logger.debug(
            "S3 bucket region not discoverable for %s; preserving configured region",
            bucket,
        )
        return storage_options
    configured_region = storage_options.get("region", storage_options.get("aws_region"))
    if configured_region is not None and str(configured_region) != region:
        logger.debug(
            "overriding configured S3 region for %s from %r to discovered bucket "
            "region %s",
            bucket,
            configured_region,
            region,
        )
    elif configured_region is None:
        logger.debug("using discovered S3 bucket region for %s: %s", bucket, region)
    else:
        logger.debug("confirmed configured S3 region for %s: %s", bucket, region)
    storage_options.pop("aws_region", None)
    storage_options["region"] = region
    return storage_options


def _has_custom_s3_endpoint(storage_options: Mapping[str, object]) -> bool:
    if os.getenv("AWS_ENDPOINT_URL") or os.getenv("AWS_ENDPOINT"):
        return True
    option_names = {str(key).lower() for key in storage_options}
    if option_names & _CUSTOM_S3_ENDPOINT_OPTION_NAMES:
        return True
    for option_name in _CUSTOM_S3_ENDPOINT_NESTED_OPTION_NAMES:
        nested = storage_options.get(option_name)
        if isinstance(nested, Mapping):
            nested_names = {str(key).lower() for key in nested}
            if nested_names & _CUSTOM_S3_ENDPOINT_OPTION_NAMES:
                return True
    return False


def _discover_s3_bucket_region(bucket: str) -> str | None:
    with _S3_REGION_CACHE_LOCK:
        cached = _S3_REGION_CACHE.get(bucket)
    if cached is not None:
        logger.debug("reusing cached S3 bucket region for %s: %s", bucket, cached)
        return cached
    request = urllib.request.Request(
        f"https://{bucket}.s3.amazonaws.com",
        method="HEAD",
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            region = response.headers.get("x-amz-bucket-region")
    except urllib.error.HTTPError as exc:
        region = exc.headers.get("x-amz-bucket-region")
    except OSError as exc:
        logger.debug("could not discover S3 bucket region for %s: %r", bucket, exc)
        return None
    if region:
        with _S3_REGION_CACHE_LOCK:
            _S3_REGION_CACHE[bucket] = region
        logger.debug("discovered S3 bucket region for %s: %s", bucket, region)
    else:
        logger.debug("S3 bucket region not discoverable for %s: no response header", bucket)
    return region


def _clear_s3_region_cache() -> None:
    with _S3_REGION_CACHE_LOCK:
        logger.debug(
            "clearing S3 bucket region cache with %d entries",
            len(_S3_REGION_CACHE),
        )
        _S3_REGION_CACHE.clear()
