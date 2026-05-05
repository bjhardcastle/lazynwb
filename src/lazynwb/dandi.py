"""DANDI Archive integration for lazynwb."""

from __future__ import annotations

import collections.abc
import concurrent.futures
import logging
import time
import typing
import urllib.parse

import requests

import lazynwb.file_io
import lazynwb.utils

logger = logging.getLogger(__name__)

DANDI_API_BASE = "https://api.dandiarchive.org/api"


def _get_session() -> requests.Session:
    """Create a requests session with automatic error handling."""
    session = requests.Session()
    session.hooks = {"response": lambda r, *args, **kwargs: r.raise_for_status()}
    return session


def _get_most_recent_dandiset_version(dandiset_id: str) -> str:
    """Resolve the default version string for a dandiset."""
    return _resolve_dandiset_version(dandiset_id, version=None)


def _resolve_dandiset_version(
    dandiset_id: str,
    version: str | None,
    *,
    session: requests.Session | None = None,
) -> str:
    if version is not None:
        logger.debug(
            "using explicit DANDI dandiset version: dandiset_id=%s version=%s",
            dandiset_id,
            version,
        )
        return version

    active_session = _get_session() if session is None else session
    metadata = _get_dandiset_metadata(session=active_session, dandiset_id=dandiset_id)
    published_version = _version_value(metadata.get("most_recent_published_version"))
    if published_version is not None:
        logger.debug(
            "using most recent published DANDI dandiset version: "
            "dandiset_id=%s version=%s",
            dandiset_id,
            published_version,
        )
        return published_version

    draft_version = _version_value(metadata.get("draft_version"))
    if draft_version is not None:
        logger.debug(
            "DANDI dandiset has no published version; falling back to draft: "
            "dandiset_id=%s version=%s",
            dandiset_id,
            draft_version,
        )
        return draft_version

    logger.debug(
        "DANDI dandiset metadata has no published or draft version: dandiset_id=%s",
        dandiset_id,
    )
    raise ValueError(
        f"No published or draft DANDI version found for dandiset {dandiset_id}"
    )


def _get_dandiset_metadata(
    *,
    session: requests.Session,
    dandiset_id: str,
) -> dict[str, typing.Any]:
    start_time = time.perf_counter()
    path = f"{DANDI_API_BASE}/dandisets/{dandiset_id}/"
    logger.debug("fetching DANDI dandiset metadata: dandiset_id=%s", dandiset_id)
    metadata = session.get(path).json()
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    if not isinstance(metadata, dict):
        logger.debug(
            "DANDI dandiset metadata response was not an object: "
            "dandiset_id=%s response_type=%s elapsed_ms=%.3f",
            dandiset_id,
            type(metadata).__name__,
            elapsed_ms,
        )
        raise ValueError(
            f"DANDI metadata response was not an object for dandiset {dandiset_id}"
        )
    logger.debug(
        "fetched DANDI dandiset metadata: dandiset_id=%s elapsed_ms=%.3f",
        dandiset_id,
        elapsed_ms,
    )
    return metadata


def _version_value(version_metadata: object) -> str | None:
    if not isinstance(version_metadata, dict):
        return None
    version = version_metadata.get("version")
    if isinstance(version, str) and version:
        return version
    return None


def _get_dandiset_assets(
    dandiset_id: str,
    version: str | None = None,
    order: typing.Literal[
        "path", "created", "modified", "-path", "-created", "-modified"
    ] = "path",
    *,
    session: requests.Session | None = None,
) -> list[dict[str, typing.Any]]:
    """
    Get all assets (i.e. files) from a DANDI dandiset using the REST API.

    Parameters
    ----------
    dandiset_id : str
        The DANDI archive dandiset ID (e.g., '000363')
    version : str | None, optional
        Specific version to retrieve. If None, uses most recent published version,
        falling back to draft.
    order : Literal, default "path"
        Order results by field. Use '-' prefix for descending.

    Returns
    -------
    list[dict[str, typing.Any]]
        List of asset metadata dictionaries.
    """
    active_session = _get_session() if session is None else session
    requested_version = version
    resolved_version = _resolve_dandiset_version(
        dandiset_id,
        version,
        session=active_session,
    )

    assets: list[dict[str, typing.Any]] = []
    paginated_url = (
        f"{DANDI_API_BASE}/dandisets/{dandiset_id}/versions/{resolved_version}/assets/"
    )
    params: dict[str, str] | None = {"order": order}
    page_count = 0
    start_time = time.perf_counter()

    while True:
        page_count += 1
        logger.debug(
            "fetching DANDI asset page: dandiset_id=%s requested_version=%s "
            "resolved_version=%s order=%s page=%d url=%s",
            dandiset_id,
            requested_version,
            resolved_version,
            order,
            page_count,
            paginated_url,
        )
        response = active_session.get(paginated_url, params=params).json()
        if not isinstance(response, dict):
            logger.debug(
                "DANDI asset page response was not an object: dandiset_id=%s "
                "resolved_version=%s page=%d response_type=%s",
                dandiset_id,
                resolved_version,
                page_count,
                type(response).__name__,
            )
            raise ValueError(
                "DANDI asset page response was not an object for "
                f"dandiset {dandiset_id} version {resolved_version}"
            )
        results = response.get("results")
        if not isinstance(results, list):
            logger.debug(
                "DANDI asset page response did not include a results list: "
                "dandiset_id=%s resolved_version=%s page=%d",
                dandiset_id,
                resolved_version,
                page_count,
            )
            raise ValueError(
                "DANDI asset page response did not include results for "
                f"dandiset {dandiset_id} version {resolved_version}"
            )
        for asset in results:
            if not isinstance(asset, dict):
                logger.debug(
                    "DANDI asset metadata entry was not an object: dandiset_id=%s "
                    "resolved_version=%s page=%d asset_type=%s",
                    dandiset_id,
                    resolved_version,
                    page_count,
                    type(asset).__name__,
                )
                raise ValueError(
                    "DANDI asset metadata entry was not an object for "
                    f"dandiset {dandiset_id} version {resolved_version}"
                )
            assets.append(asset)
        next_url = response.get("next")
        if not isinstance(next_url, str) or not next_url:
            break
        paginated_url = next_url
        params = None

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.debug(
        "fetched DANDI asset metadata: dandiset_id=%s requested_version=%s "
        "resolved_version=%s asset_count=%d page_count=%d elapsed_ms=%.3f",
        dandiset_id,
        requested_version,
        resolved_version,
        len(assets),
        page_count,
        elapsed_ms,
    )
    return assets


def _get_asset_s3_url(
    dandiset_id: str,
    asset_id: str,
    version: str | None = None,
    *,
    session: requests.Session | None = None,
) -> str:
    """
    Get the S3 URL for a specific DANDI asset (e.g. a single NWB file).

    Parameters
    ----------
    dandiset_id : str
        The DANDI archive dandiset ID
    asset_id : str
        The specific asset ID
    version : str | None, optional
        Specific version to retrieve. If None, uses most recent published version,
        falling back to draft.

    Returns
    -------
    str
        The S3 URL for the asset
    """
    active_session = _get_session() if session is None else session
    requested_version = version
    resolved_version = _resolve_dandiset_version(
        dandiset_id,
        version,
        session=active_session,
    )
    path = (
        f"{DANDI_API_BASE}/dandisets/{dandiset_id}/versions/"
        f"{resolved_version}/assets/{asset_id}/"
    )
    start_time = time.perf_counter()
    logger.debug(
        "fetching DANDI asset metadata for S3 URL: dandiset_id=%s "
        "requested_version=%s resolved_version=%s asset_id=%s",
        dandiset_id,
        requested_version,
        resolved_version,
        asset_id,
    )
    response = active_session.get(path).json()
    if not isinstance(response, dict):
        logger.debug(
            "DANDI asset metadata response was not an object: dandiset_id=%s "
            "resolved_version=%s asset_id=%s response_type=%s",
            dandiset_id,
            resolved_version,
            asset_id,
            type(response).__name__,
        )
        raise ValueError(
            "DANDI asset metadata response was not an object for "
            f"dandiset {dandiset_id} asset {asset_id}"
        )
    s3_url = _get_asset_s3_url_from_metadata(response)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.debug(
        "resolved DANDI asset S3 URL: dandiset_id=%s requested_version=%s "
        "resolved_version=%s asset_id=%s elapsed_ms=%.3f",
        dandiset_id,
        requested_version,
        resolved_version,
        asset_id,
        elapsed_ms,
    )
    return s3_url


def _get_asset_s3_url_from_metadata(
    asset_metadata: typing.Mapping[str, typing.Any],
) -> str:
    content_urls = _get_asset_content_urls(asset_metadata)
    s3_url = next((url for url in content_urls if _is_s3_url(url)), None)
    if s3_url is None:
        logger.debug(
            "DANDI asset metadata did not include an S3 content URL: "
            "asset_id=%s content_url_count=%d",
            _asset_metadata_identifier(asset_metadata),
            len(content_urls),
        )
        raise ValueError(
            "No S3 content URL found for DANDI asset "
            f"{_asset_metadata_identifier(asset_metadata)}"
        )
    return s3_url


def _get_asset_content_urls(
    asset_metadata: typing.Mapping[str, typing.Any],
) -> tuple[str, ...]:
    content_url = asset_metadata.get("contentUrl")
    if isinstance(content_url, str):
        return (content_url,)
    if not isinstance(content_url, collections.abc.Sequence):
        return ()
    return tuple(url for url in content_url if isinstance(url, str) and url)


def _is_s3_url(url: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    return parsed.scheme == "s3" or "s3" in parsed.netloc.lower()


def _asset_metadata_identifier(asset_metadata: typing.Mapping[str, typing.Any]) -> str:
    asset_id = asset_metadata.get("asset_id")
    if isinstance(asset_id, str) and asset_id:
        return asset_id
    identifier = asset_metadata.get("identifier")
    if isinstance(identifier, str) and identifier:
        return identifier
    return "<unknown>"


def get_dandi_sources(
    dandiset_id: str,
    version: str | None = None,
    order: typing.Literal[
        "path", "created", "modified", "-path", "-created", "-modified"
    ] = "path",
    *,
    asset_ids: collections.abc.Sequence[str] | None = None,
    asset_filter: collections.abc.Callable[[dict[str, typing.Any]], bool] | None = None,
    max_assets: int | None = None,
) -> list[str]:
    """
    Get lazynwb-compatible sources for NWB assets in a DANDI dandiset.

    Parameters
    ----------
    dandiset_id : str
        The DANDI archive dandiset ID (e.g., '000363')
    version : str | None, optional
        Specific version to retrieve. If None, uses most recent published version,
        falling back to draft.
    order : Literal["path", "created", "modified", "-path", "-created", "-modified"], default "path"
        Order results by field. Use '-' prefix for descending (e.g., '-created').
        Ignored when asset_ids is provided without asset_filter.
    asset_ids : Sequence[str] | None, optional
        DANDI asset IDs to resolve. When provided without asset_filter, this avoids
        listing dandiset assets and preserves the given ID order.
    asset_filter : Callable[[dict[str, Any]], bool] | None, optional
        Function to filter NWB assets. Receives each asset's metadata dict and
        returns True/False to include/exclude, respectively.
    max_assets : int | None, optional
        Maximum number of NWB assets to include. Useful for testing large dandisets.

    Returns
    -------
    list[str]
        Sources for matching NWB assets, usable with get_df(), get_timeseries(),
        and scan_nwb().

    Examples
    --------
    >>> sources = get_dandi_sources('000363', version='0.231012.2129')
    >>> len(sources)
    174
    >>> all('s3.amazonaws.com' in source for source in sources[:3])
    True
    """
    normalized_asset_ids = _normalize_dandi_asset_ids(asset_ids)
    if asset_ids is not None and not normalized_asset_ids:
        logger.debug(
            "DANDI source asset ID selection was empty: dandiset_id=%s version=%s",
            dandiset_id,
            version,
        )
        raise ValueError(f"No DANDI asset IDs provided for dandiset {dandiset_id}")

    session = _get_session()
    requested_version = version
    resolved_version = _resolve_dandiset_version(
        dandiset_id,
        version,
        session=session,
    )

    if normalized_asset_ids is not None and asset_filter is None:
        assets = _asset_metadata_from_ids(normalized_asset_ids)
        nwb_assets = assets
        selected_assets = assets
        asset_count = len(assets)
    else:
        assets = _get_dandiset_assets(
            dandiset_id,
            resolved_version,
            order,
            session=session,
        )
        nwb_assets = _filter_dandi_nwb_assets(assets, order=order)
        selected_assets = nwb_assets
        if normalized_asset_ids is not None:
            asset_id_positions = {
                asset_id: index for index, asset_id in enumerate(normalized_asset_ids)
            }
            selected_assets = [
                asset
                for asset in selected_assets
                if _dandi_asset_id(asset) in asset_id_positions
            ]
            selected_assets.sort(
                key=lambda asset: asset_id_positions[_dandi_asset_id(asset)]
            )
        asset_count = len(assets)

    if asset_filter is not None:
        selected_assets = [asset for asset in selected_assets if asset_filter(asset)]

    if max_assets is not None:
        selected_assets = selected_assets[:max_assets]

    selected_asset_paths = tuple(_dandi_asset_path(asset) for asset in selected_assets)
    logger.debug(
        "selected DANDI NWB assets for source resolution: "
        "dandiset_id=%s requested_version=%s resolved_version=%s order=%s "
        "asset_count=%d nwb_asset_count=%d selected_asset_count=%d "
        "selected_asset_paths=%s asset_ids_count=%s asset_filter=%s "
        "max_assets=%s",
        dandiset_id,
        requested_version,
        resolved_version,
        order,
        asset_count,
        len(nwb_assets),
        len(selected_assets),
        selected_asset_paths,
        None if normalized_asset_ids is None else len(normalized_asset_ids),
        asset_filter is not None,
        max_assets,
    )

    if not selected_assets:
        logger.debug(
            "DANDI source asset selection was empty: dandiset_id=%s "
            "resolved_version=%s asset_count=%d nwb_asset_count=%d "
            "asset_ids_count=%s asset_filter=%s max_assets=%s "
            "selected_asset_paths=%s",
            dandiset_id,
            resolved_version,
            asset_count,
            len(nwb_assets),
            None if normalized_asset_ids is None else len(normalized_asset_ids),
            asset_filter is not None,
            max_assets,
            selected_asset_paths,
        )
        message = _empty_dandi_source_selection_message(
            dandiset_id=dandiset_id,
            resolved_version=resolved_version,
            asset_count=asset_count,
            nwb_asset_count=len(nwb_assets),
            asset_ids=normalized_asset_ids,
            asset_filter=asset_filter,
            max_assets=max_assets,
        )
        raise ValueError(message)

    source_uris = _get_asset_s3_urls(
        dandiset_id=dandiset_id,
        version=resolved_version,
        assets=selected_assets,
    )
    if not source_uris:
        logger.debug(
            "DANDI source URL resolution returned no URLs: dandiset_id=%s "
            "resolved_version=%s selected_asset_count=%d selected_asset_paths=%s",
            dandiset_id,
            resolved_version,
            len(selected_assets),
            selected_asset_paths,
        )
        raise ValueError(f"No valid S3 URLs found for assets in dandiset {dandiset_id}")

    source_count = len(source_uris)
    logger.debug(
        "resolved DANDI sources: dandiset_id=%s requested_version=%s "
        "resolved_version=%s source_count=%d",
        dandiset_id,
        requested_version,
        resolved_version,
        source_count,
    )
    return source_uris


def _get_asset_s3_urls(
    *,
    dandiset_id: str,
    version: str,
    assets: list[dict[str, typing.Any]],
) -> list[str]:
    urls: list[str | None] = [None] * len(assets)
    executor = lazynwb.utils.get_threadpool_executor()
    future_to_asset = {
        executor.submit(
            _get_asset_s3_url,
            dandiset_id,
            _dandi_asset_id(asset),
            version,
        ): (index, asset)
        for index, asset in enumerate(assets)
    }

    start_time = time.perf_counter()
    for future in concurrent.futures.as_completed(future_to_asset):
        index, asset = future_to_asset[future]
        asset_id = _dandi_asset_id(asset)
        asset_path = _dandi_asset_path(asset)
        logger.debug(
            "waiting for DANDI asset S3 URL: dandiset_id=%s version=%s "
            "asset_id=%s path=%s",
            dandiset_id,
            version,
            asset_id,
            asset_path,
        )
        urls[index] = future.result()

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.debug(
        "resolved DANDI asset S3 URL batch: dandiset_id=%s version=%s "
        "asset_count=%d elapsed_ms=%.3f",
        dandiset_id,
        version,
        len(assets),
        elapsed_ms,
    )
    return [typing.cast(str, url) for url in urls]


def _filter_dandi_nwb_assets(
    assets: list[dict[str, typing.Any]],
    *,
    order: typing.Literal[
        "path", "created", "modified", "-path", "-created", "-modified"
    ] = "path",
) -> list[dict[str, typing.Any]]:
    nwb_assets = [asset for asset in assets if _is_dandi_nwb_asset(asset)]
    if order in ("path", "-path"):
        return sorted(
            nwb_assets,
            key=_dandi_asset_sort_key,
            reverse=order == "-path",
        )
    return nwb_assets


def _normalize_dandi_asset_ids(
    asset_ids: collections.abc.Sequence[str] | None,
) -> tuple[str, ...] | None:
    if asset_ids is None:
        return None
    if isinstance(asset_ids, str):
        raise ValueError("asset_ids must be a sequence of DANDI asset ID strings")
    return tuple(asset_id for asset_id in asset_ids if asset_id)


def _asset_metadata_from_ids(
    asset_ids: collections.abc.Sequence[str],
) -> list[dict[str, typing.Any]]:
    return [{"asset_id": asset_id} for asset_id in asset_ids]


def _is_dandi_nwb_asset(asset: typing.Mapping[str, typing.Any]) -> bool:
    path = asset.get("path")
    return isinstance(path, str) and path.lower().endswith(".nwb")


def _dandi_asset_sort_key(asset: typing.Mapping[str, typing.Any]) -> tuple[str, str]:
    return (_dandi_asset_path(asset), _dandi_asset_id(asset))


def _dandi_asset_path(asset: typing.Mapping[str, typing.Any]) -> str:
    path = asset.get("path")
    if isinstance(path, str) and path:
        return path
    return _dandi_asset_id(asset)


def _dandi_asset_id(asset: typing.Mapping[str, typing.Any]) -> str:
    asset_id = asset.get("asset_id")
    if isinstance(asset_id, str) and asset_id:
        return asset_id
    identifier = asset.get("identifier")
    if isinstance(identifier, str) and identifier:
        return identifier
    raise ValueError("DANDI asset metadata did not include an asset identifier")


def _empty_dandi_source_selection_message(
    *,
    dandiset_id: str,
    resolved_version: str,
    asset_count: int,
    nwb_asset_count: int,
    asset_ids: tuple[str, ...] | None,
    asset_filter: collections.abc.Callable[[dict[str, typing.Any]], bool] | None,
    max_assets: int | None,
) -> str:
    criteria = []
    if asset_ids is not None:
        criteria.append(f"asset_ids={len(asset_ids)}")
    if asset_filter is not None:
        criteria.append("asset_filter")
    if max_assets is not None:
        criteria.append(f"max_assets={max_assets}")

    criteria_text = ""
    if criteria:
        criteria_text = f" after applying {' and '.join(criteria)}"

    return (
        f"No NWB assets selected for dandiset {dandiset_id} "
        f"version {resolved_version}{criteria_text}. "
        f"Found {asset_count} assets, {nwb_asset_count} NWB assets, selected 0."
    )


def from_dandi_asset(
    dandiset_id: str,
    asset_id: str,
    version: str | None = None,
) -> lazynwb.file_io.FileAccessor:
    """
    Open a FileAccessor for a specific DANDI asset.

    Parameters
    ----------
    dandiset_id : str
        The DANDI archive dandiset ID (e.g., '000363')
    asset_id : str
        The specific asset ID
    version : str | None, optional
        Specific version to retrieve. If None, uses most recent published version,
        falling back to draft.

    Returns
    -------
    FileAccessor
        A FileAccessor instance for the DANDI asset

    Examples
    --------
    >>> accessor = from_dandi_asset(
    ...     dandiset_id='000363',
    ...     asset_id='21c622b7-6d8e-459b-98e8-b968a97a1585',
    ...     version='0.231012.2129'
    ... )
    >>> isinstance(accessor, lazynwb.file_io.FileAccessor)
    True
    >>> 's3.amazonaws.com' in str(accessor._path)
    True
    """
    s3_url = _get_asset_s3_url(dandiset_id, asset_id, version)
    return lazynwb.file_io.FileAccessor(s3_url)


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
