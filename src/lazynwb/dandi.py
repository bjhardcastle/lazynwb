"""DANDI Archive integration for lazynwb."""

from __future__ import annotations

import concurrent.futures
import logging
from collections.abc import Callable
from typing import Any, Literal

import polars as pl
import requests

import lazynwb.file_io
import lazynwb.lazyframe
import lazynwb.utils

logger = logging.getLogger(__name__)

DANDI_API_BASE = "https://api.dandiarchive.org/api"


def _get_session() -> requests.Session:
    """Create a requests session with automatic error handling."""
    session = requests.Session()
    session.hooks = {"response": lambda r, *args, **kwargs: r.raise_for_status()}
    return session


def _get_most_recent_dandiset_version(dandiset_id: str) -> str:
    """Get the latest version string for a dandiset."""
    session = _get_session()
    path = f"{DANDI_API_BASE}/dandisets/{dandiset_id}/"
    response = session.get(path).json()
    return response["most_recent_published_version"]["version"]


def _get_dandiset_assets(
    dandiset_id: str,
    version: str | None = None,
    order: Literal[
        "path", "created", "modified", "-path", "-created", "-modified"
    ] = "path",
) -> list[dict[str, Any]]:
    """
    Get all assets (i.e. files) from a DANDI dandiset using the REST API.

    Parameters
    ----------
    dandiset_id : str
        The DANDI archive dandiset ID (e.g., '000363')
    version : str | None, optional
        Specific version to retrieve. If None, uses most recent published version.
    order : Literal, default "path"
        Order results by field. Use '-' prefix for descending.

    Returns
    -------
    list[dict[str, Any]]
        List of asset metadata dictionaries.
    """
    session = _get_session()
    if version is None:
        version = _get_most_recent_dandiset_version(dandiset_id)

    assets = []
    paginated_url = (
        f"{DANDI_API_BASE}/dandisets/{dandiset_id}/versions/{version}/assets/"
    )

    while True:
        response = session.get(paginated_url, params={"order": order}).json()
        assets.extend(response["results"])
        if not response["next"]:
            break
        paginated_url = response["next"]

    logger.info(f"Fetched {len(assets)} assets from dandiset {dandiset_id}")
    return assets


def _get_asset_s3_url(
    dandiset_id: str,
    asset_id: str,
    version: str | None = None,
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
        Specific version to retrieve. If None, uses most recent published version.

    Returns
    -------
    str
        The S3 URL for the asset
    """
    session = _get_session()
    if version is None:
        version = _get_most_recent_dandiset_version(dandiset_id)

    path = f"{DANDI_API_BASE}/dandisets/{dandiset_id}/versions/{version}/assets/{asset_id}/"
    response = session.get(path).json()

    # Get S3 URL from contentUrl list
    s3_url = next((url for url in response["contentUrl"] if "s3" in url.lower()), None)

    if s3_url is None:
        raise ValueError(f"No S3 URL found for asset {asset_id}")

    s3_url = session.head(s3_url, allow_redirects=True).url

    return s3_url


def get_dandiset_s3_urls(
    dandiset_id: str,
    version: str | None = None,
    order: Literal[
        "path", "created", "modified", "-path", "-created", "-modified"
    ] = "path",
) -> list[str]:
    """
    Get S3 URLs for all NWB assets in a DANDI dandiset.

    Parameters
    ----------
    dandiset_id : str
        The DANDI archive dandiset ID (e.g., '000363')
    version : str | None, optional
        Specific version to retrieve. If None, uses most recent published version.
    order : Literal["path", "created", "modified", "-path", "-created", "-modified"], default "path"
        Order results by field. Use '-' prefix for descending (e.g., '-created').

    Returns
    -------
    list[str]
        List of S3 URLs for all NWB assets

    Examples
    --------
    >>> urls = get_dandiset_s3_urls('000363', version='0.231012.2129')
    >>> len(urls)
    174
    >>> all('s3.amazonaws.com' in url for url in urls[:3])
    True
    """
    assets = _get_dandiset_assets(dandiset_id, version, order)
    urls = []

    executor = lazynwb.utils.get_threadpool_executor()
    future_to_asset = {}
    for asset in assets:
        future = executor.submit(
            _get_asset_s3_url,
            dandiset_id,
            asset["asset_id"],
            version,
        )
        future_to_asset[future] = asset

    futures = concurrent.futures.as_completed(future_to_asset)
    for future in futures:
        asset = future_to_asset[future]
        urls.append(future.result())
    return urls


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
        Specific version to retrieve. If None, uses most recent published version.

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


def scan_dandiset(
    dandiset_id: str,
    table_path: str,
    version: str | None = None,
    asset_filter: Callable[[dict[str, Any]], bool] | None = None,
    max_assets: int | None = None,
    **scan_kwargs,
) -> pl.LazyFrame:
    """
    Scan a common table across all NWB assets in a DANDI dandiset.

    Parameters
    ----------
    dandiset_id : str
        The DANDI archive dandiset ID (e.g., '000363')
    table_path : str
        Path to the table within each NWB file (e.g., '/units', '/intervals/trials')
    version : str | None, optional
        Specific version to retrieve. If None, uses most recent published version.
    asset_filter : Callable[[dict[str, Any]], bool] | None, optional
        Function to filter assets. Receives each asset's metadata dict and returns True/False to include/exclude, respectively.
    max_assets : int | None, optional
        Maximum number of assets to scan. Useful for testing on large dandisets.
    **scan_kwargs
        Additional keyword arguments to pass to scan_nwb()

    Returns
    -------
    polars.LazyFrame
        LazyFrame containing concatenated tables from all matching assets

    Examples
    --------
    >>> lf = scan_dandiset(
    ...     dandiset_id='000363',
    ...     table_path='/units',
    ...     version='0.231012.2129',
    ...     max_assets=1,           # limit for testing
    ...     infer_schema_length=1, # limit for testing
    ... )
    >>> 'spike_times' in lf.collect_schema()
    True
    """
    assets = _get_dandiset_assets(dandiset_id, version)

    if asset_filter is not None:
        assets = [asset for asset in assets if asset_filter(asset)]

    if max_assets is not None:
        assets = assets[:max_assets]

    if not assets:
        msg = f"No assets found in dandiset {dandiset_id}"
        if asset_filter is not None:
            msg += " after applying asset filter"
        if max_assets is not None:
            msg += f" with {max_assets=}"
        raise ValueError(msg)

    s3_urls = []
    executor = lazynwb.utils.get_threadpool_executor()
    future_to_asset = {}
    for asset in assets:
        future = executor.submit(
            _get_asset_s3_url,
            dandiset_id,
            asset["asset_id"],
            version,
        )
        future_to_asset[future] = asset

    for future in concurrent.futures.as_completed(future_to_asset):
        asset = future_to_asset[future]
        s3_urls.append(future.result())

    if not s3_urls:
        raise ValueError(f"No valid S3 URLs found for assets in dandiset {dandiset_id}")

    logger.info(
        f"Scanning {len(s3_urls)} assets from dandiset {dandiset_id}as a LazyFrame"
    )
    return lazynwb.lazyframe.scan_nwb(
        source=s3_urls,
        table_path=table_path,
        **scan_kwargs,
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
