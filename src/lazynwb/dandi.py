"""DANDI Archive integration for lazynwb."""

from __future__ import annotations

import concurrent.futures
import logging
import pathlib
from collections.abc import Callable
from typing import Any, Literal

import platformdirs
import polars as pl
import pydantic_settings
import requests

import lazynwb.file_io
import lazynwb.lazyframe
import lazynwb.utils

logger = logging.getLogger(__name__)

DANDI_API_BASE = "https://api.dandiarchive.org/api"
NEUROSIFT_LINDI_BASE = "https://lindi.neurosift.org"


class DandiConfig(pydantic_settings.BaseSettings):
    """Global configuration for DANDI-specific behavior."""

    model_config = pydantic_settings.SettingsConfigDict(
        env_prefix="LAZYNWB_DANDI_",
    )
    prefer_neurosift: bool = False
    use_local_cache: bool = False
    local_cache_dir: str = str(
        pathlib.Path(platformdirs.user_cache_dir("lazynwb", appauthor=False)) / "dandi"
    )
    request_timeout_seconds: float = 30.0


dandi_config = DandiConfig()


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


def _resolve_dandiset_version(dandiset_id: str, version: str | None) -> str:
    """Resolve a possibly-null version string to a concrete DANDI version."""
    if version is None:
        return _get_most_recent_dandiset_version(dandiset_id)
    return version


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
    version = _resolve_dandiset_version(dandiset_id, version)

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
    version = _resolve_dandiset_version(dandiset_id, version)

    path = f"{DANDI_API_BASE}/dandisets/{dandiset_id}/versions/{version}/assets/{asset_id}/"
    response = session.get(path).json()

    # Get S3 URL from contentUrl list
    s3_url = next((url for url in response["contentUrl"] if "s3" in url.lower()), None)

    if s3_url is None:
        raise ValueError(f"No S3 URL found for asset {asset_id}")

    try:
        s3_url = session.head(
            s3_url,
            allow_redirects=True,
            timeout=dandi_config.request_timeout_seconds,
        ).url
    except requests.RequestException as exc:
        logger.warning(
            f"Failed to resolve redirected S3 URL for asset {asset_id}; using "
            f"unresolved contentUrl entry instead: {exc!r}"
        )

    return s3_url


def _get_neurosift_lindi_url(dandiset_id: str, asset_id: str) -> str:
    """Construct the Neurosift LINDI URL for a DANDI asset."""
    return (
        f"{NEUROSIFT_LINDI_BASE}/dandi/dandisets/{dandiset_id}/assets/{asset_id}/"
        "nwb.lindi.json"
    )


def _is_neurosift_lindi_available(lindi_url: str) -> bool:
    """Return True if Neurosift has a pre-generated LINDI JSON for this asset."""
    session = _get_session()
    try:
        response = session.head(
            lindi_url,
            allow_redirects=True,
            timeout=dandi_config.request_timeout_seconds,
        )
    except requests.RequestException:
        return False
    return response.ok


def _get_local_cache_path(
    dandiset_id: str,
    asset_id: str,
    version: str,
    asset_path: str | None = None,
) -> pathlib.Path:
    """Return the on-disk cache path for an optionally downloaded DANDI asset."""
    root = pathlib.Path(dandi_config.local_cache_dir).expanduser()
    asset_name = (
        pathlib.PurePosixPath(asset_path).name if asset_path else f"{asset_id}.nwb"
    )
    return root / "assets" / dandiset_id / version / f"{asset_id}--{asset_name}"


def _download_url_to_local_cache(remote_url: str, local_path: pathlib.Path) -> None:
    """Download a remote file to the optional persistent on-disk cache."""
    session = _get_session()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    logger.info(f"Downloading DANDI asset to local cache: {local_path}")
    with session.get(
        remote_url,
        stream=True,
        timeout=dandi_config.request_timeout_seconds,
    ) as response:
        with tmp_path.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    stream.write(chunk)
    tmp_path.replace(local_path)


def _resolve_neurosift_source(
    dandiset_id: str,
    asset_id: str,
    resolved_version: str,
) -> str | None:
    """Resolve a Neurosift-backed source if a pre-generated LINDI JSON exists."""
    del resolved_version
    lindi_url = _get_neurosift_lindi_url(dandiset_id, asset_id)
    if not _is_neurosift_lindi_available(lindi_url):
        return None

    logger.info(f"Using remote Neurosift LINDI for DANDI asset {asset_id}")
    return lindi_url


def _resolve_raw_asset_source(
    dandiset_id: str,
    asset_id: str,
    resolved_version: str,
    asset_path: str | None,
    use_local_cache: bool,
) -> str:
    """Resolve the raw DANDI asset source, optionally using the persistent cache."""
    if use_local_cache:
        local_path = _get_local_cache_path(
            dandiset_id,
            asset_id,
            resolved_version,
            asset_path,
        )
        if local_path.exists():
            logger.info(f"Using cached DANDI asset from {local_path}")
            return local_path.as_posix()
        remote_url = _get_asset_s3_url(dandiset_id, asset_id, resolved_version)
        try:
            _download_url_to_local_cache(remote_url, local_path)
        except requests.RequestException as exc:
            logger.warning(
                "Failed to cache DANDI asset "
                f"{asset_id} locally; falling back to remote URL: {exc!r}"
            )
        else:
            return local_path.as_posix()

    return _get_asset_s3_url(dandiset_id, asset_id, resolved_version)


def resolve_dandi_asset_source(
    dandiset_id: str,
    asset_id: str,
    version: str | None = None,
    *,
    asset_path: str | None = None,
    use_local_cache: bool | None = None,
) -> str:
    """
    Resolve the preferred lazynwb source for a DANDI asset.

    This prefers a pre-generated Neurosift LINDI URL when available. If that is not
    available and local caching is enabled, the raw asset is downloaded to a stable
    local path and opened from disk. Otherwise, lazynwb falls back to the direct
    remote HDF5 URL.
    """
    resolved_version = _resolve_dandiset_version(dandiset_id, version)
    use_local_cache = (
        dandi_config.use_local_cache if use_local_cache is None else use_local_cache
    )

    if dandi_config.prefer_neurosift:
        if source := _resolve_neurosift_source(
            dandiset_id,
            asset_id,
            resolved_version,
        ):
            return source

    return _resolve_raw_asset_source(
        dandiset_id,
        asset_id,
        resolved_version,
        asset_path,
        use_local_cache,
    )


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
    *,
    use_local_cache: bool | None = None,
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
    use_local_cache : bool | None, optional
        If True, non-Neurosift fallback assets are cached locally before opening.
        If None, uses ``LAZYNWB_DANDI_USE_LOCAL_CACHE`` /
        ``dandi_config.use_local_cache``.

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
    >>> str(accessor._path).startswith('http') or pathlib.Path(accessor._path).exists()
    True
    """
    source = resolve_dandi_asset_source(
        dandiset_id,
        asset_id,
        version,
        use_local_cache=use_local_cache,
    )
    return lazynwb.file_io.FileAccessor(source)


def scan_dandiset(
    dandiset_id: str,
    table_path: str,
    version: str | None = None,
    asset_filter: Callable[[dict[str, Any]], bool] | None = None,
    max_assets: int | None = None,
    use_local_cache: bool | None = None,
    **scan_kwargs: object,
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
        Function to filter assets. Receives each asset's metadata dict and returns
        True/False to include/exclude, respectively.
    max_assets : int | None, optional
        Maximum number of assets to scan. Useful for testing on large dandisets.
    use_local_cache : bool | None, optional
        If True, non-Neurosift fallback assets are cached locally before scanning.
        If None, uses ``LAZYNWB_DANDI_USE_LOCAL_CACHE`` /
        ``dandi_config.use_local_cache``.
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
    version = _resolve_dandiset_version(dandiset_id, version)
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

    source_urls = []
    executor = lazynwb.utils.get_threadpool_executor()
    future_to_asset = {}
    for asset in assets:
        future = executor.submit(
            resolve_dandi_asset_source,
            dandiset_id,
            asset["asset_id"],
            version,
            asset_path=asset.get("path"),
            use_local_cache=use_local_cache,
        )
        future_to_asset[future] = asset

    for future in concurrent.futures.as_completed(future_to_asset):
        future_to_asset[future]
        source_urls.append(future.result())

    if not source_urls:
        raise ValueError(
            f"No valid asset URLs found for assets in dandiset {dandiset_id}"
        )

    logger.info(
        f"Scanning {len(source_urls)} assets from dandiset {dandiset_id} as a LazyFrame"
    )
    return lazynwb.lazyframe.scan_nwb(
        source=source_urls,
        table_path=table_path,
        **scan_kwargs,
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
