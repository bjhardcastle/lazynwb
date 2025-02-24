from __future__ import annotations

import os
from collections.abc import Generator

import dandi.dandiapi

from lazynwb.file_io import FileAccessor


def get_dandi_client(token: str | None = None) -> dandi.dandiapi.DandiAPIClient:
    if token is None:
        token = os.getenv("DANDI_API_TOKEN", default=None)
    return dandi.dandiapi.DandiAPIClient(token=token)


def get_dandiset_nwbs(
    dandiset_id: str, version_id: str | None = None
) -> Generator[FileAccessor, None, None]:
    """Get a LazyFile object for each file in the specified Dandiset.

    >>> next(get_dandiset_nwbs('000363'))           # ephys dataset from the Svoboda Lab
    LazyFile('https://dandiarchive.s3.amazonaws.com/blobs/56c/31a/56c31a1f-a6fb-4b73-ab7d-98fb5ef9a553')
    """
    # assets = get_dandiset_assets(dandiset_id, version_id)
    # def _helper(asset) -> LazyFile:
    #     return LazyFile(asset.get_content_url(follow_redirects=1, strip_query=True))
    # future_to_nwb: dict[concurrent.futures.Future, LazyFile] = {}
    # with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    #     for asset in assets:
    #         pool.submit(_helper, asset)
    # for future in concurrent.futures.as_completed(future_to_nwb):
    #     yield future.result()
    assets = get_dandiset_assets(dandiset_id, version_id)
    for asset in assets:
        yield get_lazynwb_from_dandiset_asset(asset)


def get_lazynwb_from_dandiset_asset(
    asset: dandi.dandiapi.BaseRemoteAsset,
) -> FileAccessor:
    return FileAccessor(asset.get_content_url(follow_redirects=1, strip_query=False))


def get_dandiset_assets(
    dandiset_id: str, version_id: str | None = None, lazy: bool = True
) -> tuple[dandi.dandiapi.BaseRemoteAsset, ...]:
    """Get a sequence of assets from the specified Dandiset.

    >>> assets = get_dandiset_assets('000363')      # ephys dataset from the Svoboda Lab
    >>> assets[0].path
    'sub-440956/sub-440956_ses-20190208T133600_behavior+ecephys+ogen.nwb'
    >>> assets[0].version_id
    '0.231012.2129'
    """
    with get_dandi_client() as client:
        dandiset = client.get_dandiset(
            dandiset_id=dandiset_id, version_id=version_id, lazy=lazy
        )
        return tuple(dandiset.get_assets())


if __name__ == "__main__":
    from doctest import testmod

    testmod()
