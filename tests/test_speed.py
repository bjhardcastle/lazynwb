import logging
import time

import pytest

import lazynwb

logger = logging.getLogger()

MIN_OPEN_TIME_SECONDS = 2.5

def get_large_hdf5_url() -> str:
    dandiset_id = '000363'  # ephys dataset from the Svoboda Lab
    filepath = 'sub-440957/sub-440957_ses-20190211T143614_behavior+ecephys+image+ogen.nwb' # 437 GB file
    with lazynwb.get_dandi_client() as client:
        asset = client.get_dandiset(dandiset_id=dandiset_id, version_id='draft').get_asset_by_path(filepath)
        return asset.get_content_url(follow_redirects=1, strip_query=True)

def get_small_zarr_url() -> str:
    return 's3://codeocean-s3datasetsbucket-1u41qdg42ur9/39490bff-87c9-4ef2-b408-36334e748ac6/nwb/ecephys_620264_2022-08-02_15-39-59_experiment1_recording1.nwb'

@pytest.fixture
def url(request: pytest.FixtureRequest) -> str:
    if request.param == 'large_hdf5':
        return get_large_hdf5_url()
    elif request.param == 'small_zarr':
        return get_small_zarr_url()
    else:
        raise ValueError(f'Unknown url fixture value: {request.param}')

@pytest.mark.parametrize('url', ['large_hdf5', 'small_zarr'], indirect=True)
def test_open_large_hdf5_time(url: str) -> None:
    t0 = time.time()
    nwb = lazynwb.LazyNWB(url)
    t = time.time() - t0
    assert t < MIN_OPEN_TIME_SECONDS, f'Opening {url} with {nwb.__class__.__name__} took too long: {t:.1f} seconds (expected < {MIN_OPEN_TIME_SECONDS})'
    logger.info(f'Opened {url} with {nwb.__class__.__name__} in {t} seconds')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__])
