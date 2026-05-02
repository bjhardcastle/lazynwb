import logging
import time

import pytest

import lazynwb
from lazynwb.dandi import _get_asset_s3_url

logger = logging.getLogger()

MIN_OPEN_TIME_SECONDS = 2.5
TEST_DANDISET_ID = "000363"
TEST_VERSION = "0.231012.2129"
TEST_ASSET_ID = "21c622b7-6d8e-459b-98e8-b968a97a1585"

def get_large_hdf5_url() -> str:
    return _get_asset_s3_url(
        dandiset_id=TEST_DANDISET_ID,
        asset_id=TEST_ASSET_ID,
        version=TEST_VERSION,
    )

def get_small_zarr_url() -> str:
    return 's3://codeocean-s3datasetsbucket-1u41qdg42ur9/00865745-db58-495d-9c5e-e28424bb4b97/nwb/ecephys_721536_2024-05-16_12-32-31_experiment1_recording1.nwb'

@pytest.fixture
def url(request: pytest.FixtureRequest) -> str:
    if request.param == 'large_hdf5':
        return get_large_hdf5_url()
    elif request.param == 'small_zarr':
        return get_small_zarr_url()
    else:
        raise ValueError(f'Unknown url fixture value: {request.param}')

@pytest.mark.xfail(reason="Removed dandi helper function")
@pytest.mark.parametrize('url', ['large_hdf5', 'small_zarr'], indirect=True)
def test_open_time(url: str) -> None:
    # may need to try this more than once: S3 storage can be slow on first request in a while
    t0 = time.time()
    nwb = lazynwb.FileAccessor(url)
    t = time.time() - t0
    logger.info(f'Opened {url} with {nwb.__class__.__name__} in {t:.2f} seconds')
    assert t < MIN_OPEN_TIME_SECONDS, f'Opening {url} with {nwb.__class__.__name__} took too long: {t:.1f} seconds (expected < {MIN_OPEN_TIME_SECONDS})'

@pytest.mark.xfail(reason="Removed dandi helper function")
@pytest.mark.parametrize('url', ['large_hdf5', 'small_zarr'], indirect=True)
def test_metadata_df(url: str) -> None:
    t0 = time.time()
    df = lazynwb.get_metadata_df(url, disable_progress=True)
    t    = time.time() - t0
    assert t < MIN_OPEN_TIME_SECONDS, f'Fetching summary dataframe took too long: {t:.1f} seconds (expected < {MIN_OPEN_TIME_SECONDS})'
    logger.info(f'Fetched summary dataframe for {url} in {t:.2f} seconds')
    
@pytest.mark.skip(reason="Remote backend performance comparisons are too variable for automated tests")
@pytest.mark.parametrize('url', ['large_hdf5'], indirect=True)
def test_remfile_vs_h5py(url: str) -> None:
    original_use_remfile = lazynwb.config.use_remfile
    times = {}

    def time_open(*, use_remfile: bool) -> float:
        lazynwb.clear_cache()
        lazynwb.config.use_remfile = use_remfile
        t0 = time.time()
        _ = lazynwb.FileAccessor(url)
        return time.time() - t0

    try:
        # Warm both code paths first: remote open times vary enough that a single
        # cold request does not give a stable backend comparison.
        for use_remfile in [True, False]:
            _ = time_open(use_remfile=use_remfile)
        for use_remfile in [True, False]:
            times[use_remfile] = time_open(use_remfile=use_remfile)
            logger.info(
                f'Opened {url} with {use_remfile=} in {times[use_remfile]:.2f} seconds'
            )
    finally:
        lazynwb.config.use_remfile = original_use_remfile
        lazynwb.clear_cache()
    assert all(t < MIN_OPEN_TIME_SECONDS for t in times.values()), (
        f'Opening {url} took too long after warmup: {times=}'
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) 
    pytest.main([__file__, "-v"])
