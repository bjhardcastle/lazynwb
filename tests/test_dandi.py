"""Integration tests for DANDI Archive functionality.

These tests make real network calls to the DANDI API and are marked as integration tests.
They use a stable, small dandiset (000363) for testing.
"""

from __future__ import annotations

import pytest

import lazynwb.file_io
from lazynwb.dandi import (
    from_dandi_asset,
    get_dandiset_s3_urls,
    scan_dandiset,
)

# Test data constants - using a stable, published dandiset
TEST_DANDISET_ID = "000363"
TEST_VERSION = "0.231012.2129"
TEST_ASSET_ID = "21c622b7-6d8e-459b-98e8-b968a97a1585"

pytestmark = pytest.mark.integration


def test_get_dandiset_s3_urls_returns_valid_urls():
    """Test that get_dandiset_s3_urls returns a list of valid S3 URLs."""
    urls = get_dandiset_s3_urls(TEST_DANDISET_ID, TEST_VERSION)
    assert isinstance(urls, list)
    assert len(urls) > 0
    for url in urls:
        assert isinstance(url, str)
        assert "s3" in url.lower()


def test_get_dandiset_s3_urls_with_latest_version():
    """Test fetching URLs with latest version."""
    urls = get_dandiset_s3_urls(TEST_DANDISET_ID, version=None)
    assert isinstance(urls, list)
    assert len(urls) > 0


def test_from_dandi_asset_returns_file_accessor():
    """Test that from_dandi_asset returns a readable FileAccessor."""
    accessor = from_dandi_asset(TEST_DANDISET_ID, TEST_ASSET_ID, TEST_VERSION)
    assert isinstance(accessor, lazynwb.file_io.FileAccessor)
    assert accessor.file is not None


def test_from_dandi_asset_with_latest_version():
    """Test creating FileAccessor with latest version."""
    accessor = from_dandi_asset(TEST_DANDISET_ID, TEST_ASSET_ID, version=None)
    assert isinstance(accessor, lazynwb.file_io.FileAccessor)


def test_scan_dandiset():
    """Test that scan_dandiset returns a LazyFrame with data."""
    lf = scan_dandiset(
        TEST_DANDISET_ID,
        "/units",
        version=TEST_VERSION,
        max_assets=1,
        infer_schema_length=1,
    )
    assert "spike_times" in lf.collect_schema()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])