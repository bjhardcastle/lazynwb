"""Opt-in DANDI:001637 integration framework smoke tests."""

from __future__ import annotations

import logging

import pytest

import tests._dandi_sample as dandi_sample

pytestmark = [pytest.mark.integration, pytest.mark.dandi_sample]


def test_dandi_001637_sample_source_urls_resolve(
    dandi_001637_sample_assets: tuple[dandi_sample._DandiSampleAsset, ...],
    dandi_001637_resolved_sample_assets: tuple[
        dandi_sample._DandiSampleResolvedAsset, ...
    ],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger=dandi_sample._LOGGER_NAME)
    dandi_sample._log_resolved_sample_assets(
        dandi_001637_resolved_sample_assets,
        cache_scope="session",
    )

    resolved_by_id = {
        resolved_asset.asset.asset_id: resolved_asset
        for resolved_asset in dandi_001637_resolved_sample_assets
    }

    assert tuple(resolved_by_id) == tuple(
        asset.asset_id for asset in dandi_001637_sample_assets
    )
    for sample_asset in dandi_001637_sample_assets:
        resolved_asset = resolved_by_id[sample_asset.asset_id]
        assert resolved_asset.asset.path == sample_asset.path
        assert resolved_asset.source_url.startswith("https://")
        assert "s3" in resolved_asset.source_url.lower()
        assert resolved_asset.source_url == dandi_sample._sanitize_source_url_for_logging(
            resolved_asset.source_url
        )
        assert sample_asset.asset_id in caplog.text
        assert sample_asset.path in caplog.text

    assert dandi_sample._DANDI_SAMPLE_DANDISET_ID in caplog.text
    assert dandi_sample._DANDI_SAMPLE_VERSION in caplog.text
    assert "source_url=" in caplog.text
    assert "cache_scope=session" in caplog.text
