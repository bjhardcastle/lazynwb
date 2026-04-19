from __future__ import annotations

import pathlib

import pytest

import lazynwb.dandi


def test_resolve_dandi_asset_source_prefers_neurosift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefer the Neurosift LINDI URL when it is available."""
    monkeypatch.setattr(lazynwb.dandi.dandi_config, "use_local_cache", False)
    monkeypatch.setattr(
        lazynwb.dandi,
        "_is_neurosift_lindi_available",
        lambda url: True,
    )
    monkeypatch.setattr(
        lazynwb.dandi,
        "_get_asset_s3_url",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError(
                "remote fallback should not be used when Neurosift is available"
            )
        ),
    )

    source = lazynwb.dandi.resolve_dandi_asset_source(
        "000363",
        "21c622b7-6d8e-459b-98e8-b968a97a1585",
        "0.231012.2129",
    )

    assert source == (
        "https://lindi.neurosift.org/dandi/dandisets/000363/assets/"
        "21c622b7-6d8e-459b-98e8-b968a97a1585/nwb.lindi.json"
    )


def test_resolve_dandi_asset_source_does_not_cache_neurosift_lindi(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Neurosift LINDI JSONs should stay remote even when cache is enabled."""
    monkeypatch.setattr(lazynwb.dandi.dandi_config, "local_cache_dir", str(tmp_path))
    monkeypatch.setattr(lazynwb.dandi.dandi_config, "use_local_cache", True)
    monkeypatch.setattr(
        lazynwb.dandi,
        "_is_neurosift_lindi_available",
        lambda url: True,
    )
    monkeypatch.setattr(
        lazynwb.dandi,
        "_get_asset_s3_url",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError(
                "raw asset fallback should not be used when Neurosift is available"
            )
        ),
    )
    monkeypatch.setattr(
        lazynwb.dandi,
        "_download_url_to_local_cache",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError(
                "Neurosift LINDI JSON should not be downloaded into local cache"
            )
        ),
    )

    source = lazynwb.dandi.resolve_dandi_asset_source(
        "000363",
        "asset-789",
        "0.231012.2129",
    )

    assert source == (
        "https://lindi.neurosift.org/dandi/dandisets/000363/assets/"
        "asset-789/nwb.lindi.json"
    )


def test_resolve_dandi_asset_source_uses_existing_local_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Use the optional local cache after the Neurosift lookup misses."""
    monkeypatch.setattr(
        lazynwb.dandi,
        "_is_neurosift_lindi_available",
        lambda url: False,
    )
    monkeypatch.setattr(
        lazynwb.dandi,
        "_get_asset_s3_url",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cached local asset should be used before remote fallback")
        ),
    )
    monkeypatch.setattr(lazynwb.dandi.dandi_config, "local_cache_dir", str(tmp_path))

    cached_path = lazynwb.dandi._get_local_cache_path(
        "000363",
        "asset-123",
        "0.231012.2129",
        "sub-001/example.nwb",
    )
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.write_bytes(b"cached")

    source = lazynwb.dandi.resolve_dandi_asset_source(
        "000363",
        "asset-123",
        "0.231012.2129",
        asset_path="sub-001/example.nwb",
        use_local_cache=True,
    )

    assert source == cached_path.as_posix()


def test_resolve_dandi_asset_source_populates_local_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Populate the optional local cache when enabled and Neurosift is unavailable."""
    monkeypatch.setattr(
        lazynwb.dandi,
        "_is_neurosift_lindi_available",
        lambda url: False,
    )
    monkeypatch.setattr(
        lazynwb.dandi,
        "_get_asset_s3_url",
        lambda *args, **kwargs: "https://example.org/test.nwb",
    )
    monkeypatch.setattr(lazynwb.dandi.dandi_config, "local_cache_dir", str(tmp_path))

    def _fake_download(remote_url: str, local_path: pathlib.Path) -> None:
        assert remote_url == "https://example.org/test.nwb"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(b"downloaded")

    monkeypatch.setattr(
        lazynwb.dandi,
        "_download_url_to_local_cache",
        _fake_download,
    )

    source = lazynwb.dandi.resolve_dandi_asset_source(
        "000363",
        "asset-456",
        "0.231012.2129",
        asset_path="sub-001/example.nwb",
        use_local_cache=True,
    )

    resolved_path = pathlib.Path(source)
    assert resolved_path.exists()
    assert resolved_path.read_bytes() == b"downloaded"
