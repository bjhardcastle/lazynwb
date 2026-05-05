from __future__ import annotations

import collections.abc
import concurrent.futures
import logging

import pytest

import lazynwb.dandi as dandi
import lazynwb.utils as utils
import tests._dandi_sample as dandi_sample


class _FakeResponse:
    def __init__(self: _FakeResponse, payload: object) -> None:
        self._payload = payload

    def json(self: _FakeResponse) -> object:
        return self._payload


class _FakeSession:
    def __init__(
        self: _FakeSession, responses: collections.abc.Iterable[object]
    ) -> None:
        self._responses = list(responses)
        self.get_calls: list[tuple[str, dict[str, str] | None]] = []

    def get(
        self: _FakeSession,
        url: str,
        params: dict[str, str] | None = None,
    ) -> _FakeResponse:
        self.get_calls.append((url, None if params is None else dict(params)))
        if not self._responses:
            raise AssertionError(f"Unexpected DANDI API GET: {url}")
        return _FakeResponse(self._responses.pop(0))


class _ImmediateExecutor:
    def submit(
        self: _ImmediateExecutor,
        fn: collections.abc.Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> concurrent.futures.Future[object]:
        future: concurrent.futures.Future[object] = concurrent.futures.Future()
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            future.set_exception(exc)
        else:
            future.set_result(result)
        return future


def test_version_none_uses_most_recent_published_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = _FakeSession(
        [
            {
                "draft_version": {"version": "draft"},
                "most_recent_published_version": {"version": "0.1.0"},
            }
        ]
    )
    caplog.set_level(logging.DEBUG, logger="lazynwb.dandi")

    resolved_version = dandi._resolve_dandiset_version(
        "000001",
        version=None,
        session=session,
    )

    assert resolved_version == "0.1.0"
    assert session.get_calls == [(f"{dandi.DANDI_API_BASE}/dandisets/000001/", None)]
    assert "using most recent published DANDI dandiset version" in caplog.text


def test_version_none_falls_back_to_draft_metadata_with_debug_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = _FakeSession(
        [
            {
                "draft_version": {"version": "draft"},
                "most_recent_published_version": None,
            }
        ]
    )
    caplog.set_level(logging.DEBUG, logger="lazynwb.dandi")

    resolved_version = dandi._resolve_dandiset_version(
        "001637",
        version=None,
        session=session,
    )

    assert resolved_version == "draft"
    assert "falling back to draft" in caplog.text
    assert "001637" in caplog.text


def test_get_dandiset_s3_urls_with_explicit_draft_resolves_draft_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3_url = "https://dandiarchive.s3.amazonaws.com/blobs/aaa"
    session = _FakeSession(
        [
            {
                "next": None,
                "results": [
                    {
                        "asset_id": "asset-a",
                        "path": "sub-820454/session.nwb",
                    }
                ],
            },
            {
                "contentUrl": [
                    "https://api.dandiarchive.org/api/assets/asset-a/download/",
                    s3_url,
                ],
                "identifier": "asset-a",
            },
        ]
    )
    monkeypatch.setattr(dandi, "_get_session", lambda: session)
    monkeypatch.setattr(utils, "get_threadpool_executor", lambda: _ImmediateExecutor())

    urls = dandi.get_dandiset_s3_urls("001637", version="draft")

    assert urls == [s3_url]
    assert session.get_calls == [
        (
            f"{dandi.DANDI_API_BASE}/dandisets/001637/versions/draft/assets/",
            {"order": "path"},
        ),
        (
            f"{dandi.DANDI_API_BASE}/dandisets/001637/versions/draft/assets/asset-a/",
            None,
        ),
    ]


def test_get_dandiset_s3_urls_filters_nwb_sorts_by_path_and_paginates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3_url_a = "https://dandiarchive.s3.amazonaws.com/blobs/aaa"
    s3_url_b = "https://dandiarchive.s3.amazonaws.com/blobs/bbb"
    next_url = (
        f"{dandi.DANDI_API_BASE}/dandisets/000001/versions/0.1.0/assets/"
        "?order=path&page=2"
    )
    session = _FakeSession(
        [
            {
                "draft_version": {"version": "draft"},
                "most_recent_published_version": {"version": "0.1.0"},
            },
            {
                "next": next_url,
                "results": [
                    {"asset_id": "asset-b", "path": "sub-b/session.nwb"},
                    {"asset_id": "asset-zarr", "path": "sub-z/session.zarr"},
                ],
            },
            {
                "next": None,
                "results": [
                    {"asset_id": "asset-txt", "path": "sub-a/notes.txt"},
                    {"asset_id": "asset-a", "path": "sub-a/session.nwb"},
                ],
            },
            {
                "contentUrl": [
                    "https://api.dandiarchive.org/api/assets/asset-a/download/",
                    s3_url_a,
                ],
                "identifier": "asset-a",
            },
            {
                "contentUrl": [
                    "https://api.dandiarchive.org/api/assets/asset-b/download/",
                    s3_url_b,
                ],
                "identifier": "asset-b",
            },
        ]
    )
    monkeypatch.setattr(dandi, "_get_session", lambda: session)
    monkeypatch.setattr(utils, "get_threadpool_executor", lambda: _ImmediateExecutor())

    urls = dandi.get_dandiset_s3_urls("000001", version=None)

    assert urls == [s3_url_a, s3_url_b]
    assert session.get_calls == [
        (f"{dandi.DANDI_API_BASE}/dandisets/000001/", None),
        (
            f"{dandi.DANDI_API_BASE}/dandisets/000001/versions/0.1.0/assets/",
            {"order": "path"},
        ),
        (next_url, None),
        (
            f"{dandi.DANDI_API_BASE}/dandisets/000001/versions/0.1.0/assets/asset-a/",
            None,
        ),
        (
            f"{dandi.DANDI_API_BASE}/dandisets/000001/versions/0.1.0/assets/asset-b/",
            None,
        ),
    ]


def test_asset_s3_url_extraction_prefers_s3_content_url() -> None:
    s3_url = "https://dandiarchive.s3.amazonaws.com/blobs/aaa"

    assert (
        dandi._get_asset_s3_url_from_metadata(
            {
                "contentUrl": [
                    "https://api.dandiarchive.org/api/assets/asset-a/download/",
                    s3_url,
                ],
                "identifier": "asset-a",
            }
        )
        == s3_url
    )


@pytest.mark.parametrize(
    "asset_metadata",
    [
        {},
        {"contentUrl": []},
        {"contentUrl": ["https://api.dandiarchive.org/api/assets/asset-a/download/"]},
    ],
)
def test_asset_s3_url_extraction_rejects_missing_or_non_s3_content_url(
    asset_metadata: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="No S3 content URL found"):
        dandi._get_asset_s3_url_from_metadata(asset_metadata)


def test_scan_dandiset_uses_url_helper_then_scan_nwb_without_table_read_logic(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    assets = [
        {"asset_id": "asset-b", "path": "sub-keep-b/session.nwb"},
        {"asset_id": "asset-note", "path": "sub-keep-a/notes.txt"},
        {"asset_id": "asset-a", "path": "sub-keep-a/session.nwb"},
        {"asset_id": "asset-skip", "path": "sub-skip/session.nwb"},
        {"asset_id": "asset-c", "path": "sub-keep-c/session.nwb"},
    ]
    selected_assets: list[dict[str, object]] = []
    scan_calls: list[dict[str, object]] = []
    sentinel = object()
    session = object()

    def _get_dandiset_assets(
        dandiset_id: str,
        version: str | None = None,
        order: str = "path",
        *,
        session: object | None = None,
    ) -> list[dict[str, object]]:
        assert dandiset_id == "000001"
        assert version == "0.1.0"
        assert order == "path"
        assert session is not None
        return assets

    def _get_asset_s3_urls(
        *,
        dandiset_id: str,
        version: str,
        assets: list[dict[str, object]],
    ) -> list[str]:
        assert dandiset_id == "000001"
        assert version == "0.1.0"
        selected_assets.extend(assets)
        return [f"s3://bucket/{asset['asset_id']}.nwb" for asset in assets]

    def _get_asset_s3_url(*args: object, **kwargs: object) -> str:
        raise AssertionError("scan_dandiset must use the DANDI URL batch helper")

    def _scan_nwb(**kwargs: object) -> object:
        scan_calls.append(kwargs)
        return sentinel

    def _read_table(*args: object, **kwargs: object) -> object:
        raise AssertionError("scan_dandiset must delegate table reads to scan_nwb")

    monkeypatch.setattr(dandi, "_get_session", lambda: session)
    monkeypatch.setattr(dandi, "_get_dandiset_assets", _get_dandiset_assets)
    monkeypatch.setattr(dandi, "_get_asset_s3_urls", _get_asset_s3_urls)
    monkeypatch.setattr(dandi, "_get_asset_s3_url", _get_asset_s3_url)
    monkeypatch.setattr(dandi.lazynwb.lazyframe, "scan_nwb", _scan_nwb)
    monkeypatch.setattr(dandi.lazynwb.tables, "get_df", _read_table)
    monkeypatch.setattr(
        dandi.lazynwb.tables,
        "_get_table_schema_with_catalog_snapshots",
        _read_table,
    )
    caplog.set_level(logging.DEBUG, logger="lazynwb.dandi")

    result = dandi.scan_dandiset(
        "000001",
        "/units",
        version="0.1.0",
        asset_filter=lambda asset: str(asset["path"]).startswith("sub-keep"),
        max_assets=2,
        raise_on_missing=True,
        ignore_errors=True,
        infer_schema_length=3,
        disable_progress=True,
    )

    assert result is sentinel
    assert selected_assets == [assets[2], assets[0]]
    assert len(scan_calls) == 1
    scan_call = scan_calls[0]
    assert tuple(scan_call["source"]) == (
        "s3://bucket/asset-a.nwb",
        "s3://bucket/asset-b.nwb",
    )
    assert scan_call["table_path"] == "/units"
    assert scan_call["raise_on_missing"] is True
    assert scan_call["ignore_errors"] is True
    assert scan_call["infer_schema_length"] == 3
    assert scan_call["disable_progress"] is True
    assert "000001" in caplog.text
    assert "0.1.0" in caplog.text
    assert "asset_count=5" in caplog.text
    assert (
        "selected_asset_paths=('sub-keep-a/session.nwb', 'sub-keep-b/session.nwb')"
        in caplog.text
    )
    assert "delegating DANDI scan to scan_nwb" in caplog.text


def test_scan_dandiset_empty_filtered_result_is_clear(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    assets = [{"asset_id": "asset-a", "path": "sub-a/session.nwb"}]

    def _get_dandiset_assets(
        dandiset_id: str,
        version: str | None = None,
        order: str = "path",
        *,
        session: object | None = None,
    ) -> list[dict[str, object]]:
        assert dandiset_id == "000001"
        assert version == "0.1.0"
        assert order == "path"
        assert session is not None
        return assets

    def _get_asset_s3_urls(*args: object, **kwargs: object) -> list[str]:
        raise AssertionError("empty DANDI scan selections should not resolve URLs")

    def _scan_nwb(**kwargs: object) -> object:
        raise AssertionError("empty DANDI scan selections should not call scan_nwb")

    monkeypatch.setattr(dandi, "_get_session", lambda: object())
    monkeypatch.setattr(dandi, "_get_dandiset_assets", _get_dandiset_assets)
    monkeypatch.setattr(dandi, "_get_asset_s3_urls", _get_asset_s3_urls)
    monkeypatch.setattr(dandi.lazynwb.lazyframe, "scan_nwb", _scan_nwb)
    caplog.set_level(logging.DEBUG, logger="lazynwb.dandi")

    with pytest.raises(ValueError) as exc_info:
        dandi.scan_dandiset(
            "000001",
            "/units",
            version="0.1.0",
            asset_filter=lambda asset: False,
        )

    message = str(exc_info.value)
    assert "No NWB assets selected" in message
    assert "dandiset 000001" in message
    assert "version 0.1.0" in message
    assert "asset_filter" in message
    assert "Found 1 assets, 1 NWB assets, selected 0." in message
    assert "selected_asset_paths=()" in caplog.text


def test_dandi_001637_sample_registry_uses_fixed_small_assets() -> None:
    assert dandi_sample._DANDI_SAMPLE_DANDISET_ID == "001637"
    assert dandi_sample._DANDI_SAMPLE_VERSION == "draft"
    assert dandi_sample._sample_asset_ids() == (
        "ca248278-e1b2-4896-ad1c-900e4506cd04",
        "1e37bc82-fd23-4cb5-a253-e794cea932ba",
    )
    assert dandi_sample._sample_asset_paths() == (
        "sub-830849/"
        "sub-830849_ses-ecephys-830849-2026-03-07-09-48-16_ecephys.nwb",
        "sub-830795/"
        "sub-830795_ses-ecephys-830795-2026-02-25-16-03-31_ecephys.nwb",
    )
    assert all(
        asset.max_bounded_read_bytes <= 1024 * 1024
        for asset in dandi_sample._DANDI_SAMPLE_ASSETS
    )


def test_dandi_001637_sample_resolver_uses_fixed_ids_and_redacts_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[tuple[str, str, str | None]] = []

    def _get_asset_s3_url(
        dandiset_id: str,
        asset_id: str,
        version: str | None = None,
    ) -> str:
        calls.append((dandiset_id, asset_id, version))
        return (
            f"https://dandiarchive.s3.amazonaws.com/blobs/{asset_id}"
            "?X-Amz-Credential=secret&X-Amz-Signature=also-secret"
        )

    monkeypatch.setattr(dandi_sample.dandi, "_get_asset_s3_url", _get_asset_s3_url)
    caplog.set_level(logging.DEBUG, logger=dandi_sample._LOGGER_NAME)

    resolved_assets = dandi_sample._resolve_sample_asset_urls()

    assert calls == [
        (
            dandi_sample._DANDI_SAMPLE_DANDISET_ID,
            asset.asset_id,
            dandi_sample._DANDI_SAMPLE_VERSION,
        )
        for asset in dandi_sample._DANDI_SAMPLE_ASSETS
    ]
    assert tuple(resolved.asset.asset_id for resolved in resolved_assets) == (
        dandi_sample._sample_asset_ids()
    )
    assert dandi_sample._DANDI_SAMPLE_DANDISET_ID in caplog.text
    assert dandi_sample._DANDI_SAMPLE_VERSION in caplog.text
    for asset in dandi_sample._DANDI_SAMPLE_ASSETS:
        assert asset.asset_id in caplog.text
        assert asset.path in caplog.text
    assert "source_url=https://dandiarchive.s3.amazonaws.com" in caplog.text
    assert "?<redacted>" in caplog.text
    assert "secret" not in caplog.text
    assert "X-Amz-Signature" not in caplog.text


def test_dandi_001637_sample_cached_url_logging_redacts_query_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    resolved_assets = (
        dandi_sample._DandiSampleResolvedAsset(
            asset=dandi_sample._DANDI_SAMPLE_ASSETS[0],
            source_url=(
                "https://user:password@dandiarchive.s3.amazonaws.com/blobs/sample"
                "?token=secret#fragment-secret"
            ),
        ),
    )
    caplog.set_level(logging.DEBUG, logger=dandi_sample._LOGGER_NAME)

    dandi_sample._log_resolved_sample_assets(resolved_assets, cache_scope="session")

    assert dandi_sample._DANDI_SAMPLE_DANDISET_ID in caplog.text
    assert dandi_sample._DANDI_SAMPLE_VERSION in caplog.text
    assert dandi_sample._DANDI_SAMPLE_ASSETS[0].asset_id in caplog.text
    assert dandi_sample._DANDI_SAMPLE_ASSETS[0].path in caplog.text
    assert "source_url=https://<redacted>@dandiarchive.s3.amazonaws.com" in caplog.text
    assert "?<redacted>#<redacted>" in caplog.text
    assert "password" not in caplog.text
    assert "secret" not in caplog.text
    assert "token" not in caplog.text
