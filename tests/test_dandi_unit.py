from __future__ import annotations

import collections.abc
import concurrent.futures
import logging

import pytest

import lazynwb.dandi as dandi
import lazynwb.utils as utils


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
