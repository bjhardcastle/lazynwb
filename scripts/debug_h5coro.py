from __future__ import annotations

import sys
import time

import requests

import h5coro
from h5coro.webdriver import HTTPDriver


DEFAULT_S3_PATH = (
    "s3://aind-scratch-data/dynamic-routing/cache/nwb/v0.0.272/620263_2022-07-26.nwb"
)
DEFAULT_TABLE_PATH = "units"


def timed_step(label: str, fn):
    print(f"\n{label}", flush=True)
    t0 = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"{label} failed after {elapsed:.2f}s: {exc!r}", flush=True)
        raise
    elapsed = time.perf_counter() - t0
    print(f"{label} succeeded in {elapsed:.2f}s", flush=True)
    return result, elapsed


def s3_to_https(path: str) -> str:
    if path.startswith("s3://"):
        bucket = path[5:].split("/")[0]
        key = "/".join(path[5:].split("/")[1:])
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return path


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_S3_PATH
    table_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TABLE_PATH
    https_url = s3_to_https(path)

    print(f"path={path}")
    print(f"https_url={https_url}")
    print(f"table_path={table_path}")

    try:
        response, _ = timed_step(
            "[1/4] HEAD request with requests",
            lambda: requests.head(https_url, timeout=10, allow_redirects=True),
        )
        print(f"HEAD status={response.status_code}", flush=True)
    except Exception as exc:
        print(f"HEAD failed: {exc!r}", flush=True)

    try:
        h5c, _ = timed_step(
            "[2/4] h5coro open",
            lambda: h5coro.H5Coro(
                https_url,
                HTTPDriver,
                credentials={},
                errorChecking=False,
                verbose=False,
            ),
        )
    except Exception as exc:
        print(f"h5coro open failed: {exc!r}", flush=True)
        return 1

    try:
        result, _ = timed_step(
            "[3/4] h5coro inspectPath",
            lambda: h5c.inspectPath(f'/{table_path.strip("/")}'),
        )
        if result is None:
            print("inspectPath returned None", flush=True)
            return 1
        group_keys = sorted(result[0])
        print(f"group key count={len(group_keys)}", flush=True)
        print(f"first keys={group_keys[:20]}", flush=True)
    except Exception as exc:
        print(f"inspectPath failed: {exc!r}", flush=True)
        return 1

    dataset_paths = [f"/{table_path.strip('/')}/{name}" for name in group_keys]
    try:
        _, _ = timed_step(
            "[4/4] h5coro readDatasets(metaOnly=True)",
            lambda: h5c.readDatasets(
                dataset_paths,
                block=True,
                metaOnly=True,
                enableAttributes=False,
            ),
        )
        populated = sum(
            1 for name in group_keys if h5c.metadataTable.get(f"{table_path.strip('/')}/{name}") is not None
        )
        print(f"metadata populated for {populated}/{len(group_keys)} datasets", flush=True)
    except Exception as exc:
        print(f"readDatasets failed: {exc!r}", flush=True)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
