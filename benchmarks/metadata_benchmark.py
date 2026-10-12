"""Measure metadata reads on the sources in a NeuroDataBench benchmark JSON.

Run with the same interpreter/dependencies for both checkouts, for example::

    python benchmarks/metadata_benchmark.py --checkout . --benchmark path/to/benchmark.json

Each process starts cold and performs two reads. Output includes the imported
source path, actual backend-open counts, timing, and a metadata content hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time


def _main() -> None:
    """Run cold and repeated metadata reads using one explicit checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    sys.path.insert(0, str(checkout / "src"))

    import lazynwb
    import lazynwb.file_io as file_io

    sources = json.loads(args.benchmark.read_text(encoding="utf-8"))["data_sources"]
    lazynwb.config.anon = True
    opens: list[str] = []
    original = file_io._open_file

    def _counted_open(path: object) -> object:
        """Count backend opens while preserving the real remote read path."""
        opens.append(str(path))
        return original(path)

    file_io._open_file = _counted_open
    for repeat in range(2):
        opens.clear()
        start = time.perf_counter()
        frame = lazynwb.get_metadata_df(sources, disable_progress=True, as_polars=False)
        elapsed = time.perf_counter() - start
        normalized = frame.sort_values(lazynwb.NWB_PATH_COLUMN_NAME).to_json(
            orient="records", date_format="iso"
        )
        print(
            json.dumps(
                {
                    "source": str(lazynwb.__file__),
                    "benchmark": str(args.benchmark),
                    "cache": "cold" if repeat == 0 else "warm",
                    "seconds": elapsed,
                    "file_opens": len(opens),
                    "rows": len(frame),
                    "metadata_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    _main()
