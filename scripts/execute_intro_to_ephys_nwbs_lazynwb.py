from __future__ import annotations

import importlib.util
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

logger = logging.getLogger(__name__)

_NOTEBOOK_RELATIVE_PATH = pathlib.Path("notebooks") / "intro_to_ephys_nwbs-lazynwb.ipynb"
_HTML_RELATIVE_PATH = _NOTEBOOK_RELATIVE_PATH.with_suffix(".html")
_NBCONVERT_REQUIREMENT = "nbconvert>=7.16.6"


def main() -> int:
    _configure_logging()

    if len(sys.argv) != 1:
        logger.error("This script does not accept arguments.")
        return 2

    repo_root = _repo_root()
    notebook_path = repo_root / _NOTEBOOK_RELATIVE_PATH
    html_path = repo_root / _HTML_RELATIVE_PATH

    logger.debug("Resolved repository root: %s", repo_root)
    logger.debug("Notebook input path: %s", notebook_path)
    logger.debug("HTML output path: %s", html_path)

    if not notebook_path.exists():
        logger.error("Notebook not found: %s", notebook_path)
        return 1

    html_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return _execute_notebook(
            repo_root=repo_root, notebook_path=notebook_path, html_path=html_path
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1


def _configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        level=logging.DEBUG,
    )


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _execute_notebook(
    *,
    repo_root: pathlib.Path,
    notebook_path: pathlib.Path,
    html_path: pathlib.Path,
) -> int:
    env = _build_notebook_environment()
    command = _build_nbconvert_command(notebook_path=notebook_path, html_path=html_path)
    logger.info("Executing notebook and rendering HTML.")
    logger.debug("Running command: %s", subprocess.list2cmdline(command))
    logger.debug("Using LAZYNWB_CATALOG_CACHE_PATH=%s", env["LAZYNWB_CATALOG_CACHE_PATH"])

    start = time.perf_counter()
    # Pass the cache path to the immediate subprocess. In the uv fallback path,
    # that means uv receives it and forwards it to nbconvert and the notebook kernel.
    completed_process = subprocess.run(command, cwd=repo_root, env=env, check=False)
    elapsed = time.perf_counter() - start

    if completed_process.returncode == 0:
        logger.info("Wrote %s in %.2f seconds.", html_path, elapsed)
    else:
        logger.error(
            "Notebook execution failed with exit code %d after %.2f seconds.",
            completed_process.returncode,
            elapsed,
        )

    return int(completed_process.returncode)


def _build_notebook_environment() -> dict[str, str]:
    lazynwb_demo_cache_dir = pathlib.Path(tempfile.mkdtemp(prefix="lazynwb-demo-cache-"))
    catalog_cache_path = str(lazynwb_demo_cache_dir / "catalog.sqlite")
    os.environ["LAZYNWB_CATALOG_CACHE_PATH"] = catalog_cache_path

    env = os.environ.copy()
    env["LAZYNWB_CATALOG_CACHE_PATH"] = catalog_cache_path
    return env


def _build_nbconvert_command(
    *, notebook_path: pathlib.Path, html_path: pathlib.Path
) -> list[str]:
    nbconvert_args = [
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "html",
        "--execute",
        "--output",
        html_path.stem,
        "--output-dir",
        str(html_path.parent),
        "--ExecutePreprocessor.kernel_name=python3",
        "--ExecutePreprocessor.timeout=-1",
        str(notebook_path),
    ]

    if importlib.util.find_spec("nbconvert") is not None:
        logger.debug("Using nbconvert from the current Python environment.")
        return [sys.executable, *nbconvert_args]

    uv_path = shutil.which("uv")
    if uv_path is None:
        raise RuntimeError(
            "nbconvert is not installed and uv is not available. "
            f"Install {_NBCONVERT_REQUIREMENT!r} or run this repository with uv."
        )

    logger.debug(
        "nbconvert is not installed in the current Python environment; using uv with %s.",
        _NBCONVERT_REQUIREMENT,
    )
    return [uv_path, "run", "--with", _NBCONVERT_REQUIREMENT, "python", *nbconvert_args]


if __name__ == "__main__":
    raise SystemExit(main())
