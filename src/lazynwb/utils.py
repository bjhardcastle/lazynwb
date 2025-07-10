from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing
import os

logger = logging.getLogger(__name__)

thread_pool_executor: concurrent.futures.ThreadPoolExecutor | None = None
process_pool_executor: concurrent.futures.ProcessPoolExecutor | None = None


def get_threadpool_executor() -> concurrent.futures.ThreadPoolExecutor:
    global thread_pool_executor
    if thread_pool_executor is None:
        thread_pool_executor = concurrent.futures.ThreadPoolExecutor()
    return thread_pool_executor


def get_processpool_executor() -> concurrent.futures.ProcessPoolExecutor:
    global process_pool_executor
    if process_pool_executor is None:
        process_pool_executor = concurrent.futures.ProcessPoolExecutor(
            mp_context=(
                multiprocessing.get_context("spawn") if os.name == "posix" else None
            )
        )
    return process_pool_executor


def normalize_internal_file_path(path: str) -> str:
    """
    Normalize the internal file path for an NWB file.

    - remove leading '/' if present
    """
    return path.removeprefix("/") or "/"  # ensure at least root path is returned


if __name__ == "__main__":
    from npc_io import testmod

    testmod()
