from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import multiprocessing
import os
import typing

logger = logging.getLogger(__name__)

TABLE_SHORTCUTS = {
    "trials": "intervals/trials",
    "epochs": "intervals/epochs",
    "electrodes": "general/extracellular_ephys/electrodes",
    "optophysiology": "general/optophysiology",
    "session": "general",
}


thread_pool_executor: concurrent.futures.ThreadPoolExecutor | None = None
process_pool_executor: concurrent.futures.ProcessPoolExecutor | None = None
_AsyncValueType = typing.TypeVar("_AsyncValueType")


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


def _run_async_value(
    coroutine: typing.Coroutine[object, object, _AsyncValueType],
) -> _AsyncValueType:
    """Run a private async operation from sync APIs, even inside an event loop."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    future = get_threadpool_executor().submit(asyncio.run, coroutine)
    return future.result()


def normalize_internal_file_path(path: str) -> str:
    """
    Normalize the internal file path for an NWB file.

    - remove leading '/' if present
    """
    path = path.removeprefix("/") or "/"  # ensure at least root path is returned
    return TABLE_SHORTCUTS.get(path, path)


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
