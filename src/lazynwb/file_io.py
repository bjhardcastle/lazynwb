from __future__ import annotations

import contextlib
from typing import Any

import h5py
import npc_io
import remfile
import upath
import zarr


def open(path: npc_io.PathLike, use_remfile: bool = False, **fsspec_storage_options: Any) -> h5py.File | zarr.Group:
    """
    Open a file that meets the NWB spec, minimizing the amount of data/metadata read.

    - file is opened in read-only mode
    - file is not closed when the function returns
    - currently supports NWB files saved in .hdf5 and .zarr format

    Examples:
        >>> nwb = open('https://dandiarchive.s3.amazonaws.com/blobs/f78/fe2/f78fe2a6-3dc9-4c12-a288-fbf31ce6fc1c')
        >>> nwb = open('https://dandiarchive.s3.amazonaws.com/blobs/f78/fe2/f78fe2a6-3dc9-4c12-a288-fbf31ce6fc1c', use_remfile=False)
        >>> nwb = open('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/39490bff-87c9-4ef2-b408-36334e748ac6/nwb/ecephys_620264_2022-08-02_15-39-59_experiment1_recording1.nwb')
    """
    path = npc_io.from_pathlike(path)
    if path.protocol == "s3":
        fsspec_storage_options.setdefault('anon', True)
    path = upath.UPath(path, **fsspec_storage_options)
    
    # zarr ------------------------------------------------------------- #
    # there's no file-name convention for what is a zarr file, so we have to try opening it and see if it works
    # - zarr.open() is fast regardless of size
    with contextlib.suppress(Exception):
        return zarr.open(store=path, mode="r")

    # hdf5 ------------------------------------------------------------- #
    if not use_remfile:
        # conventional method is open the file with fsspec and then pass the file handle to h5py:
        file = path.open(mode="rb", cache_type="first")
    else:
        # but using remfile is slightly faster in practice, at least for the initial opening:
        file = remfile.File(url=path.as_posix())
    return h5py.File(file, mode="r")


if __name__ == "__main__":
    from npc_io import testmod

    testmod()
