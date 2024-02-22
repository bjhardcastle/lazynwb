import io

import h5py
import npc_io
import zarr


def lazy_open_nwb(path: npc_io.PathLike) -> h5py.File | zarr.Group:
    """
    Open a file that meets the NWB spec, minimizing the amount of data/metadata read.

    - file is opened in read-only mode
    - file is not closed when the function returns
    - currently supports NWB files saved in .hdf5 and .zarr format

    Examples:
        >>> nwb = lazy_open_nwb('https://dandiarchive.s3.amazonaws.com/blobs/f78/fe2/f78fe2a6-3dc9-4c12-a288-fbf31ce6fc1c')
        # >>> nwb = lazy_open_nwb('')
    """
    path = npc_io.from_pathlike(path)
    if "zarr" in path.as_posix():
        return zarr.open(store=path, mode="r")
    return h5py.File(path.open(mode="rb"), mode="r")


if __name__ == "__main__":
    from npc_io import testmod

    testmod()
