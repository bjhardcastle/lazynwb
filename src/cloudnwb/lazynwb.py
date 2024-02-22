import npc_io
import h5py
import zarr

import cloudnwb.io

class LazyNWB():
    def __init__(self, path: npc_io.PathLike) -> None:
        self._path = npc_io.from_pathlike(path)
        self._nwb = cloudnwb.io.lazy_open_nwb(self._path)
        
    def __getattr__(self, name):
        if name in self.nwbfile.fields:
            return self.nwbfile.fields[name]
        else:
            raise AttributeError(f'NWB file does not have attribute {name}')