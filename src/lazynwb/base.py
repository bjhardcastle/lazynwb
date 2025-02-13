from __future__ import annotations

import contextlib
import dataclasses
import datetime
import enum
from typing import Any, Iterable

import h5py
import npc_io
import polars as pl
import upath
import zarr

import lazynwb.file_io
import lazynwb.funcs

class LazyNWB:
    """
    High-level interface for accessing components of an NWB file.

    - initialize with a path to an NWB file or an open h5py.File, h5py.Group, or
    zarr.Group object
    
    Examples:
        >>> nwb = LazyNWB('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/39490bff-87c9-4ef2-b408-36334e748ac6/nwb/ecephys_620264_2022-08-02_15-39-59_experiment1_recording1.nwb')
        >>> nwb.subject.date_of_birth
        datetime.datetime(2022, 2, 3, 0, 0, tzinfo=datetime.timezone.utc)
        >>> nwb.session_start_time
        datetime.datetime(2022, 8, 2, 15, 39, 59, tzinfo=datetime.timezone.utc)
    """
    _file: LazyFile

    def __init__(
        self, 
        path_or_data: npc_io.PathLike | h5py.File | h5py.Group | zarr.Group,
        fsspec_storage_options: dict[str, Any] | None = None,
    ) -> None:
        self._file = LazyFile(path_or_data, fsspec_storage_options)

    @property
    def subject(self) -> Subject:
        return Subject(self._file, 'general/subject')
    
    @property
    def session_start_time(self) -> datetime.datetime:
        return LazyComponent(self._file).session_start_time
    
    @property
    def session_id(self) -> str:
        return LazyComponent(self._file, 'general').session_id
    
    @property
    def session_description(self) -> str:
        return LazyComponent(self._file).session_description
    
    @property
    def units(self) -> pl.LazyFrame:
        return lazynwb.funcs.get_units(self._file)

    @property
    def experiment_description(self) -> str:
        return LazyComponent(self._file, 'general').experiment_description
    
    @property
    def experimenter(self) -> str:
        return LazyComponent(self._file, 'general').experimenter
    
    @property
    def lab(self) -> str:
        return LazyComponent(self._file, 'general').lab
    
    @property
    def institution(self) -> str:
        return LazyComponent(self._file, 'general').institution
    
    @property
    def related_publications(self) -> str:
        return LazyComponent(self._file, 'general').related_publications
    
    @property
    def keywords(self) -> str | None:
        k: str | Iterable[str] | None = LazyComponent(self._file, 'general').keywords
        if k is None:
            return None
        if isinstance(k, str):
            k = [k]
        return list(k)
    
    @property
    def notes(self) -> str:
        return LazyComponent(self._file, 'general').notes
    
    @property
    def data_collection(self) -> str:
        return LazyComponent(self._file, 'general').data_collection
    
    @property
    def surgery(self) -> str:
        return LazyComponent(self._file, 'general').surgery
    
    @property
    def pharmacology(self) -> str:
        return LazyComponent(self._file, 'general').pharmacology
    
    @property
    def virus(self) -> str:
        return LazyComponent(self._file, 'general').virus
    
    @property
    def source_script(self) -> str:
        return LazyComponent(self._file, 'general').source_script
    
    @property
    def source_script_file_name(self) -> str:
        return LazyComponent(self._file, 'general').source_script_file_name
    
    
    
class LazyComponent:
    def __init__(
        self,
        file: LazyFile,
        path: str | None = None,
    ) -> None:
        self._file = file
        if path is None:
            path = ''
        self._path = path.strip().strip('/')
            
    def __getattr__(self, name: str) -> Any:
        path = f"{self._path}/{name}" if self._path else name
        v = self._file.get(path, None)
        if v is None:
            return None
        if isinstance(v[0], bytes):
            s = v[0].decode()
            try:
                return datetime.datetime.fromisoformat(s)
            except ValueError:
                return s
        if len(v) > 1:
            return v
        return v[0]
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._file}, {self._path!r})"
    
    
class Subject(LazyComponent):
    age: str | None
    """The age of the subject. The ISO 8601 Duration format is recommended, e.g., “P90D” for 90 days old."""
    
    age__reference: str | None
    """Age is with reference to this event. Can be ‘birth’ or ‘gestational’. If reference is omitted, then ‘birth’ is implied. Value can be None when read from an NWB file with schema version 2.0 to 2.5 where age__reference is missing."""
    
    description: str | None
    """A description of the subject, e.g., “mouse A10”."""
    
    genotype: str | None
    """The genotype of the subject, e.g., “Sst-IRES-Cre/wt;Ai32(RCL-ChR2(H134R)_EYFP"""
    
    sex: str | None
    """The sex of the subject. Using “F” (female), “M” (male), “U” (unknown), or
    “O” (other) is recommended."""
    
    species: str | None
    """The species of the subject. The formal latin binomal name is recommended, e.g., “Mus musculus”."""
    
    subject_id: str | None
    """A unique identifier for the subject, e.g., “A10”."""
    
    weight: str | None
    """The weight of the subject, including units. Using kilograms is recommended. e.g., “0.02 kg”. If a float is provided, then the weight will be stored as “[value] kg”."""
    
    strain: str | None
    """The strain of the subject, e.g., “C57BL/6J”."""
    
    date_of_birth: datetime.datetime | None
    """The datetime of the date of birth. May be supplied instead of age."""
    
class LazyFile:
    """
    A lazy file object (h5py.File, h5py.Group, or zarr.Group) that can be used to
    access components via their standard dict-like interface or as instance attributes.
    
    - initialize with a path to an NWB file or an open h5py.File, h5py.Group, or
    zarr.Group object
    
    Examples:
        >>> file = LazyFile('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/39490bff-87c9-4ef2-b408-36334e748ac6/nwb/ecephys_620264_2022-08-02_15-39-59_experiment1_recording1.nwb')
        >>> file.units
        <zarr.hierarchy.Group '/units' read-only>
        >>> file['units']
        <zarr.hierarchy.Group '/units' read-only>
        >>> file.units.spike_times
        <zarr.core.Array '/units/spike_times' (18185563,) float64 read-only>
        >>> file.units.spike_times_index[0]
        6966
        >>> file.units.id[0]
        0
        >>> 'spike_times' in file.units
        True
        >>> next(iter(file))
        'acquisition'
        >>> next(iter(file.units))
        'amplitude'
    """
    class HDMFBackend(enum.StrEnum):
        HDF5 = "hdf5"
        ZARR = "zarr"

    _path: upath.UPath | None
    _data: h5py.File | h5py.Group | zarr.Group
    _backend: HDMFBackend

    def __init__(
        self,
        path_or_data: npc_io.PathLike | h5py.File | h5py.Group | zarr.Group,
        fsspec_storage_options: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(path_or_data, (h5py.File, h5py.Group, zarr.Group)):
            self._path = None
            self._data = path_or_data
        else:
            self._path = npc_io.from_pathlike(path_or_data)
            self._data = lazynwb.file_io.open(self._path, **(fsspec_storage_options or {}))
        self._backend = self.get_hdmf_backend()

    def get_hdmf_backend(self) -> HDMFBackend:
        if isinstance(self._data, (h5py.File, h5py.Group)):
            return self.HDMFBackend.HDF5
        elif isinstance(self._data, zarr.Group):
            return self.HDMFBackend.ZARR
        raise ValueError(f"Unknown backend for {self._data!r}")

    def __getattr__(self, name) -> Any:
        # for built-in properties/methods of the underlying h5py/zarr object:
        with contextlib.suppress(AttributeError):
            return getattr(self._data, name)
        
        # for components of the NWB file:
        with contextlib.suppress(KeyError):
            component = self._data[name]
            # provide a new instance of the class for conveninet access to components:
            #! this is now slower than using __getitem__ directly
            if isinstance(component, (h5py.Group, zarr.Group)):
                return LazyFile(component)
            return component
        raise AttributeError(f"No attribute named {name!r} in NWB file")

    def __getitem__(self, name) -> Any:
        return self._data[name]

    def __contains__(self, name) -> bool:
        return name in self._data

    def __iter__(self):
        return iter(self._data)

    def __repr__(self) -> str:
        if self._path is not None:
            return f"{self.__class__.__name__}({self._path.as_posix()!r})"
        return repr(self._data)

    def __enter__(self) -> LazyFile:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        if self._path is not None:
            if isinstance(self._data, h5py.File):
                self._data.close()
            elif isinstance(self._data, zarr.Group):
                self._data.store.close()

if __name__ == "__main__":
    from npc_io import testmod

    testmod()
