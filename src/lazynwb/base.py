from __future__ import annotations

import concurrent.futures
import contextlib
import datetime
import inspect
import logging
from collections.abc import Iterable
from typing import Any

import npc_io
import pandas as pd
import tqdm

import lazynwb.file_io
import lazynwb.funcs

logger = logging.getLogger(__name__)


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

    _file: lazynwb.file_io.LazyFile

    def __init__(
        self,
        path: npc_io.PathLike,
        fsspec_storage_options: dict[str, Any] | None = None,
    ) -> None:
        self._file = lazynwb.file_io.LazyFile(
            path=path, fsspec_storage_options=fsspec_storage_options
        )

    def __repr__(self) -> str:
        return f"LazyNWB({self._file._path!r})"

    def _repr_html_(self) -> str:
        main_info = self._to_dict()
        subject_info = self.subject._to_dict()
        paths = self.describe().get("paths", [])

        html = f"""
        <h3>NWB file: {self._file._path}</h3>
        <ul>
        """
        for key, value in main_info.items():
            if isinstance(value, list):
                value = ", ".join(map(str, value)) or "[]"
            html += f"<li><strong>{key}:</strong> {value}</li>"
        html += "</ul>"

        html += "<h4>Subject</h4><ul>"
        for key, value in subject_info.items():
            if isinstance(value, list):
                value = ", ".join(map(str, value))
            html += f"<li><strong>{key}:</strong> {value}</li>"
        html += "</ul>"

        html += "<h4>Paths</h4><details><summary>Click to expand</summary><ul>"
        for path in paths:
            html += f"<li>{path}</li>"
        html += "</ul></details>"

        return html

    @property
    def identifier(self) -> str:
        return LazyComponent(self._file).identifier

    @property
    def subject(self) -> Subject:
        return Subject(self._file, "/general/subject")

    @property
    def session_start_time(self) -> datetime.datetime:
        return LazyComponent(self._file).session_start_time

    @property
    def session_id(self) -> str:
        return LazyComponent(self._file, "/general").session_id

    @property
    def session_description(self) -> str:
        return LazyComponent(self._file).session_description

    @property
    def trials(self) -> pd.DataFrame:
        return lazynwb.funcs.get_df(self._file, table_path="/intervals/trials")

    @property
    def epochs(self) -> pd.DataFrame:
        return lazynwb.funcs.get_df(self._file, table_path="/intervals/epochs")

    @property
    def electrodes(self) -> pd.DataFrame:
        return lazynwb.funcs.get_df(
            self._file, table_path="/general/extracellular_ephys/electrodes"
        )

    @property
    def units(self) -> pd.DataFrame:
        return lazynwb.funcs.get_df(
            self._file,
            table_path="/units",
            exclude_array_columns=True,
        ).pipe(lazynwb.funcs.merge_array_column, "obs_intervals")

    @property
    def experiment_description(self) -> str:
        return LazyComponent(self._file, "/general").experiment_description

    @property
    def experimenter(self) -> str:
        return LazyComponent(self._file, "/general").experimenter

    @property
    def lab(self) -> str:
        return LazyComponent(self._file, "/general").lab

    @property
    def institution(self) -> str:
        return LazyComponent(self._file, "/general").institution

    @property
    def related_publications(self) -> str:
        return LazyComponent(self._file, "/general").related_publications

    @property
    def keywords(self) -> list[str]:
        k: str | Iterable[str] | None = LazyComponent(self._file, "/general").keywords
        if k is None:
            return []
        if isinstance(k, str):
            k = [k]
        return list(k)

    @property
    def notes(self) -> str:
        return LazyComponent(self._file, "/general").notes

    @property
    def data_collection(self) -> str:
        return LazyComponent(self._file, "/general").data_collection

    @property
    def surgery(self) -> str:
        return LazyComponent(self._file, "/general").surgery

    @property
    def pharmacology(self) -> str:
        return LazyComponent(self._file, "/general").pharmacology

    @property
    def virus(self) -> str:
        return LazyComponent(self._file, "/general").virus

    @property
    def source_script(self) -> str:
        return LazyComponent(self._file, "/general").source_script

    @property
    def source_script_file_name(self) -> str:
        return LazyComponent(self._file, "/general").source_script_file_name

    def _to_dict(self) -> dict[str, str | list[str] | datetime.datetime]:
        def _get_attr_names(obj: Any) -> list[str]:
            return [
                name
                for name, prop in obj.__class__.__dict__.items()
                if isinstance(prop, property)
                and inspect.signature(prop.fget).return_annotation
                in ("str", "list[str]", "datetime.datetime")
            ]

        return {name: getattr(self, name) for name in _get_attr_names(self)}

    def get_timeseries(
        self, search_term: str | None = None
    ) -> dict[str, lazynwb.funcs.TimeSeries]:
        return lazynwb.funcs.get_timeseries(self._file, search_term=search_term)

    def describe(self) -> dict[str, Any]:
        return {
            **self._to_dict(),
            **self.subject._to_dict(),
            "paths": list(lazynwb.funcs._get_internal_file_paths(self._file).keys()),
        }


class LazyComponent:
    def __init__(
        self,
        file: lazynwb.file_io.LazyFile,
        path: str | None = None,
    ) -> None:
        self._file = file
        if path is None:
            path = ""
        self._path = path.strip().strip("/")

    def __getattr__(self, name: str) -> Any:
        return interpret(self._file, f"{self._path}/{name}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._file}, {self._path!r})"

def interpret(file: lazynwb.file_io.FileAccessor, path: str) -> Any:
    """Read attribute from NWB file and interpret it as the appropriate Python object."""
    path = lazynwb.file_io.normalize_internal_file_path(path)
    v = file.get(path, None)
    if v is None:
        return None
    if not getattr(v, "shape", True):
        v = [v[()]]
    if isinstance(v[0], bytes):
        s: str = v[0].decode()
        with contextlib.suppress(ValueError):
            return datetime.datetime.fromisoformat(s)
        if s.startswith('[') and s.endswith(']') and s.count('[') == s.count(']') == 1:
            with contextlib.suppress(Exception):
                return eval(s)
        return s
    if len(v) > 1:
        return v
    return v[0]

class Subject(LazyComponent):
    age: str | None
    """The age of the subject. The ISO 8601 Duration format is recommended, e.g., “P90D” for 90 days old."""

    age__reference: str | None
    """Age is with reference to this event. Can be `birth` or `gestational`. If reference is omitted, then `birth` is implied. Value can be None when read from an NWB file with schema version 2.0 to 2.5 where age__reference is missing."""

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

    def _to_dict(self) -> dict[str, str | list[str]]:
        return {name: getattr(self, name) for name in self.__class__.__annotations__}


def get_metadata_df(
    nwb_path_or_paths: npc_io.PathLike | Iterable[npc_io.PathLike],
    disable_progress: bool = False,
) -> pd.DataFrame:
    if isinstance(nwb_path_or_paths, str) or not isinstance(
        nwb_path_or_paths, Iterable
    ):
        paths = (nwb_path_or_paths,)
    else:
        paths = tuple(nwb_path_or_paths)

    def _get_metadata_df_helper(nwb_path: npc_io.PathLike) -> dict[str, Any]:
        nwb = LazyNWB(nwb_path)
        return {
            **nwb._to_dict(),
            **nwb.subject._to_dict(),
            lazynwb.funcs.NWB_PATH_COLUMN_NAME: nwb._file._path.as_posix(),
        }

    future_to_path = {}
    for path in paths:
        future = lazynwb.funcs.get_threadpool_executor().submit(
            _get_metadata_df_helper,
            nwb_path=path,
        )
        future_to_path[future] = path
    futures = concurrent.futures.as_completed(future_to_path)
    if not disable_progress:
        futures = tqdm.tqdm(
            futures,
            total=len(future_to_path),
            desc="Getting metadata",
            unit="file",
            ncols=80,
        )
    records = []
    for future in futures:
        path = future_to_path[future]
        try:
            records.append(future.result())
        except:
            logger.error(f"Error processing {path}:")
            raise
    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    from npc_io import testmod

    testmod()
