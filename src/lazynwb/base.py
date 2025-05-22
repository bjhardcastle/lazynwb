from __future__ import annotations

import concurrent.futures
import contextlib
import datetime
import inspect
import logging
import typing
from collections.abc import Iterable
from typing import Any, Literal, Protocol

import npc_io
import pandas as pd
import polars as pl
import tqdm

import lazynwb.file_io
import lazynwb.tables
import lazynwb.timeseries
import lazynwb.utils

logger = logging.getLogger(__name__)


def _cast(file: lazynwb.file_io.FileAccessor, path: str) -> Any:
    """Read attribute from NWB file and interpret it as the appropriate Python object."""
    path = lazynwb.utils.normalize_internal_file_path(path)
    v = file.get(path, None)
    if v is None:
        return None
    if not getattr(v, "shape", True):
        v = [v[()]]
    if isinstance(v[0], bytes):
        s: str = v[0].decode()
        with contextlib.suppress(ValueError):
            return datetime.datetime.fromisoformat(s)
        if s.startswith("[") and s.endswith("]") and s.count("[") == s.count("]") == 1:
            with contextlib.suppress(Exception):
                return eval(s)
        if len(v) > 1:
            return v.asstr()[:].tolist()
        return s
    if len(v) > 1:
        return v
    return v[0]


class LazyNWB:
    """
    PyNWB-like interface for accessing components of an NWB file.

    - initialize with a path to an NWB file or an open h5py.File, h5py.Group, or
    zarr.Group object

    - forwards attributes to the underlying NWB file accessor (h5py.File, h5py.Group), with
      intermediate objects used for convenient dot attr access. Will be slightly slower than
      accessing components directly with the NWB file accessor due to the overhead of creating
      python objects.

    Examples:
        >>> nwb = LazyNWB('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/00865745-db58-495d-9c5e-e28424bb4b97/nwb/ecephys_721536_2024-05-16_12-32-31_experiment1_recording1.nwb')
        >>> nwb.subject.date_of_birth
        datetime.datetime(2022, 2, 3, 0, 0, tzinfo=datetime.timezone.utc)
        >>> nwb.session_start_time
        datetime.datetime(2022, 8, 2, 15, 39, 59, tzinfo=datetime.timezone.utc)
    """

    _file: lazynwb.file_io.FileAccessor

    def __init__(
        self,
        path_or_accessor: npc_io.PathLike | lazynwb.file_io.FileAccessor,
        fsspec_storage_options: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(path_or_accessor, lazynwb.file_io.FileAccessor):
            self._file = path_or_accessor
        else:
            self._file = lazynwb.file_io.FileAccessor(
                path=path_or_accessor, fsspec_storage_options=fsspec_storage_options
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
        return _cast(self._file, "identifier")

    @property
    def subject(self) -> Subject:
        return Subject(self._file)

    @property
    def session_start_time(self) -> datetime.datetime:
        return _cast(self._file, "session_start_time")

    @property
    def session_id(self) -> str:
        return _cast(self._file, "/general/session_id")

    @property
    def session_description(self) -> str:
        return _cast(self._file, "session_description")

    @property
    def trials(self) -> pd.DataFrame:
        return lazynwb.tables.get_df(
            self._file, search_term="/intervals/trials", exact_path=True
        )

    @property
    def epochs(self) -> pd.DataFrame:
        return lazynwb.tables.get_df(
            self._file, search_term="/intervals/epochs", exact_path=True
        )

    @property
    def electrodes(self) -> pd.DataFrame:
        return lazynwb.tables.get_df(
            self._file,
            search_term="/general/extracellular_ephys/electrodes",
            exact_path=True,
        )

    @property
    def units(self) -> pd.DataFrame:
        return lazynwb.tables.get_df(
            self._file,
            search_term="/units",
            exclude_array_columns=True,
            exact_path=True,
        ).pipe(lazynwb.tables.merge_array_column, "obs_intervals")

    @property
    def experiment_description(self) -> str:
        return _cast(self._file, "/general/experiment_description")

    @property
    def experimenter(self) -> str:
        return _cast(self._file, "/general/experimenter")

    @property
    def lab(self) -> str:
        return _cast(self._file, "/general/lab")

    @property
    def institution(self) -> str:
        return _cast(self._file, "/general/institution")

    @property
    def related_publications(self) -> str:
        return _cast(self._file, "/general/related_publications")

    @property
    def keywords(self) -> list[str]:
        k: str | Iterable[str] | None = _cast(self._file, "/general/keywords")
        if k is None:
            return []
        if isinstance(k, str):
            k = [k]
        return list(k)

    @property
    def notes(self) -> str:
        return _cast(self._file, "/general/notes")

    @property
    def data_collection(self) -> str:
        return _cast(self._file, "/general/data_collection")

    @property
    def surgery(self) -> str:
        return _cast(self._file, "/general/surgery")

    @property
    def pharmacology(self) -> str:
        return _cast(self._file, "/general/pharmacology")

    @property
    def virus(self) -> str:
        return _cast(self._file, "/general/virus")

    @property
    def source_script(self) -> str:
        return _cast(self._file, "/general/source_script")

    @property
    def source_script_file_name(self) -> str:
        return _cast(self._file, "/general/source_script_file_name")

    def _to_dict(self) -> dict[str, Any]:
        return to_dict(self)

    def get_timeseries(
        self, search_term: str | None = None
    ) -> lazynwb.timeseries.TimeSeries:
        return lazynwb.timeseries.get_timeseries(
            self._file, search_term=search_term, match_all=False
        )

    @typing.overload
    def get_df(
        self,
        search_term: str,
        exact_path: bool = False,
        include_column_names: str | Iterable[str] | None = None,
        exclude_column_names: str | Iterable[str] | None = None,
        exclude_array_columns: bool = True,
        use_process_pool: bool = False,
        disable_progress: bool = True,
        raise_on_missing: bool = True,
        ignore_errors: bool = False,
        as_polars: Literal[False] = False,
    ) -> pd.DataFrame: ...

    @typing.overload
    def get_df(
        self,
        search_term: str,
        exact_path: bool = False,
        include_column_names: str | Iterable[str] | None = None,
        exclude_column_names: str | Iterable[str] | None = None,
        exclude_array_columns: bool = True,
        use_process_pool: bool = False,
        disable_progress: bool = True,
        raise_on_missing: bool = True,
        ignore_errors: bool = False,
        as_polars: Literal[True] = True,
    ) -> pl.DataFrame: ...

    def get_df(
        self,
        search_term: str,
        exact_path: bool = False,
        include_column_names: str | Iterable[str] | None = None,
        exclude_column_names: str | Iterable[str] | None = None,
        exclude_array_columns: bool = True,
        use_process_pool: bool = False,
        disable_progress: bool = True,
        raise_on_missing: bool = True,
        ignore_errors: bool = False,
        as_polars: bool = False,
    ) -> pd.DataFrame | pl.DataFrame:
        return lazynwb.tables.get_df(
            nwb_data_sources=self._file,
            search_term=search_term,
            exact_path=exact_path,
            include_column_names=include_column_names,
            exclude_column_names=exclude_column_names,
            exclude_array_columns=exclude_array_columns,
            use_process_pool=use_process_pool,
            disable_progress=disable_progress,
            raise_on_missing=raise_on_missing,
            ignore_errors=ignore_errors,
            as_polars=as_polars,
        )  # type: ignore[call-overload]

    def describe(self) -> dict[str, Any]:
        return {
            **self._to_dict(),
            **self.subject._to_dict(),
            "paths": list(lazynwb.utils.get_internal_file_paths(self._file).keys()),
        }


class NWBComponent(Protocol):
    @property
    def _file(self) -> lazynwb.file_io.FileAccessor: ...


def to_dict(obj: NWBComponent) -> dict[str, str | list[str] | datetime.datetime]:
    def _get_attr_names(obj: Any) -> list[str]:
        return [
            name
            for name, prop in obj.__class__.__dict__.items()
            if isinstance(prop, property)
            and any(t in inspect.signature(prop.fget).return_annotation for t in ("str", "list[str]", "datetime.datetime"))  # type: ignore[arg-type]
        ]

    results = {}
    for name in _get_attr_names(obj):
        results[name] = getattr(obj, name)
    return results


class Subject:

    _file: lazynwb.file_io.FileAccessor

    def __init__(
        self,
        path_or_accessor: npc_io.PathLike | lazynwb.file_io.FileAccessor,
        fsspec_storage_options: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(path_or_accessor, lazynwb.file_io.FileAccessor):
            self._file = path_or_accessor
        else:
            self._file = lazynwb.file_io.FileAccessor(
                path=path_or_accessor, fsspec_storage_options=fsspec_storage_options
            )

    def __repr__(self) -> str:
        return f"Subject({self._file._path!r})"

    @property
    def age(self) -> str | None:
        """The age of the subject. The ISO 8601 Duration format is recommended, e.g., “P90D” for 90 days old."""
        return _cast(self._file, "/general/subject/age")

    @property
    def age__reference(self) -> str | None:
        """Age is with reference to this event. Can be `birth` or `gestational`. If reference is omitted, then `birth` is implied. Value can be None when read from an NWB file with schema version 2.0 to 2.5 where age__reference is missing."""
        return _cast(self._file, "/general/subject/age__reference")

    @property
    def description(self) -> str | None:
        """A description of the subject, e.g., “mouse A10”."""
        return _cast(self._file, "/general/subject/description")

    @property
    def genotype(self) -> str | None:
        """The genotype of the subject, e.g., “Sst-IRES-Cre/wt;Ai32(RCL-ChR2(H134R)_EYFP"""
        return _cast(self._file, "/general/subject/genotype")

    @property
    def sex(self) -> str | None:
        """The sex of the subject. Using “F” (female), “M” (male), “U” (unknown), or “O” (other) is recommended."""
        return _cast(self._file, "/general/subject/sex")

    @property
    def species(self) -> str | None:
        """The species of the subject. The formal latin binomal name is recommended, e.g., “Mus musculus”."""
        return _cast(self._file, "/general/subject/species")

    @property
    def subject_id(self) -> str | None:
        """A unique identifier for the subject, e.g., “A10”."""
        return _cast(self._file, "/general/subject/subject_id")

    @property
    def weight(self) -> str | None:
        """The weight of the subject, including units. Using kilograms is recommended. e.g., “0.02 kg”. If a float is provided, then the weight will be stored as “[value] kg”."""
        return _cast(self._file, "/general/subject/weight")

    @property
    def strain(self) -> str | None:
        """The strain of the subject, e.g., “C57BL/6J”."""
        return _cast(self._file, "/general/subject/strain")

    @property
    def date_of_birth(self) -> datetime.datetime | None:
        """The datetime of the date of birth. May be supplied instead of age."""
        return _cast(self._file, "/general/subject/date_of_birth")

    def _to_dict(self) -> dict[str, Any]:
        return to_dict(self)


@typing.overload
def get_metadata_df(
    nwb_path_or_paths: npc_io.PathLike | Iterable[npc_io.PathLike],
    disable_progress: bool = False,
    as_polars: Literal[False] = False,
) -> pd.DataFrame: ...


@typing.overload
def get_metadata_df(
    nwb_path_or_paths: npc_io.PathLike | Iterable[npc_io.PathLike],
    disable_progress: bool = False,
    as_polars: Literal[True] = True,
) -> pl.DataFrame: ...


def get_metadata_df(
    nwb_path_or_paths: npc_io.PathLike | Iterable[npc_io.PathLike],
    disable_progress: bool = False,
    as_polars: bool = False,
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
            lazynwb.tables.NWB_PATH_COLUMN_NAME: nwb._file._path.as_posix(),
        }

    future_to_path = {}
    for path in paths:
        future = lazynwb.utils.get_threadpool_executor().submit(
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
    if not as_polars:
        return pd.DataFrame.from_records(records)
    else:
        return pl.DataFrame(records)


if __name__ == "__main__":
    from npc_io import testmod

    testmod()
