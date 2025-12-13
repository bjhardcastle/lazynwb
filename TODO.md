- make this func to normalize inputs:
  def get_nwb_sources(source: str | PathLike | Iterable[str | PathLike], glob_pattern: str | None = None) -> tuple[str]:
  behavior:
    - if source corresponds to a single file, use directly
    - if source is an iterable of paths, use them directly
    - if source glob_pattern exists apply recursively to find files in the single source or in every source if an iterable (we assume each source is a dir, but don't check defensively)
    - if the soure is a dir and glob_pattern is None we raise an error

  we need to handle checking for dir vs files efficiently, for as few calls to cloud storage as possible. it would be nice to use the presence of glob_pattern to signal file(s) from dir(s), but it's also convenient to have a default glob pattern ('*.nwb') NEED TO DECIDE HERE 


- make a 'project' class that connects to all nwbs in a dir, with concatenated tables available (should not necessarily contain individual session instances, but could make them available on-demand)

- restore and update DANDIset access and convenience functions 

- make nicer readme docs  