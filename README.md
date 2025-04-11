# lazynwb



[![PyPI](https://img.shields.io/pypi/v/lazynwb.svg?label=PyPI&color=blue)](https://pypi.org/project/lazynwb/)
[![Python version](https://img.shields.io/pypi/pyversions/lazynwb)](https://pypi.org/project/lazynwb/)

[![Coverage](https://img.shields.io/codecov/c/github/AllenInstitute/lazynwb?logo=codecov)](https://app.codecov.io/github/AllenInstitute/lazynwb)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/AllenInstitute/lazynwb/publish.yml?label=CI/CD&logo=github)](https://github.com/bjhardcastle/lazynwb/actions/workflows/publish.yml)
[![GitHub issues](https://img.shields.io/github/issues/AllenInstitute/lazynwb?logo=github)](https://github.com/bjhardcastle/lazynwb/issues)

# Usage
```bash
uv add lazynwb
```

## Python
```python
>>> import lazynwb
```

# Development
See instructions in https://github.com/bjhardcastle/lazynwb/CONTRIBUTING.md and the original template: https://github.com/bjhardcastle/copier-pdm-npc/blob/main/README.md

## notes

- hdf5 access seems to have a mutex lock that threads spend a long time waiting to
  acquire (with remfile)
- seems to slow down over time in single-threaded loop
    - on laptop, first 5 are fast (2-3 s per iteration) - successive iterations
      are much slower (>60 s)