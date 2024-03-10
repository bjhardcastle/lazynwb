# Contributing

Contributions are welcome, and they are greatly appreciated!
Every little bit helps, and credit will always be given.

## Environment setup

1. Fork and clone the repository:
```bash
git clone https://github.com/AllenInstitute/lazynwb
cd lazynwb
```

2. Create a new virtual environment:
- conda is convenient for getting a specific Python version, but adds additional packages
- it's preferable to use a completely clean environment to properly test the project's specified dependencies in isolation
- where possible, use the lowest supported Python version (specified in `pyproject.toml` `project/requires-python`) 
```bash
python3 -m venv .venv
```

3. Activate the environment:
- Windows
  ```bash
  .venv\scripts\activate
  ```

- Unix
  ```bash
  source .venv/bin/scripts/activate
  ```

4. Add [PDM](https://pdm.fming.dev) to manage the project's dependencies and run pre-build jobs:
```bash
pip install pdm
pdm install
```

You now have an editable pip install of the project, with all dev dependencies.
The following should work:
```bash
python -c "import lazynwb; print(lazynwb.__version__)"
```

### Using PDM

The project uses [PDM](https://pdm.fming.dev) for reproducible dev environments, with pre-defined `pyproject.toml` configuration for tools
While working on the project, use PDM to manage dependencies:
- add dependencies: `pdm add numpy pandas`
  - add dev dependencies: `pdm add -G dev mypy`
- remove dependencies correctly: `pdm remove numpy`   # does nothing because pandas still needs numpy!
- update the environment to reflect changes in `pyproject.toml`: `pdm update`
Always commit & push `pdm.lock` to share the up-to-date dev environment


## Development (internal contributors)

1. Edit the code and/or the documentation on the main branch

2. Add simple doctests to functions or more elaborate tests to modules in `tests`

3. If you updated the project's dependencies (or you pulled changes):
  - run `pdm update`
  - if it fails due to dependencies you added, follow any error messages to resolve dependency version conflicts
  - when it doesn't fail, commit any changes to `pdm.lock` along with the changes to `pyproject.toml`

4. Run tests with `pdm run test`
  - mypy will check all functions that contain type annotations in their signature
  - pytest will run doctests and any tests in the `tests` dir

5. If you updated the documentation or the project dependencies:
  - run `pdm run doc`
  - go to http://localhost:8000 and check that everything looks good
 
- if you are unsure about how to fix a test, just push your changes - the continuous integration will fail on Github and someone else can have a look

- don't update the changelog, it will be taken care of automatically

- link to any related issue number in the Commit message: `Fix variable name #13` 

- pull changes with `git pull --rebase` to keep the commit history easy to read

## Updating from the original template
With a clean working directory, run `pipx run copier update --defaults`.

See [here](https://github.com/AllenInstitute/copier-pdm-npc/blob/main/README.md)
for more info.