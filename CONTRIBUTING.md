# Contributing

Contributions are welcome, and they are greatly appreciated!
Every little bit helps, and credit will always be given.

## Environment setup

1. Fork and clone the repository:
```bash
git clone https://github.com/bjhardcastle/lazynwb
cd lazynwb
```

2. Install with `uv` 
```bash
uv python pin 3.11
uv sync
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

You now have an editable pip install of the project, with all dev dependencies.
The following should work:
```bash
python -c "import lazynwb; print(lazynwb.__version__)"
```

### Using PDM

The project uses [uv](https://docs.astral.sh/uv/) for reproducible dev environments, with
configuration for tools in `pyproject.toml`
While working on the project, use `uv` to manage dependencies:
- add dependencies: `uv add numpy pandas`
  - add dev dependencies: `uv add --dev mypy`
- remove dependencies correctly: `uv remove numpy`   # does nothing because pandas still needs numpy!
- update the environment to reflect changes in `pyproject.toml`: `uv sync`
Always commit & push `uv.lock` to share the up-to-date dev environment


## Development (internal contributors)

1. Edit the code and/or the documentation on the main branch

2. Add simple doctests to functions or more elaborate tests to modules in `tests`

3. If you updated the project's dependencies (or you pulled changes):
  - run `uv sync`
  - if it fails due to dependencies you added, follow any error messages to resolve dependency version conflicts
  - when it doesn't fail, commit any changes to `uv.lock` along with the changes to `pyproject.toml`

4. Run tests with `uv run task test`
  - mypy will check all functions that contain type annotations in their signature
  - pytest will run doctests and any tests in the `tests` dir
 
- if you are unsure about how to fix a test, just push your changes - the continuous integration will fail on Github and someone else can have a look

- don't update the changelog, it will be taken care of automatically

- link to any related issue number in the Commit message: `Fix variable name #13` 

- pull changes with `git pull --rebase` to keep the commit history easy to read

## Updating from the original template
With a clean working directory, run `pipx run copier update --defaults`.

See [here](https://github.com/bjhardcastle/copier-pdm-npc/blob/main/README.md)
for more info.