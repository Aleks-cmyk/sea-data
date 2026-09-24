## Project Domain

This is a project for generating synthetic data for training AI models to detect objects and a horizon in a maritime environment. Simulation is being done using blender and the python API. 

## Project Setup and Management

- Python version: 3.14. Don't use newer syntax.
- Dependency management: `uv` and `pyproject.toml`. Never use `pip`
  or a `requirements.txt` file.
- Add a dependency with `uv add <package>`. Never use `uv pip`
  for dependencies.
- Branch from `main` as `feature/<name>` and use Conventional Commits.
- Stage changes for review. Don't commit to `main` or push without
  being asked.

## Coding Conventions
- Type-hint public functions and methods, including their return types.
- Use `pathlib` for path management. Don't use `os.path`.
- Prefer f-strings over `str.format()` or `%` formatting.
- Follow EAFP: handle exceptions rather than checking conditions up front.
- Write Google-style docstrings for every public function and method.
- Embrace idiomatic Python like comprehensions, generators, and decorators.

## Project Structure

- The project uses a `src/`-layout.
- Put tests in `tests/`
- Don't create new packages or new files without being asked.

## Quality Gates

A task is done only when all of these pass:

- `uv run ruff format` leaves the code unchanged.
- `uv run ruff check` reports no errors.
- `uv run mypy main.py` reports no errors.
- `uv run pytest` passes, with a test added for every new endpoint.

## Constraints

- Ask before adding any external dependency.
- Declare a task done only after the gates pass and docstrings are
  updated.

## Ignore

Treat everything in `.gitignore` as off-limits to read or edit. On top of
that, never open:

- Secrets and `.env` files
- Large data files unrelated to the current task
- Vendored or generated code
