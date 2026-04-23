# HES-Sharp

Hierarchical Evolution Strategy for optimization on Sharp Ridge problems.

## Sample Files

- `main.py`
- `meta_elitist_cma.ipynb`

## How to Run

1. Prepare a virtual environment.
   - Check `pyproject.toml` for Python version and dependencies.
   - Follow the setup steps in [Build a Virtual Environment with uv](#build-a-virtual-environment-with-uv).
2. Run optimization.
   - Run `main.py`:

     ```bash
     cd hes-sharp
     uv run main.py
     ```

   - Run `meta_elitist_cma.ipynb`:
     1. Open the notebook in VS Code.
     2. Select the kernel for this project.
     3. Click Run to execute cells.
       4. If the kernel does not appear, try reloading the VS Code window.

## Build a Virtual Environment with uv

1. Install `uv`.

   ```bash
   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # or
   brew install uv

   # Windows (PowerShell)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Sync project dependencies.

   ```bash
   cd hes-sharp
   uv sync
   ```

3. Create and activate a virtual environment.

   ```bash
   uv venv .venv
   source .venv/bin/activate
   ```

4. Install and register a Jupyter kernel.

   ```bash
   uv add --dev ipykernel
   uv run python -m ipykernel install --user --name hes-sharp --display-name "Python (hes-sharp)"
   ```
