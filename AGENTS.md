# PancakeDrip

## Cursor Cloud specific instructions

### Project overview
PancakeDrip is a Python-based repository. Feature branches show the project evolving toward a financial statement compilation tool using `pandas`. The `main` branch is minimal (README only); active development happens on feature branches.

### Python environment
- Python 3.12 with a virtual environment at `.venv/`.
- Activate with: `source .venv/bin/activate`
- Core dependency: `pandas` (install via `pip install -r requirements.txt` when the file exists on the current branch, otherwise `pip install pandas`).

### Running the application
- The main CLI script (when present on feature branches) is `src/compile_statements.py`.
- Run with: `python src/compile_statements.py --config config/accounts.json --year 2025`
- Requires `config/accounts.json` (copy from `config/accounts.example.json`).

### Development notes
- `python3.12-venv` must be installed at the system level (`sudo apt-get install python3.12-venv`) for venv creation to succeed on fresh VMs.
- No lint/test/build tooling is configured on `main` yet. Check feature branches for any additions.
- Raw bank CSV data and `config/accounts.json` are gitignored for security.
