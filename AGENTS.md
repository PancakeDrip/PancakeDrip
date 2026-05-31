# PancakeDrip

## Cursor Cloud specific instructions

### Repository layout
- `main` contains only a stub README. The runnable **Financial Statement Compiler** lives on `cursor/financial-statement-compilation-c4f7` (Python CLI, `pandas`, `src/compile_statements.py`).
- Check out that branch (or a branch that includes `requirements.txt` and `src/`) before running the app.

### Python environment
- **Python 3.12** with a project venv at `.venv/`.
- On fresh Ubuntu VMs, install the venv module once if `python3 -m venv` fails: `sudo apt-get install -y python3.12-venv`.
- Activate: `source .venv/bin/activate`
- Dependencies: `pip install -r requirements.txt` (currently `pandas`).

### Running the compiler (hello-world / E2E)
There is no long-running server. E2E is a one-shot CLI run:

1. `cp config/accounts.example.json config/accounts.json`
2. Place bank CSV exports under paths in `accounts.json` (e.g. `data/raw/wells_fargo/*.csv`, `data/raw/amex/*.csv`). These paths are gitignored.
3. `python src/compile_statements.py --config config/accounts.json --year 2025`
4. Outputs appear under `output/<year>/<account_slug>/` (`transactions_*.csv`, `monthly_summary_*.csv`, `statement_*.md`).

### Lint / test / build
- No ESLint, pytest, or Makefile targets are configured on the compiler branch.
- Quick sanity check: `python -m py_compile src/compile_statements.py` with the venv activated.

### Security
- Do not commit real `config/accounts.json` or raw bank CSVs (see `.gitignore`).
