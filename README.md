# Financial Statement Compiler (Wells Fargo + Amex)

This repo now includes a starter tool to compile one **12-month (2025) statement per account** from exported CSV data.

## What this does

- Reads transaction CSV exports from Wells Fargo and Amex.
- Normalizes date, description, and amount across formats.
- Filters records to a target year (default: `2025`).
- Generates per-account outputs:
  - `transactions_YYYY.csv` (normalized transaction ledger)
  - `monthly_summary_YYYY.csv` (month-by-month totals)
  - `statement_YYYY.md` (human-readable annual statement)

## Project layout

```text
config/
  accounts.example.json      # template you copy and edit
data/
  raw/
    wells_fargo/             # put Wells CSV exports here
    amex/                    # put Amex CSV exports here
output/
  2025/
    <account_slug>/          # generated outputs
src/
  compile_statements.py      # main CLI
```

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies.
3. Copy `config/accounts.example.json` to `config/accounts.json`.
4. Edit `config/accounts.json` so each account points at your CSV files and column names.
5. Run the compiler.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp config/accounts.example.json config/accounts.json
# edit config/accounts.json

python src/compile_statements.py --config config/accounts.json --year 2025
```

## CSV mapping notes

Each account mapping tells the compiler:

- where files live (`input_glob`)
- which columns to use (`date_column`, `description_column`, `amount_column`)
- how to interpret signs (`debit_is_negative`)

If your CSV has credits and debits in separate columns, set:

- `amount_column` to `null`
- `credit_column` and `debit_column` to the proper headers

The script computes `net_amount = credits - debits` (or the inverse if `debit_is_negative` is `true`).

## Cursor workflow (recommended)

Use these prompts in Cursor chat:

1. **Scaffold and configure**
   - "Read config/accounts.json and help me map my Wells Fargo and Amex CSV headers correctly."
2. **Validate input data**
   - "Run the statement compiler for 2025 and show me any schema/mapping issues."
3. **Finalize outputs**
   - "Generate annual statements for each configured account and summarize totals."

## Security / compliance guidance

- Do **not** store bank passwords in this repo.
- Prefer CSV exports or aggregator APIs (Plaid/MX) with secure token storage.
- Keep raw files out of git (see `.gitignore`).

## Optional next step (full auto pull)

For fully automated pulling (instead of manual CSV export), add an ingestion layer via an API aggregator (for example, Plaid) that writes normalized CSV files into `data/raw/...` before running the same compiler.
