from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class AccountConfig:
    name: str
    account_slug: str
    institution: str
    currency: str
    input_glob: str
    date_column: str
    description_column: str
    amount_column: Optional[str]
    credit_column: Optional[str]
    debit_column: Optional[str]
    date_format: Optional[str]
    debit_is_negative: bool

    @classmethod
    def from_dict(cls, raw: dict) -> "AccountConfig":
        required = [
            "name",
            "account_slug",
            "institution",
            "currency",
            "input_glob",
            "date_column",
            "description_column",
            "debit_is_negative",
        ]
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"Missing required account config fields: {missing}")
        return cls(
            name=raw["name"],
            account_slug=raw["account_slug"],
            institution=raw["institution"],
            currency=raw["currency"],
            input_glob=raw["input_glob"],
            date_column=raw["date_column"],
            description_column=raw["description_column"],
            amount_column=raw.get("amount_column"),
            credit_column=raw.get("credit_column"),
            debit_column=raw.get("debit_column"),
            date_format=raw.get("date_format"),
            debit_is_negative=bool(raw["debit_is_negative"]),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile annual per-account statements from Wells/Amex CSV exports."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/accounts.json"),
        help="Path to account config JSON (default: config/accounts.json)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Target statement year (default: 2025)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root output directory (default: output)",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> list[AccountConfig]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{config_path}'. "
            "Create it from config/accounts.example.json."
        )
    with config_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    accounts = raw.get("accounts", [])
    if not accounts:
        raise ValueError("No accounts found in config.")
    return [AccountConfig.from_dict(item) for item in accounts]


def parse_amount(value: object) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    cleaned = (
        text.replace("$", "")
        .replace(",", "")
        .replace("(", "-")
        .replace(")", "")
        .replace("CR", "")
    )
    return float(cleaned)


def compute_net_amount(df: pd.DataFrame, cfg: AccountConfig) -> pd.Series:
    if cfg.amount_column:
        if cfg.amount_column not in df.columns:
            raise ValueError(
                f"[{cfg.account_slug}] amount_column '{cfg.amount_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        net = df[cfg.amount_column].map(parse_amount)
        # Some exports use positive values for debits; allow inversion via config.
        if not cfg.debit_is_negative:
            net = -net
        return net

    if not cfg.credit_column or not cfg.debit_column:
        raise ValueError(
            f"[{cfg.account_slug}] configure either amount_column OR both "
            "credit_column and debit_column."
        )
    if cfg.credit_column not in df.columns or cfg.debit_column not in df.columns:
        raise ValueError(
            f"[{cfg.account_slug}] credit/debit columns not found. "
            f"Available columns: {list(df.columns)}"
        )
    credit = df[cfg.credit_column].map(parse_amount)
    debit = df[cfg.debit_column].map(parse_amount)
    if cfg.debit_is_negative:
        return credit - debit
    return debit - credit


def read_account_transactions(cfg: AccountConfig) -> pd.DataFrame:
    csv_paths = sorted(Path(".").glob(cfg.input_glob))
    if not csv_paths:
        raise FileNotFoundError(
            f"[{cfg.account_slug}] no CSV files found for glob '{cfg.input_glob}'."
        )

    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if cfg.date_column not in df.columns:
            raise ValueError(
                f"[{cfg.account_slug}] date_column '{cfg.date_column}' not found in {path}. "
                f"Available columns: {list(df.columns)}"
            )
        if cfg.description_column not in df.columns:
            raise ValueError(
                f"[{cfg.account_slug}] description_column '{cfg.description_column}' "
                f"not found in {path}. Available columns: {list(df.columns)}"
            )

        parsed_dates = pd.to_datetime(
            df[cfg.date_column],
            format=cfg.date_format if cfg.date_format else None,
            errors="coerce",
        )
        net_amount = compute_net_amount(df, cfg)
        normalized = pd.DataFrame(
            {
                "account_slug": cfg.account_slug,
                "institution": cfg.institution,
                "account_name": cfg.name,
                "currency": cfg.currency,
                "date": parsed_dates,
                "description": df[cfg.description_column].astype(str).str.strip(),
                "net_amount": net_amount,
                "source_file": str(path),
            }
        )
        frames.append(normalized)

    all_txns = pd.concat(frames, ignore_index=True)
    all_txns = all_txns.dropna(subset=["date"])
    all_txns = all_txns.sort_values("date")
    return all_txns


def write_statement_files(
    txns: pd.DataFrame, cfg: AccountConfig, year: int, output_root: Path
) -> None:
    year_txns = txns[txns["date"].dt.year == year].copy()
    year_txns["month"] = year_txns["date"].dt.to_period("M").astype(str)

    out_dir = output_root / str(year) / cfg.account_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    tx_out = out_dir / f"transactions_{year}.csv"
    month_out = out_dir / f"monthly_summary_{year}.csv"
    statement_out = out_dir / f"statement_{year}.md"

    year_txns = year_txns.sort_values("date")
    year_txns["date"] = year_txns["date"].dt.strftime("%Y-%m-%d")
    year_txns.to_csv(tx_out, index=False)

    monthly = (
        year_txns.groupby("month", as_index=False)["net_amount"]
        .sum()
        .rename(columns={"net_amount": "net_total"})
    )
    monthly.to_csv(month_out, index=False)

    inflow = float(year_txns.loc[year_txns["net_amount"] > 0, "net_amount"].sum())
    outflow = float(year_txns.loc[year_txns["net_amount"] < 0, "net_amount"].sum())
    net_total = float(year_txns["net_amount"].sum())
    txn_count = int(len(year_txns))

    lines = [
        f"# Annual Statement {year}",
        "",
        f"- Account: {cfg.name}",
        f"- Account Slug: {cfg.account_slug}",
        f"- Institution: {cfg.institution}",
        f"- Currency: {cfg.currency}",
        f"- Transactions: {txn_count}",
        f"- Inflow Total: {inflow:,.2f}",
        f"- Outflow Total: {outflow:,.2f}",
        f"- Net Total: {net_total:,.2f}",
        "",
        "## Monthly Totals",
        "",
    ]
    if monthly.empty:
        lines.append("No transactions found for this year.")
    else:
        lines.append("| Month | Net Total |")
        lines.append("|---|---:|")
        for _, row in monthly.iterrows():
            lines.append(f"| {row['month']} | {row['net_total']:,.2f} |")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append(f"- `{tx_out}`")
    lines.append(f"- `{month_out}`")
    lines.append("")
    statement_out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    accounts = load_config(args.config)
    for cfg in accounts:
        txns = read_account_transactions(cfg)
        write_statement_files(txns, cfg, args.year, args.output_root)
        print(
            f"[ok] Compiled {cfg.account_slug} for {args.year} -> "
            f"{args.output_root / str(args.year) / cfg.account_slug}"
        )


if __name__ == "__main__":
    main()
