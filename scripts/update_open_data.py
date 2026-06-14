"""Download open international football data and prepare model-ready CSVs.

This script does not overwrite the original root CSVs. It writes raw downloads
under data/raw/international_results and model-ready files under data/processed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import ssl
from urllib.request import urlopen

import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw", "international_results")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MIN_YEAR = 2006
MAX_DATE = pd.Timestamp.today().normalize()

RAW_URLS = {
    "results.csv": "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
    "goalscorers.csv": "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv",
    "shootouts.csv": "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv",
    "former_names.csv": "https://raw.githubusercontent.com/martj42/international_results/master/former_names.csv",
}


def _download(force: bool = False) -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    context = _ssl_context()
    for filename, url in RAW_URLS.items():
        target = os.path.join(RAW_DIR, filename)
        if os.path.exists(target) and not force:
            continue
        print(f"Downloading {filename}")
        with urlopen(url, context=context) as response, open(target, "wb") as output:
            shutil.copyfileobj(response, output)


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl._create_unverified_context()


def _filter_by_year(path: str, output_name: str, date_col: str = "date", min_year: int | None = MIN_YEAR) -> dict:
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    original_rows = len(df)
    mask = df[date_col] <= MAX_DATE
    if min_year is not None:
        mask &= df[date_col].dt.year >= min_year
    df = df[mask].copy()
    df = df.sort_values(date_col).reset_index(drop=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, output_name)
    df.to_csv(output_path, index=False)
    return {
        "file": output_path,
        "original_rows": original_rows,
        "filtered_rows": len(df),
        "min_date": str(df[date_col].min().date()) if not df.empty else None,
        "max_date": str(df[date_col].max().date()) if not df.empty else None,
    }


def prepare_processed() -> list[dict]:
    return [
        _filter_by_year(os.path.join(RAW_DIR, "results.csv"), "matches_2006.csv"),
        _filter_by_year(os.path.join(RAW_DIR, "goalscorers.csv"), "goalscorers_2006.csv"),
        _filter_by_year(os.path.join(RAW_DIR, "shootouts.csv"), "shootouts_2006.csv"),
        _filter_by_year(os.path.join(RAW_DIR, "results.csv"), "matches_all.csv", min_year=None),
        _filter_by_year(os.path.join(RAW_DIR, "goalscorers.csv"), "goalscorers_all.csv", min_year=None),
        _filter_by_year(os.path.join(RAW_DIR, "shootouts.csv"), "shootouts_all.csv", min_year=None),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download files even if they already exist.")
    args = parser.parse_args()

    _download(force=args.force)
    summaries = prepare_processed()
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
