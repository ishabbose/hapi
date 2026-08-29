#!/usr/bin/env python3
"""Step 11 — Extract native images + 4 reader masks for the 325-nodule cohort."""

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import common_parser
from hapi.config import (
    ORIGINAL_ZIP,
    PROCESSED_DIR,
    PROCESSED_MANIFEST_CSV,
    SPLIT_MANIFEST_CSV,
    ensure_dirs,
    require_file,
)
from hapi.extract import extract_processed_cohort
from hapi.io_utils import should_skip


def main() -> None:
    args = common_parser("Extract processed 325-nodule dataset from original ZIP").parse_args()
    ensure_dirs()
    if should_skip(PROCESSED_MANIFEST_CSV, args.force):
        return

    master = pd.read_csv(require_file(SPLIT_MANIFEST_CSV, "split manifest (run 10)"))
    require_file(ORIGINAL_ZIP, "original LIDC ZIP")
    processed = extract_processed_cohort(ORIGINAL_ZIP, master, PROCESSED_DIR)
    processed.to_csv(PROCESSED_MANIFEST_CSV, index=False)
    print(f"Saved {PROCESSED_MANIFEST_CSV} ({len(processed)} nodules)")


if __name__ == "__main__":
    main()
