#!/usr/bin/env python3
"""Step 10 — Patient-isolated 70/15/15 train/val/test split (no patient leakage)."""

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import common_parser
from hapi.config import (
    MASTER_MANIFEST_CSV,
    PATIENT_SPLITS_CSV,
    SEED,
    SPLIT_MANIFEST_CSV,
    ensure_dirs,
    require_file,
)
from hapi.io_utils import should_skip
from hapi.volumes import patient_isolated_split


def main() -> None:
    args = common_parser("Create patient-isolated dataset splits").parse_args()
    ensure_dirs()
    if should_skip(SPLIT_MANIFEST_CSV, args.force):
        return

    df = pd.read_csv(require_file(MASTER_MANIFEST_CSV, "master manifest (run 08)"))
    split_df = patient_isolated_split(df, seed=SEED)
    split_df.to_csv(SPLIT_MANIFEST_CSV, index=False)

    patients = (
        split_df.groupby("patient_id")["split"]
        .first()
        .reset_index()
        .sort_values("patient_id")
    )
    patients.to_csv(PATIENT_SPLITS_CSV, index=False)
    print(split_df["split"].value_counts().to_string())
    print(f"Saved {SPLIT_MANIFEST_CSV}")


if __name__ == "__main__":
    main()
