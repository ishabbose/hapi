#!/usr/bin/env python3
"""Step 08 — Drop the two unresolved multi-series nodules (029, 085) → 325 cohort."""

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import common_parser
from hapi.config import (
    ELIGIBLE_COHORT_CSV,
    EXCLUDED_CSV,
    EXCLUDED_NODULES,
    MASTER_MANIFEST_CSV,
    NODULE_SINGLE_CT_CSV,
    NODULE_TO_LIDC_CSV,
    ensure_dirs,
    require_file,
)
from hapi.io_utils import should_skip


def main() -> None:
    args = common_parser("Filter to the 325-nodule eligible cohort").parse_args()
    ensure_dirs()
    if should_skip(MASTER_MANIFEST_CSV, args.force):
        return

    mapping = pd.read_csv(require_file(NODULE_TO_LIDC_CSV, "nodule mapping (run 02)"))
    series = pd.read_csv(require_file(NODULE_SINGLE_CT_CSV, "single-CT mapping (run 07)"))

    excluded = mapping[mapping["subset_nodule_id"].astype(str).isin(EXCLUDED_NODULES)].copy()
    eligible = mapping[~mapping["subset_nodule_id"].astype(str).isin(EXCLUDED_NODULES)].copy()
    excluded.to_csv(EXCLUDED_CSV, index=False)
    eligible.to_csv(ELIGIBLE_COHORT_CSV, index=False)

    master = eligible.merge(
        series,
        on=["subset_nodule_id", "patient_id", "native_nodule_id"],
        how="inner",
        suffixes=("", "_series"),
    )
    master.to_csv(MASTER_MANIFEST_CSV, index=False)
    print(f"Eligible mapping rows: {len(eligible)}")
    print(f"Master (with series): {len(master)}")
    print(f"Saved {MASTER_MANIFEST_CSV}")


if __name__ == "__main__":
    main()
