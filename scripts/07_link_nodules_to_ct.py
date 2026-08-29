#!/usr/bin/env python3
"""Step 07 — Join subset nodules to TCIA CT series; keep unambiguous 1:1 mappings."""

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import common_parser
from hapi.config import (
    NODULE_CT_CANDIDATES_CSV,
    NODULE_SERIES_CANDIDATES_CSV,
    NODULE_SINGLE_CT_CSV,
    NODULE_TO_LIDC_CSV,
    TCIA_SERIES_CSV,
    ensure_dirs,
    require_file,
)
from hapi.io_utils import should_skip


def main() -> None:
    args = common_parser("Link nodules to candidate CT series").parse_args()
    ensure_dirs()
    if should_skip(NODULE_SINGLE_CT_CSV, args.force):
        return

    mapping = pd.read_csv(require_file(NODULE_TO_LIDC_CSV, "nodule mapping (run 02)"))
    tcia = pd.read_csv(require_file(TCIA_SERIES_CSV, "TCIA series CSV (run 06)"))

    nodule_col = "subset_nodule_id" if "subset_nodule_id" in mapping.columns else "nodule_id"
    mapping["patient_id"] = mapping["patient_id"].astype(str).str.strip()
    tcia["PatientID"] = tcia["PatientID"].astype(str).str.strip()

    subset_patients = set(mapping["patient_id"])
    relevant = tcia[tcia["PatientID"].isin(subset_patients)].copy()
    joined = mapping.merge(relevant, left_on="patient_id", right_on="PatientID", how="left")
    joined.to_csv(NODULE_SERIES_CANDIDATES_CSV, index=False)

    ct = joined[joined["Modality"].astype(str).str.upper() == "CT"].copy()
    ct.to_csv(NODULE_CT_CANDIDATES_CSV, index=False)

    series_counts = ct.groupby(nodule_col)["SeriesInstanceUID"].nunique()
    unambiguous = series_counts[series_counts == 1].index
    ambiguous = series_counts[series_counts > 1].index.tolist()
    print(f"Unambiguous nodules: {len(unambiguous)}")
    print(f"Ambiguous nodules: {ambiguous}")

    clean = ct[ct[nodule_col].isin(unambiguous)].copy()
    clean.to_csv(NODULE_SINGLE_CT_CSV, index=False)
    print(f"Saved {NODULE_SINGLE_CT_CSV} ({len(clean)} rows)")


if __name__ == "__main__":
    main()
