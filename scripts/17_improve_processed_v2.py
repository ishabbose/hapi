#!/usr/bin/env python3
"""Step 17 — Improved processed 325 dataset (v2).

Does not modify data/processed_325_nodules. Writes a new version with:
  - intensity window fitted on the train split only, applied to all splits
  - union-mask in-plane crop, then center-pad back to original H×W (depth unchanged)
  - mask_consensus.npy and mask_union.npy
  - bit-for-bit copies of original PNGs and reader masks
  - original metadata plus preprocess parameters
  - QC flags on every manifest row
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import common_parser
from hapi.config import (
    EXPECTED_ELIGIBLE_NODULES,
    IMPROVED_MANIFEST_CSV,
    IMPROVED_PARAMS_JSON,
    IMPROVED_PROCESSED_DIR,
    IMPROVED_QC_CSV,
    PROCESSED_DIR,
    PROCESSED_MANIFEST_CSV,
    ensure_dirs,
    require_file,
)
from hapi.improve_processed import (
    CONSENSUS_THRESHOLD,
    CROP_MARGIN_PX,
    INTENSITY_HIGH_P,
    INTENSITY_LOW_P,
    build_improved_processed_v2,
)
from hapi.io_utils import should_skip


def main() -> None:
    parser = common_parser("Build processed 325 nodules v2")
    parser.add_argument("--low-percentile", type=float, default=INTENSITY_LOW_P)
    parser.add_argument("--high-percentile", type=float, default=INTENSITY_HIGH_P)
    parser.add_argument("--crop-margin", type=int, default=CROP_MARGIN_PX)
    parser.add_argument("--consensus-threshold", type=float, default=CONSENSUS_THRESHOLD)
    args = parser.parse_args()
    ensure_dirs()
    if should_skip(IMPROVED_MANIFEST_CSV, args.force):
        return

    df = pd.read_csv(
        require_file(PROCESSED_MANIFEST_CSV, "processed manifest (run 11)")
    )
    require_file(PROCESSED_DIR, "processed 325 nodules (run 11)")
    if len(df) != EXPECTED_ELIGIBLE_NODULES:
        raise SystemExit(f"Processed manifest has {len(df)} rows, expected {EXPECTED_ELIGIBLE_NODULES}.")

    result, params = build_improved_processed_v2(
        df,
        IMPROVED_PROCESSED_DIR,
        PROCESSED_DIR,
        low_p=args.low_percentile,
        high_p=args.high_percentile,
        crop_margin_px=args.crop_margin,
        consensus_threshold=args.consensus_threshold,
    )
    result.to_csv(IMPROVED_MANIFEST_CSV, index=False)
    qc_cols = [c for c in result.columns if c.startswith("qc_")]
    result[["subset_nodule_id", "patient_id", "split", *qc_cols]].to_csv(
        IMPROVED_QC_CSV, index=False
    )
    IMPROVED_PARAMS_JSON.write_text(json.dumps(params, indent=2) + "\n")

    print(f"Wrote {IMPROVED_PROCESSED_DIR} ({len(result)} nodules)")
    print(f"QC pass: {int(result['qc_pass'].sum())}/{len(result)}")
    print(result["split"].value_counts().to_string())
    print(f"Saved {IMPROVED_MANIFEST_CSV}")
    print(f"Saved {IMPROVED_QC_CSV}")
    print(f"Saved {IMPROVED_PARAMS_JSON}")


if __name__ == "__main__":
    main()
