#!/usr/bin/env python3
"""Run pipeline steps 01–N in order. Existing artifacts are skipped unless --force."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import common_parser

SCRIPTS_DIR = Path(__file__).resolve().parent

STEPS = [
    "01_check_inputs.py",
    "02_match_subset_to_lidc.py",
    "03_download_lidc_xml.py",
    "04_build_xml_inventory.py",
    "05_parse_tcia_manifest.py",
    "06_map_tcia_series.py",
    "07_link_nodules_to_ct.py",
    "08_filter_cohort.py",
    "09_match_xml_to_series.py",
    "10_patient_splits.py",
    "11_extract_processed.py",
    "12_normalize_volumes.py",
    "13_train_3d_autoencoder.py",
    "14_train_2d_unet.py",
]


def main() -> None:
    parser = common_parser("Run numbered pipeline scripts in order")
    parser.add_argument("--from-step", type=int, default=1, help="First step number (1–14)")
    parser.add_argument("--to-step", type=int, default=14, help="Last step number (1–14)")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Stop after step 12 (no autoencoder / U-Net training).",
    )
    args = parser.parse_args()

    start = max(1, args.from_step)
    end = min(14, args.to_step)
    if args.skip_train:
        end = min(end, 12)

    extra = ["--force"] if args.force else []
    for i in range(start, end + 1):
        script = SCRIPTS_DIR / STEPS[i - 1]
        print("\n" + "=" * 72)
        print(f"RUNNING STEP {i:02d}: {script.name}")
        print("=" * 72)
        cmd = [sys.executable, str(script), *extra]
        result = subprocess.run(cmd, cwd=str(SCRIPTS_DIR.parent))
        if result.returncode != 0:
            raise SystemExit(f"Step {i:02d} failed with exit code {result.returncode}")


if __name__ == "__main__":
    main()
