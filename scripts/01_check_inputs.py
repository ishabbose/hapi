#!/usr/bin/env python3
"""Step 01 — Check that expected LIDC / subset files are present."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import common_parser

from hapi.config import (
    ORIGINAL_ZIP,
    SUBSET_ZIP,
    TCIA_MANIFEST,
    ensure_dirs,
)


def main() -> None:
    common_parser("Check local dataset files").parse_args()
    ensure_dirs()
    print("TCIA manifest:", TCIA_MANIFEST, "exists=", TCIA_MANIFEST.exists())
    print("Subset ZIP:   ", SUBSET_ZIP, "exists=", SUBSET_ZIP.exists())
    print("Original ZIP: ", ORIGINAL_ZIP, "exists=", ORIGINAL_ZIP.exists())
    if not TCIA_MANIFEST.exists():
        raise SystemExit(
            "Place TCIA_LIDC-IDRI_20200921.tcia under content/ (already in repo)."
        )
    if not SUBSET_ZIP.exists() or not ORIGINAL_ZIP.exists():
        print(
            "\nCopy the Colab ZIPs into content/ (or set HAPI_SUBSET_ZIP / "
            "HAPI_ORIGINAL_ZIP) before running matching and extraction steps."
        )


if __name__ == "__main__":
    main()
