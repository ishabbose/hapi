#!/usr/bin/env python3
"""Step 04 — Parse all LIDC XML files into an inventory CSV."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import common_parser
from hapi.config import LIDC_XML_DIR, XML_INVENTORY_CSV, ensure_dirs, require_file
from hapi.io_utils import should_skip
from hapi.xml_annotations import build_xml_inventory


def main() -> None:
    args = common_parser("Build LIDC XML annotation inventory").parse_args()
    ensure_dirs()
    if should_skip(XML_INVENTORY_CSV, args.force):
        return
    require_file(LIDC_XML_DIR, "LIDC XML directory (run 03 first)")
    inventory = build_xml_inventory(LIDC_XML_DIR)
    inventory.to_csv(XML_INVENTORY_CSV, index=False)
    print(f"Saved {XML_INVENTORY_CSV} ({len(inventory)} files)")


if __name__ == "__main__":
    main()
