"""Hash-match Kaggle 2,000-image subset nodules to native LIDC PNG paths."""

from __future__ import annotations

import hashlib
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def index_original_images(original_zip: Path) -> dict[str, list[str]]:
    original_hash_index: dict[str, list[str]] = defaultdict(list)
    with zipfile.ZipFile(original_zip, "r") as z:
        original_png_names = [
            name
            for name in z.namelist()
            if name.lower().endswith(".png") and "/images/" in name
        ]
        print(f"Original image files found: {len(original_png_names):,}")
        for i, name in enumerate(original_png_names, start=1):
            original_hash_index[sha256_bytes(z.read(name))].append(name)
            if i % 5000 == 0:
                print(f"Indexed {i:,} / {len(original_png_names):,}")
    return original_hash_index


def match_subset_to_lidc(
    subset_zip: Path,
    original_zip: Path,
) -> pd.DataFrame:
    print("Indexing original LIDC image files...")
    original_hash_index = index_original_images(original_zip)

    unmatched_images: list[tuple[str, str]] = []
    ambiguous_images: list[tuple[str, str, list[str]]] = []
    mapping_rows = []

    with zipfile.ZipFile(subset_zip, "r") as z:
        subset_png_names = [
            name for name in z.namelist() if name.lower().endswith(".png")
        ]
        print(f"Subset PNG files found: {len(subset_png_names):,}")

        subset_groups: dict[str, list[str]] = defaultdict(list)
        for name in subset_png_names:
            parts = name.replace("\\", "/").split("/")
            if len(parts) >= 2:
                subset_groups[parts[0]].append(name)

        print(f"Subset nodule folders found: {len(subset_groups)}")

        for subset_nodule in sorted(subset_groups.keys()):
            matched_native_paths = []
            for subset_path in subset_groups[subset_nodule]:
                subset_hash = sha256_bytes(z.read(subset_path))
                candidates = original_hash_index.get(subset_hash, [])
                if len(candidates) == 1:
                    matched_native_paths.append(candidates[0])
                elif len(candidates) == 0:
                    unmatched_images.append((subset_nodule, subset_path))
                else:
                    ambiguous_images.append(
                        (subset_nodule, subset_path, candidates)
                    )

            native_folder_counts: dict[tuple[str, str], int] = defaultdict(int)
            for native_path in matched_native_paths:
                parts = native_path.replace("\\", "/").split("/")
                try:
                    images_idx = parts.index("images")
                except ValueError:
                    continue
                if images_idx < 2:
                    continue
                patient_id = parts[images_idx - 2]
                native_nodule_id = parts[images_idx - 1]
                native_folder_counts[(patient_id, native_nodule_id)] += 1

            ranked = sorted(
                native_folder_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            total_subset_slices = len(subset_groups[subset_nodule])
            total_matched = len(matched_native_paths)

            if not ranked:
                mapping_status = "UNMATCHED"
                patient_id = None
                native_nodule_id = None
                best_match_count = 0
                second_match_count = 0
            else:
                (patient_id, native_nodule_id), best_match_count = ranked[0]
                second_match_count = ranked[1][1] if len(ranked) > 1 else 0
                if total_matched == total_subset_slices and len(ranked) == 1:
                    mapping_status = "EXACT_UNIQUE_MATCH"
                elif (
                    total_matched == total_subset_slices
                    and best_match_count > second_match_count
                ):
                    mapping_status = "MATCHED_BUT_REVIEW"
                else:
                    mapping_status = "PARTIAL_MATCH"

            mapping_rows.append(
                {
                    "subset_nodule_id": subset_nodule,
                    "subset_slice_count": total_subset_slices,
                    "matched_slice_count": total_matched,
                    "patient_id": patient_id,
                    "native_nodule_id": native_nodule_id,
                    "best_match_count": best_match_count,
                    "second_best_match_count": second_match_count,
                    "status": mapping_status,
                }
            )

    mapping_df = pd.DataFrame(mapping_rows)
    print(mapping_df["status"].value_counts().to_string())
    print(f"Unmatched individual images: {len(unmatched_images)}")
    print(f"Ambiguous individual images: {len(ambiguous_images)}")
    return mapping_df
