"""Extract native LIDC crops + reader masks from the original ZIP."""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def slice_number(path: str) -> int:
    match = re.search(r"(\d+)\.png$", path.replace("\\", "/"))
    return int(match.group(1)) if match else 10**9


def extract_processed_cohort(
    zip_path: Path,
    master_df: pd.DataFrame,
    output_root: Path,
) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)
    processed_rows = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_names = set(zf.namelist())
        print(f"Original ZIP entries: {len(zip_names)}")

        for _, row in master_df.iterrows():
            subset_id = str(row["subset_nodule_id"])
            patient_id = str(row["patient_id"])
            native_id = str(row["native_nodule_id"])
            series_uid = str(row.get("SeriesInstanceUID", row.get("series_instance_uid", "")))

            case_dir = output_root / subset_id
            images_dir = case_dir / "images"
            mask_dirs = {m: case_dir / f"mask-{m}" for m in range(4)}
            images_dir.mkdir(parents=True, exist_ok=True)
            for d in mask_dirs.values():
                d.mkdir(parents=True, exist_ok=True)

            source_prefix = f"LIDC-IDRI-slices/{patient_id}/{native_id}/"
            source_image_prefix = source_prefix + "images/"
            image_sources = sorted(
                [
                    x
                    for x in zip_names
                    if x.startswith(source_image_prefix) and x.lower().endswith(".png")
                ],
                key=slice_number,
            )

            case_failed = False
            image_dimensions = set()
            for src in image_sources:
                filename = Path(src).name
                dst = images_dir / filename
                raw = zf.read(src)
                with Image.open(io.BytesIO(raw)) as img:
                    image_dimensions.add(img.size)
                dst.write_bytes(raw)

            if not image_sources:
                case_failed = True

            mask_file_counts = {}
            mask_npy_paths = {}
            for m in range(4):
                prefix = source_prefix + f"mask-{m}/"
                sources = sorted(
                    [
                        x
                        for x in zip_names
                        if x.startswith(prefix) and x.lower().endswith(".png")
                    ],
                    key=slice_number,
                )
                arrays = []
                for src in sources:
                    raw = zf.read(src)
                    dst = mask_dirs[m] / Path(src).name
                    dst.write_bytes(raw)
                    with Image.open(io.BytesIO(raw)) as mask_img:
                        arrays.append(np.array(mask_img))
                mask_file_counts[m] = len(sources)
                if arrays:
                    stacked = np.stack(arrays, axis=0)
                    npy_path = case_dir / f"mask_{m}.npy"
                    np.save(npy_path, stacked)
                    mask_npy_paths[m] = str(npy_path)
                else:
                    mask_npy_paths[m] = ""

            image_files = sorted(str(p) for p in images_dir.glob("*.png"))
            metadata = {
                "subset_nodule_id": subset_id,
                "patient_id": patient_id,
                "native_nodule_id": native_id,
                "SeriesInstanceUID": series_uid,
                "image_slice_count": len(image_files),
                "processing_status": "FAIL" if case_failed else "PASS",
            }
            (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

            processed_rows.append(
                {
                    **{k: row[k] for k in row.index},
                    "images_dir": str(images_dir),
                    "image_files": "|".join(image_files),
                    "mask_0_path": mask_npy_paths[0],
                    "mask_1_path": mask_npy_paths[1],
                    "mask_2_path": mask_npy_paths[2],
                    "mask_3_path": mask_npy_paths[3],
                    "image_slice_count": len(image_files),
                    "processing_status": metadata["processing_status"],
                }
            )

    return pd.DataFrame(processed_rows)
