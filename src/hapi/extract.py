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

from hapi.config import SUBSET_EXTRACT_DIR, SUBSET_ZIP


def ensure_kaggle_extracted(
    zip_path: Path | None = None,
    extract_dir: Path | None = None,
    refresh_from_zip: bool = False,
) -> Path:
    import shutil

    zip_path = zip_path or SUBSET_ZIP
    extract_dir = extract_dir or SUBSET_EXTRACT_DIR
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing Kaggle subset ZIP: {zip_path}")
    if (
        not refresh_from_zip
        and extract_dir.exists()
        and any(extract_dir.rglob("*.png"))
    ):
        return extract_dir
    if refresh_from_zip and extract_dir.exists():
        shutil.rmtree(extract_dir)
    print(f"Extracting {zip_path} to {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    return extract_dir


def v2_nodule_ids(v2_root: Path) -> set[str]:
    return {p.parent.name.lower() for p in v2_root.glob("*/image_volume.npy")}


def filter_kaggle_zip_to_ids(
    source_zip: Path,
    dest_zip: Path,
    keep_ids: set[str],
) -> pd.DataFrame:
    """Copy unprocessed PNG (and sidecar) members for keep_ids into dest_zip."""
    from hapi.volumes import volume_id_for_path

    keep_ids = {str(i).lower() for i in keep_ids}
    if not source_zip.exists():
        raise FileNotFoundError(f"Missing Kaggle ZIP: {source_zip}")

    import tempfile
    import os

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".zip", dir=dest_zip.parent)
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    kept_files = 0
    kept_ids: set[str] = set()
    dropped_ids: set[str] = set()
    try:
        with zipfile.ZipFile(source_zip, "r") as zin, zipfile.ZipFile(
            tmp_path, "w"
        ) as zout:
            for item in zin.infolist():
                name = item.filename.replace("\\", "/")
                if name.startswith("__MACOSX") or name.startswith("."):
                    continue
                vol_id = volume_id_for_path(Path(name))
                if not re.match(r"^nodule_\d+$", vol_id, re.IGNORECASE):
                    continue
                vol_id = vol_id.lower()
                if vol_id not in keep_ids:
                    dropped_ids.add(vol_id)
                    continue
                kept_ids.add(vol_id)
                zout.writestr(item, zin.read(item.filename))
                if not item.is_dir():
                    kept_files += 1
        tmp_path.replace(dest_zip)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    missing = sorted(keep_ids - kept_ids)
    if missing:
        raise RuntimeError(
            f"Kaggle ZIP is missing v2 nodules: {missing[:20]}"
            + (" …" if len(missing) > 20 else "")
        )
    rows = [
        {
            "subset_nodule_id": nid,
            "kept": True,
        }
        for nid in sorted(kept_ids)
    ]
    rows.extend(
        {"subset_nodule_id": nid, "kept": False} for nid in sorted(dropped_ids)
    )
    print(
        f"Wrote {dest_zip}: {len(kept_ids)} nodules kept, "
        f"{len(dropped_ids)} dropped, {kept_files} files"
    )
    return pd.DataFrame(rows)


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
