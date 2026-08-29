"""Project paths and constants. Override with environment variables if needed."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONTENT_DIR = Path(os.environ.get("HAPI_CONTENT_DIR", PROJECT_ROOT / "content"))
DATA_DIR = Path(os.environ.get("HAPI_DATA_DIR", PROJECT_ROOT / "data"))
ARTIFACTS_DIR = DATA_DIR / "artifacts"
EXTRACTED_DIR = DATA_DIR / "extracted"
PROCESSED_DIR = DATA_DIR / "processed_325_nodules"
MODEL_READY_DIR = DATA_DIR / "model_ready_325_nodules"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"

SUBSET_ZIP = Path(
    os.environ.get("HAPI_SUBSET_ZIP", CONTENT_DIR / "kaggle_dataset_2000.zip")
)
ORIGINAL_ZIP = Path(
    os.environ.get("HAPI_ORIGINAL_ZIP", CONTENT_DIR / "kagl_lidc_idri.zip")
)
TCIA_MANIFEST = Path(
    os.environ.get(
        "HAPI_TCIA_MANIFEST",
        CONTENT_DIR / "TCIA_LIDC-IDRI_20200921.tcia",
    )
)

LIDC_XML_ZIP_URL = (
    "https://wiki.cancerimagingarchive.net/download/attachments/1966254/LIDC-XML-only.zip"
)
LIDC_XML_ZIP = DATA_DIR / "LIDC-XML-only.zip"
LIDC_XML_DIR = DATA_DIR / "lidc_xml_annotations"

TCIA_GET_SERIES_URL = (
    "https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries"
)

SEED = 42
EXCLUDED_NODULES = frozenset({"nodule_029", "nodule_085"})
EXPECTED_SUBSET_NODULES = 327
EXPECTED_ELIGIBLE_NODULES = 325

TARGET_DEPTH = 16
TARGET_SIZE = (64, 64)

# Artifact filenames
NODULE_TO_LIDC_CSV = ARTIFACTS_DIR / "nodule_to_lidc_mapping.csv"
XML_INVENTORY_CSV = ARTIFACTS_DIR / "lidc_xml_inventory.csv"
TCIA_SERIES_CSV = ARTIFACTS_DIR / "tcia_series_to_patient_mapping.csv"
NODULE_SERIES_CANDIDATES_CSV = ARTIFACTS_DIR / "nodule_to_series_candidates.csv"
NODULE_CT_CANDIDATES_CSV = ARTIFACTS_DIR / "nodule_to_ct_series_candidates.csv"
NODULE_SINGLE_CT_CSV = ARTIFACTS_DIR / "nodule_to_single_ct_series.csv"
ELIGIBLE_COHORT_CSV = ARTIFACTS_DIR / "final_eligible_325_nodules.csv"
EXCLUDED_CSV = ARTIFACTS_DIR / "excluded_nodules.csv"
NODULE_XML_CSV = ARTIFACTS_DIR / "nodule_to_xml_candidates.csv"
MASTER_MANIFEST_CSV = ARTIFACTS_DIR / "final_325_master_manifest.csv"
SPLIT_MANIFEST_CSV = ARTIFACTS_DIR / "final_325_patient_split_manifest.csv"
PATIENT_SPLITS_CSV = ARTIFACTS_DIR / "final_325_patient_splits.csv"
PROCESSED_MANIFEST_CSV = ARTIFACTS_DIR / "processed_325_manifest.csv"
PREPROCESS_MANIFEST_CSV = ARTIFACTS_DIR / "final_325_preprocessed_manifest.csv"
SUBSET_EXTRACT_DIR = EXTRACTED_DIR / "kaggle_dataset_2000"


def ensure_dirs() -> None:
    for path in (
        DATA_DIR,
        ARTIFACTS_DIR,
        EXTRACTED_DIR,
        PROCESSED_DIR,
        MODEL_READY_DIR,
        CHECKPOINTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def require_file(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {hint}: {path}")
    return path
