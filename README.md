# HAPI

Pipeline for building a lung-cancer imaging model from **LIDC-IDRI** (The Cancer Imaging Archive) and a 2,000-image Kaggle nodule subset.

The original exploration lives in Colab notebooks under `notebooks/`. Runnable work is split into numbered scripts so each stage can be executed **on its own** and **in order**. Later steps read CSVs and folders produced by earlier ones.

This repo does **not** assign binary cancer labels in the baseline experiments. The 3D model is an unsupervised reconstruction autoencoder. The 2D U-Net predicts reader-consensus nodule masks.

## Repository layout

```
content/          Datasets and the TCIA download manifest
data/             Generated artifacts, extracts, checkpoints (gitignored)
notebooks/        Original Colab notebooks (dataset + testing)
scripts/          Numbered pipeline steps
src/hapi/         Shared Python library used by the scripts
```

## Data

Place these files in `content/` (or point at them with environment variables):

| File | Role |
|------|------|
| `TCIA_LIDC-IDRI_20200921.tcia` | TCIA series manifest (already in the repo; 1,308 SeriesInstanceUIDs) |
| `kaggle_dataset_2000.zip` | Flattened ~2,000-image subset (`nodule_001/slice-0.png`, …) |
| `kagl_lidc_idri.zip` | Native LIDC crops with `images/` and `mask-0` … `mask-3` |

Official LIDC XML annotations are downloaded in step 03 from TCIA wiki (`LIDC-XML-only.zip`).

Cohort used for modeling:

- Hash-match the subset to native LIDC patient/nodule folders (~327 nodules).
- Drop two unresolved multi-series cases: `nodule_029`, `nodule_085`.
- Remaining **325** nodules, split **70 / 15 / 15** by **patient** (no patient in more than one split).

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run scripts from the **repository root**.

## Incremental pipeline

Each script writes under `data/`. If the main output already exists, the step **skips** unless you pass `--force`.

```bash
# Check that content files are present
python scripts/01_check_inputs.py

# One step at a time
python scripts/05_parse_tcia_manifest.py

# Range of steps (stops before training)
python scripts/run_pipeline.py --from-step 1 --to-step 12 --skip-train

# Full run including training
python scripts/run_pipeline.py

# Overwrite existing artifacts
python scripts/02_match_subset_to_lidc.py --force
```

| Step | Script | What it does | Main output |
|------|--------|----------------|-------------|
| 01 | `01_check_inputs.py` | Verify ZIPs and TCIA manifest | prints paths |
| 02 | `02_match_subset_to_lidc.py` | SHA-256 match subset PNGs to native LIDC | `data/artifacts/nodule_to_lidc_mapping.csv` |
| 03 | `03_download_lidc_xml.py` | Download and extract official XML | `data/lidc_xml_annotations/` |
| 04 | `04_build_xml_inventory.py` | Index XML patient/series/malignancy fields | `data/artifacts/lidc_xml_inventory.csv` |
| 05 | `05_parse_tcia_manifest.py` | Read SeriesInstanceUIDs from `.tcia` | `data/artifacts/tcia_series_uids.txt` |
| 06 | `06_map_tcia_series.py` | Query TCIA NBIA `getSeries` for PatientID / modality | `data/artifacts/tcia_series_to_patient_mapping.csv` |
| 07 | `07_link_nodules_to_ct.py` | Join nodules to CT series; keep 1:1 mappings | `data/artifacts/nodule_to_single_ct_series.csv` |
| 08 | `08_filter_cohort.py` | Exclude `029`/`085`; write 325-nodule master | `data/artifacts/final_325_master_manifest.csv` |
| 09 | `09_match_xml_to_series.py` | Match each CT series to LIDC XML | `data/artifacts/nodule_to_xml_candidates.csv` |
| 10 | `10_patient_splits.py` | Patient-isolated train/val/test | `data/artifacts/final_325_patient_split_manifest.csv` |
| 11 | `11_extract_processed.py` | Extract images + four reader masks from original ZIP | `data/processed_325_nodules/` |
| 12 | `12_normalize_volumes.py` | 1st–99th percentile window → `[0, 1]` `.npy` volumes | `data/model_ready_325_nodules/` |
| 13 | `13_train_3d_autoencoder.py` | Unsupervised 3D conv autoencoder (MSE reconstruction) | `data/checkpoints/3d_autoencoder.pt` |
| 14 | `14_train_2d_unet.py` | 2D U-Net on consensus masks | `data/checkpoints/unet2d_consensus.pt` |

Step 06 calls the public TCIA API once per series (~1,300 requests). Cache the CSV and do not re-run unless you need a refresh (`--force`).

Training knobs for 13 and 14:

```bash
python scripts/13_train_3d_autoencoder.py --epochs 10 --batch-size 4 --lr 1e-3
python scripts/14_train_2d_unet.py --epochs 5 --batch-size 8 --lr 1e-3
```

## Environment variables

Optional overrides (see `src/hapi/config.py`):

| Variable | Default |
|----------|---------|
| `HAPI_CONTENT_DIR` | `content/` |
| `HAPI_DATA_DIR` | `data/` |
| `HAPI_SUBSET_ZIP` | `content/kaggle_dataset_2000.zip` |
| `HAPI_ORIGINAL_ZIP` | `content/kagl_lidc_idri.zip` |
| `HAPI_TCIA_MANIFEST` | `content/TCIA_LIDC-IDRI_20200921.tcia` |

## Library (`src/hapi`)

Scripts import this package. You can use the same modules in a notebook after adding `src` to `PYTHONPATH`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))

from hapi.config import TCIA_MANIFEST
from hapi.tcia import parse_series_uids
```

Notable modules: `matching`, `tcia`, `xml_annotations`, `extract`, `preprocess`, `volumes`, `datasets`, `models`, `train`.

## Notebooks

`notebooks/dataset.ipynb` and `notebooks/testing.ipynb` are the original Colab sessions (paths under `/content/`, Drive mounts, exploratory duplicates). Prefer `scripts/` for local, incremental runs.

## License and data use

LIDC-IDRI is distributed by [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/). Follow TCIA usage and citation requirements for any publication or redistribution of images or annotations.
