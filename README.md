# HAPI

Pipeline for building a lung-cancer imaging model from **LIDC-IDRI** (The Cancer Imaging Archive) and a 2,000-image Kaggle nodule subset.

The original exploration lives in Colab notebooks under `notebooks/`. Runnable work is split into numbered scripts so each stage can be executed **on its own** and **in order**. Later steps read CSVs and folders produced by earlier ones.

The original autoencoder and consensus U-Net are retained as exploratory history. The final study is a leakage-safe, patient-grouped comparison for multi-reader segmentation inside supplied candidate-centered nodule ROIs. It does **not** assign cancer labels or claim full-scan nodule detection.

## Final v3.1 study

The final dataset contains 325 candidate volumes from 247 patients and 2,000 axial slices. It preserves raw `uint8` source images, repairs numeric z-order, removes target-derived cropping/recentering, and stores four cumulative support targets:

```text
T1 = at least one supporting annotation (325 positive cases)
T2 = at least two supporting annotations (237 positive cases; primary endpoint)
T3 = at least three supporting annotations (175 positive cases)
T4 = four supporting annotations         (106 positive cases)
```

The proposed model is a residual 2.5D U-Net whose four logits are nested by construction. Final comparisons use five patient-disjoint outer folds, a separate validation fold inside every outer run, fold-fitted intensity scaling, all-case training cohorts, validation-only threshold selection, and patient-clustered bootstrap intervals.

Start with the student-facing implementation guide in `docs/ISHA_HANDS_ON_FINISHING_GUIDE.md`. The shortest verified workflow is:

```bash
python scripts/20_validate_final_v3.py --data-root data/final_325_nodules_v3
python scripts/23_run_ablation_suite.py --suite smoke
python scripts/23_run_ablation_suite.py --suite smoke --execute --device cuda --workers 4
python scripts/23_run_ablation_suite.py --suite full
python scripts/23_run_ablation_suite.py --suite full --execute --device cuda --workers 4
```

The suite runner is dry-run by default. Smoke checkpoints remain isolated under `_smoke`; non-smoke suites train, evaluate, aggregate, and export figures only after complete run coverage. Numerical smoke outputs are integration diagnostics, not paper results.

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
| `LIDC-XML-only.zip` | Official LIDC-IDRI XML annotations from TCIA wiki |

Step 03 uses `content/LIDC-XML-only.zip` if present; otherwise it downloads that archive from the TCIA wiki.

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
| 15 | `15_attach_tcia_manifest.py` | Attach `.tcia` basket fields + NBIA series row to processed 325 and Kaggle 2000 scans | `data/processed_325_nodules/*/metadata.json`, `content/kaggle_dataset_2000.zip` (`*/metadata.json`), `data/artifacts/processed_325_tcia_metadata.csv`, `data/artifacts/kaggle_2000_tcia_metadata.csv` |
| 16 | `16_run_inference.py` | Run AE / U-Net on processed v2 and the 325-nodule Kaggle ZIP | `data/artifacts/inference_autoencoder.csv`, `data/artifacts/inference_unet.csv` |
| 17 | `17_improve_processed_v2.py` | Train-only intensity, foreground crop/pad, consensus+union masks (new v2 dataset) | `data/processed_325_nodules_v2/`, `data/artifacts/processed_325_v2_manifest.csv` |
| 18 | `18_filter_kaggle_to_v2_cohort.py` | Keep only v2’s 325 unprocessed nodules in `content/kaggle_dataset_2000.zip` | `content/kaggle_dataset_2000.zip`, `content/kaggle_dataset_2000_original.zip`, `data/artifacts/kaggle_325_zip_cohort.csv` |
| 19 | `19_build_final_v3.py` | Rebuild target-independent raw volumes and T1–T4 reader-support targets | `data/final_325_nodules_v3/` |
| 20 | `20_validate_final_v3.py` | Validate all arrays, target identities, paths, hashes, and patient folds | `data/final_325_nodules_v3/validation_report.json` |
| 21 | `21_train_agreement_model.py` | Train one variant/seed/outer fold with validation-only selection | `results/final_agreement/<variant>/seed_<seed>/fold_<fold>/` |
| 22 | `22_evaluate_agreement_model.py` | Evaluate one frozen checkpoint on its outer test fold | Per-case metrics and held-out predictions in the run directory |
| 23 | `23_run_ablation_suite.py` | Dry-run or execute smoke, screening, confirmatory, and full suites | Complete experiment tree and aggregate outputs |
| 24 | `24_aggregate_final_results.py` | Fail-closed OOF validation, patient bootstrap, paired comparisons, and paper tables | `results/final_agreement/aggregates/<suite>/` |
| 25 | `25_export_paper_figures.py` | Export data-backed 600-dpi PNG/PDF figures and source manifest | `paper_figures/` under the aggregate directory |

Step 06 calls the public TCIA API once per series (~1,300 requests). Cache the CSV and do not re-run unless you need a refresh (`--force`).

Training knobs for 13 and 14:

```bash
python scripts/13_train_3d_autoencoder.py --epochs 10 --batch-size 4 --lr 1e-3
python scripts/14_train_2d_unet.py --epochs 5 --batch-size 8 --lr 1e-3
python scripts/15_attach_tcia_manifest.py
python scripts/16_run_inference.py --model both --dataset both
python scripts/16_run_inference.py --limit 8   # smoke test
python scripts/17_improve_processed_v2.py
python scripts/18_filter_kaggle_to_v2_cohort.py
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
| `HAPI_LIDC_XML_ZIP` | `content/LIDC-XML-only.zip` |

## Library (`src/hapi`)

Scripts import this package. You can use the same modules in a notebook after adding `src` to `PYTHONPATH`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))

from hapi.config import TCIA_MANIFEST
from hapi.tcia import parse_series_uids, parse_tcia_manifest
```

Notable modules: `matching`, `tcia`, `xml_annotations`, `extract`, `preprocess`, `improve_processed`, `volumes`, `datasets`, `models`, `train`.

## Notebooks

`notebooks/dataset.ipynb` and `notebooks/testing.ipynb` are the original Colab sessions (paths under `/content/`, Drive mounts, exploratory duplicates). Prefer `scripts/` for local, incremental runs.

## License and data use

LIDC-IDRI is distributed by [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/). Follow TCIA usage and citation requirements for any publication or redistribution of images or annotations.
