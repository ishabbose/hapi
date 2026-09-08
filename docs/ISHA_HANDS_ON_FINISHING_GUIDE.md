# Isha's Hands-On Finishing Guide

This guide is a starting point for the final implementation pass. The code is organized so that you can run each stage, inspect its outputs, explain the design decisions, and extend individual components without rewriting the full pipeline. Treat the commands, configuration files, prediction archives, and result tables as part of the research record.

## 1. Locked study question

The final study asks:

> Can a 2.5D residual segmentation network represent the spatial hierarchy of four-reader contour agreement in candidate-centered lung-nodule regions more accurately and more consistently than hard-consensus and unconstrained alternatives?

The input is a candidate-centered, `128 x 128` axial image region with the immediately previous, current, and next slices stacked as channels. The output is a set of four cumulative segmentation probabilities:

\[
T_k(x)=\mathbb{1}[V(x)\ge k], \qquad k\in\{1,2,3,4\},
\]

where \(V(x)\) is the number of the four LIDC-IDRI review sessions contributing a contour at voxel \(x\). Thus, `T1` is the union support region, `T2` is the at-least-two-reader region, `T3` is the at-least-three-reader region, and `T4` is the four-reader region. The empirical support target is

\[
q(x)=V(x)/4=\frac{1}{4}\sum_{k=1}^{4}T_k(x).
\]

The study is specifically candidate-ROI segmentation. It does not evaluate full-scan nodule detection or malignancy classification, because neither full-scan negative candidates nor verified malignancy labels are present in this release.

### Narrow novelty statement

The proposed contribution is a 2.5D residual U-Net that directly predicts all four cumulative reader-support regions, guarantees their voxelwise nesting by construction, and jointly calibrates their expected reader-support fraction. The design is tested against hard `T2` baselines, dimensional and output-parameterization ablations, an optional depth-context transformer, an architecture-matched exact-count ordinal comparator, and a no-Brier loss ablation that isolates the support-calibration objective. The contribution is the combination and controlled evaluation of cumulative support segmentation, exact output ordering, local through-plane context, and soft support calibration in this candidate-ROI setting; it should not be described as the first multi-rater or ordinal segmentation method.

## 2. Final dataset release

The validated `v3.1` dataset contains:

- 325 nodule candidates from 247 patients;
- 2,000 axial slices;
- 88 cases with one nonempty clustered contour, 62 with two, 69 with three, and 106 with four;
- five patient-disjoint outer folds of 65 candidates each;
- 237 cases with a nonempty `T2` target; and
- raw `uint8` image volumes, four binary reader-mask slots, vote-count and vote-fraction volumes, `T1` through `T4` masks, disagreement entropy, provenance, and relative paths.

This release repairs numeric z-order, preserves the original image frame, and never crops or recenters an input with its target mask. Intensity normalization is not baked into the stored images. Instead, each run fits its 1st and 99th percentile values on the three optimization folds and applies the frozen transform to the validation and outer-test folds.

The source audit motivates each correction:

| Audited issue | Final treatment |
|---|---|
| Image filenames were previously sorted lexicographically while masks were ordered numerically, misaligning 636 of 2,000 slice positions across 40 cases | Reconstruct every image and mask volume in natural numeric z-order and validate shared shapes and target identities |
| The prior in-plane crop and recenter transform was derived from the union ground-truth mask in every case | Preserve the full supplied `128 x 128` image frame; no target-derived spatial transform is allowed |
| A single precomputed normalized image would expose validation or outer-test intensities during cross-validation | Store raw `uint8`; fit the intensity window separately inside each run's optimization folds |
| Single consensus masks discard the structure of reader disagreement | Retain four mask slots, vote counts, `q`, entropy, and all cumulative targets `T1` through `T4` |
| A single legacy split cannot provide a complete out-of-fold comparison | Freeze five patient-grouped, contour-count-stratified outer folds |

Earlier autoencoder MSE, predicted-positive fractions, and preliminary consensus Dice values remain exploratory diagnostics. They are not transferred into the final paper because they do not use the locked `v3.1` task, model matrix, or outer-fold protocol.

The frozen release fingerprints are:

| Artifact | SHA-256 |
|---|---|
| Dataset array content | `a4b5725765518f42c36acc29cbe0b65e1a2508f2ee5f695ee032702bff76fd09` |
| `final_325_manifest.csv` | `0d0e7e5950cf71371e7ac4532aa80ab44a4b3e9b303d570ab646a7e9fec9d06a` |
| `final_2000_slice_manifest.csv` | `692dc1fa8838b7ae6d05e6679b45d32d783cb26a2995948bf9cfa7615aae3830` |

The manifest hashes freeze the exact patient-fold assignments used for every reported run. The current release was built with `sklearn.model_selection.StratifiedGroupKFold` from scikit-learn 1.8.0, which is recorded in `dataset_summary.json`.

The main files are:

| File | Purpose |
|---|---|
| `data/final_325_nodules_v3/final_325_manifest.csv` | One row per candidate, patient grouping, outer fold, source paths, and audit statistics |
| `data/final_325_nodules_v3/final_2000_slice_manifest.csv` | One row per axial slice |
| `data/final_325_nodules_v3/dataset_summary.json` | Cohort and split summary |
| `data/final_325_nodules_v3/validation_report.json` | Strict validation result and dataset, case-manifest, and slice-manifest fingerprints |
| `data/final_325_nodules_v3/cases/<case_id>/` | Image, reader masks, agreement targets, entropy, and case metadata |

Keep the supplied `v2` directory unchanged as source provenance. Use only `v3.1` for the final experiments.

## 3. Environment and dataset validation

Run every command from the repository root. Python 3.10 or newer and a CUDA-capable PyTorch installation are recommended for the full matrix.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "scikit-learn==1.8.0"
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Confirm that the intended accelerator is visible before launching a suite:

```bash
nvidia-smi
python -c "import torch; print({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})"
```

If the supplied archive has not yet been extracted, place it in the repository root and run:

```bash
python -m zipfile -e processed_325_nodules_v2.zip data
```

The result must be `data/processed_325_nodules_v2/nodule_001`, not a second nested `processed_325_nodules_v2` directory.

Rebuild and validate the final release:

```bash
python analysis/audit_dataset.py \
  data/processed_325_nodules_v2 \
  --output-dir analysis/v2_audit

python scripts/19_build_final_v3.py \
  --source data/processed_325_nodules_v2 \
  --output data/final_325_nodules_v3 \
  --seed 42

python scripts/20_validate_final_v3.py \
  --data-root data/final_325_nodules_v3
```

If `final_325_manifest.csv` already exists, do not overwrite it during an experiment. A deliberate clean rebuild uses `--force`, followed immediately by validation. Before training, open `validation_report.json` and verify all of the following:

- `status` is `PASS`;
- `n_cases_checked` is 325;
- `n_patients` is 247;
- `n_slices` is 2,000;
- `n_failures` is 0; and
- `dataset_content_sha256` matches the dataset-array fingerprint above;
- `manifest_sha256` is `0d0e7e5950cf71371e7ac4532aa80ab44a4b3e9b303d570ab646a7e9fec9d06a`; and
- `slice_manifest_sha256` is `692dc1fa8838b7ae6d05e6679b45d32d783cb26a2995948bf9cfa7615aae3830`.

Also compile the implementation before spending GPU time:

```bash
python -m compileall -q scripts src/hapi
```

## 4. Read the implementation before running it

The final implementation is split into inspectable units:

| File | What to inspect |
|---|---|
| `analysis/audit_dataset.py` | Read-only source audit, reproducibility checks, hashes, target statistics, and provenance tables |
| `src/hapi/agreement_dataset.py` | Adjacent-slice construction, fold-fitted intensity scaling, target loading, volume padding, and augmentation |
| `src/hapi/agreement_model.py` | Residual encoder-decoder, GroupNorm, ordered head, optional slice-token transformer, and exact-count head |
| `src/hapi/agreement_losses.py` | Four-session target semantics and the focal, volume-Dice, and balanced-Brier objective |
| `src/hapi/agreement_metrics.py` | 3D case reconstruction, target-positive overlap, empty-aware secondary summaries, surface distance, calibration, and patient bootstrap |
| `scripts/21_train_agreement_model.py` | Fold roles, normalization fitting, optimization, early stopping, threshold selection, and provenance |
| `scripts/22_evaluate_agreement_model.py` | Outer-test-only evaluation, saved predictions, metric tables, and evaluation summary |
| `scripts/23_run_ablation_suite.py` | Dry-run planning, safe resumption, suite execution, aggregation, and figure export |
| `scripts/24_aggregate_final_results.py` | Run-coverage checks, out-of-fold assembly, patient-level bootstrap, paired comparisons, and fail-closed aggregation |
| `scripts/25_export_paper_figures.py` | Data-backed paper figures and deterministic qualitative-case selection |

Before the full run, be able to explain these three points in your own words:

1. The ordered head constructs \(z_1=a\) and \(z_k=z_{k-1}-\operatorname{softplus}(g_k)\), which guarantees \(p_1\ge p_2\ge p_3\ge p_4\) after the sigmoid.
2. The loss combines case-balanced focal BCE, target-positive volume-level soft Dice, and a foreground/background-balanced Brier term for \(\hat q=\frac{1}{4}\sum_k p_k\). The default component weights are `0.45`, `0.45`, and `0.10`, and the `T1` through `T4` head weights are `0.4`, `0.3`, `0.2`, and `0.1`.
3. For outer fold \(f\), fold \((f+1)\bmod 5\) is used for validation and threshold selection, the other three folds are used for optimization, and fold \(f\) remains unopened until evaluation.

## 5. Integration checks

Smoke runs confirm that data loading, forward and backward passes, checkpointing, and outer-fold evaluation work together. They are never paper results and are stored under `results/final_agreement/_smoke/` so the final aggregator cannot consume them accidentally. First print the planned jobs without executing them:

```bash
python scripts/23_run_ablation_suite.py --suite smoke
```

The plan must contain exactly two runs: `A4_proposed` and `A7_ordinal_rps`, both using seed 42 and outer fold 0. Then execute it:

```bash
python scripts/23_run_ablation_suite.py \
  --suite smoke \
  --execute \
  --device cuda \
  --workers 4 \
  --evaluation-bootstrap 20
```

If CUDA is unavailable, substitute `--device cpu --no-amp` for this integration check. Do not use CPU for the complete experiment matrix unless the runtime has been estimated and accepted.

## 6. Locked model and ablation matrix

Run the same folds, seeds, optimizer settings, stopping rule, and evaluation code for every variant.

| ID | Configuration | Question answered |
|---|---|---|
| `A0_unet2d_t2` | 2D U-Net, hard `T2` | What does the original single-consensus formulation achieve? |
| `A1_resunet2d_t2` | 2D residual U-Net, hard `T2` | Does the residual backbone alone change performance? |
| `A2_resunet2p5d_t2` | 2.5D residual U-Net, hard `T2` | What does adjacent-slice context add without agreement modeling? |
| `A3_resunet2d_nested` | 2D residual U-Net, ordered `T1`-`T4` | What does ordered multi-support supervision add without 2.5D input? |
| `A4_proposed` | 2.5D residual U-Net, ordered `T1`-`T4` | Full proposed model |
| `A5_independent_heads` | 2.5D agreement model with four unconstrained heads | Is exact nesting useful beyond multi-head supervision? |
| `A6_transformer` | Proposed model plus two-layer slice-token transformer | Does full-depth conditioning add value beyond local 2.5D context? |
| `A7_ordinal_rps` | Architecture-matched exact-count `K=0,...,4` model with ranked probability score | How does cumulative segmentation compare with exact-count ordinal modeling? |
| `A8_no_brier` | Proposed 2.5D ordered model with the Brier component removed | Does the support-calibration objective improve calibration or overlap when architecture and supervision remain fixed? |

The final matrix is five outer folds by three deterministic seeds (`41`, `42`, and `43`) for all nine variants: 135 trained checkpoints. Seed `42` alone may be used to estimate runtime and inspect behavior, but it is a pilot result until the complete locked matrix is present.

### One canonical GPU run

```bash
python scripts/21_train_agreement_model.py \
  --variant A4_proposed \
  --outer-fold 0 \
  --seed 42 \
  --epochs 80 \
  --patience 15 \
  --batch-size 1 \
  --accumulation-steps 4 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --base-channels 24 \
  --dropout 0.10 \
  --workers 4 \
  --device cuda

python scripts/22_evaluate_agreement_model.py \
  --run-dir results/final_agreement/A4_proposed/seed_42/fold_0 \
  --device cuda \
  --batch-size 1 \
  --workers 4 \
  --bootstrap 2000
```

Automatic mixed precision is enabled on CUDA. If GPU memory is insufficient, retain `--batch-size 1` and reduce data-loader workers before changing the architecture. Do not lower `base-channels`, change the loss, or alter the split after seeing outer-test results.

### Orchestrated screening and full run

The suite runner is dry-run by default, skips a run only after finding a valid completed evaluation, evaluates an existing complete checkpoint when metrics are missing, and stops at a nonempty incomplete run directory. When `--execute` is supplied, it reruns dataset validation before starting a training job. Use it instead of copying 135 commands by hand.

First, print and review the 45-run seed-42 screening matrix:

```bash
python scripts/23_run_ablation_suite.py --suite screen
```

Execute the screening suite after confirming `n_runs: 45`, nine variants, seed 42, and folds 0 through 4:

```bash
python scripts/23_run_ablation_suite.py \
  --suite screen \
  --execute \
  --device cuda \
  --workers 4 \
  --evaluation-bootstrap 2000 \
  --aggregate-bootstrap 10000
```

The screening suite automatically writes aggregate tables and figures under `results/final_agreement/aggregates/screen/`. Use them to verify runtime, learning curves, thresholds, metric direction, and figure integrity. Do not change the locked architecture or analysis after this inspection.

Next, print the full 135-run plan. Completed seed-42 screening runs should appear as `complete`, leaving the other 90 runs pending:

```bash
python scripts/23_run_ablation_suite.py --suite full
```

Execute the complete matrix:

```bash
python scripts/23_run_ablation_suite.py \
  --suite full \
  --execute \
  --device cuda \
  --workers 4 \
  --evaluation-bootstrap 2000 \
  --aggregate-bootstrap 10000
```

After all model evaluations finish, the full suite automatically performs strict aggregation and produces the final figures under `results/final_agreement/aggregates/full/`. The `confirm` suite remains a three-seed comparison limited to `A0`, `A4`, and `A7`; it does not replace the full nine-variant matrix required for the final ablation figure.

Do not pass `--force-incomplete` routinely. If the suite reports an incomplete directory, inspect that exact run, preserve its logs, determine why it stopped, and use `--force-incomplete` only when restarting that run is justified.

### Multi-GPU scheduling

Each `(variant, seed, fold)` directory is independent, so separate GPUs can process disjoint variant groups. Assign one explicit `CUDA_VISIBLE_DEVICES` value to each shell, keep one run per GPU initially, and confirm memory use with `nvidia-smi`. Pass `--skip-aggregate --skip-figures` to each worker, then run the strict aggregation once after every worker has finished. For example:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/23_run_ablation_suite.py \
  --suite full \
  --variants A0_unet2d_t2,A1_resunet2d_t2,A2_resunet2p5d_t2,A3_resunet2d_nested \
  --execute --device cuda --workers 4 --skip-aggregate --skip-figures

CUDA_VISIBLE_DEVICES=1 python scripts/23_run_ablation_suite.py \
  --suite full \
  --variants A4_proposed,A5_independent_heads,A6_transformer,A7_ordinal_rps,A8_no_brier \
  --execute --device cuda --workers 4 --skip-aggregate --skip-figures
```

Do not allow two jobs to write the same run directory. Save the terminal log for each GPU worker, and record GPU model, VRAM, CUDA version, PyTorch version, wall-clock time, and repository commit in the experiment notes.

## 7. Run outputs and how to inspect them

Every training run writes to:

```text
results/final_agreement/<variant>/seed_<seed>/fold_<outer_fold>/
```

Training produces:

| Output | Meaning |
|---|---|
| `config.json` | Complete run settings, fold roles, parameter count, dataset and manifest fingerprints, fold-fitted intensity window, and software/device record |
| `split_assignments.csv` | Candidate and patient IDs assigned to optimization, validation, and unopened outer test |
| `history.csv` | Per-epoch training and validation objectives |
| `best.pt` | Best validation-selected model, optimizer state, epoch, and threshold |
| `validation_thresholds.csv` | Patient-macro `T2` Dice across validation-only threshold candidates |
| `run_summary.json` | Completion state, best epoch, best validation score, and selected threshold |

Evaluation adds:

| Output | Meaning |
|---|---|
| `predictions/<case_id>.npz` | Held-out probabilities and threshold for each candidate in the outer fold |
| `case_metrics.csv` | Target-positive 3D overlap and surface metrics by candidate and support level |
| `patient_metrics.csv` | Patient-level target-positive summaries |
| `all_case_patient_metrics.csv` | Secondary empty-aware summaries |
| `target_summary.csv` | Fold-level `T1`-`T4` summary |
| `bootstrap_cis.csv` | Fold diagnostic confidence intervals |
| `soft_case_metrics.csv` | Support-fraction Brier and nesting diagnostics for agreement models |
| `soft_patient_metrics.csv` | Patient-level soft-target diagnostics |
| `calibration.csv` | Reader-support reliability bins for the full ROI and `T1` union |
| `evaluation_summary.json` | Outer-test cardinality, threshold provenance, primary fold metrics, and prediction location |

The hard `T2` variants (`A0`-`A2`) intentionally do not create soft-agreement or calibration files. Surface distances are reported in index-space voxel units because native physical spacing is not available in the PNG-derived source. HD95 is the 95th percentile of the concatenated bidirectional surface-to-surface distances, and ASSD is their mean. If either the target or prediction is empty, the surface distance is undefined and the case is handled by the separately reported support-presence metrics.

All nine variants retain the same all-case optimization and validation cohorts. For the hard `T2` baselines, candidates with an empty `T2` mask are meaningful negative-support examples rather than missing labels.

After each run, check that `config.json` has `"smoke": false`, `run_summary.json` reports `TRAINING_COMPLETE`, `evaluation_summary.json` reports `EVALUATION_COMPLETE`, and the recorded `dataset_content_sha256` matches the validated dataset.

## 8. Fail-closed aggregation

The `full` suite runs this aggregation automatically. If training was divided across multiple GPUs with automatic aggregation disabled, run it once after all 135 train/evaluate pairs have completed:

```bash
python scripts/24_aggregate_final_results.py \
  --results-root results/final_agreement \
  --output-root results/final_agreement/aggregates/full \
  --variants A0_unet2d_t2,A1_resunet2d_t2,A2_resunet2p5d_t2,A3_resunet2d_nested,A4_proposed,A5_independent_heads,A6_transformer,A7_ordinal_rps,A8_no_brier \
  --seeds 41,42,43 \
  --folds 0,1,2,3,4 \
  --baseline-variant A0_unet2d_t2 \
  --prediction-variant A4_proposed \
  --prediction-seed 42 \
  --calibration-scope t1_union \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 20260904 \
  --confidence 0.95
```

This command is intentionally strict. It refuses to create or update paper tables unless every requested run exists and every model/seed has exactly one out-of-fold prediction for each of the 325 candidates from 247 patients, including exactly 237 target-positive `T2` cases. On incomplete coverage, inspect `results/final_agreement/aggregates/full/run_coverage_report.json`, repair only the named runs, and rerun aggregation.

For every model, aggregation first averages each case-level out-of-fold metric across seeds 41, 42, and 43, then averages candidates within patient, and finally computes the unweighted mean across patients. Confidence intervals resample patients. This is a metric-level repeated-seed aggregation; model predictions are not pooled across seeds.

Successful aggregation writes:

- `ablation_metrics.csv`: primary patient-macro target-positive `T2` Dice and patient-cluster 95% confidence intervals for all variants;
- `agreement_metrics.csv`: `T1` through `T4` target-positive patient-macro Dice for agreement-capable variants;
- `detailed_metrics.csv`: target-positive Dice, IoU, precision, recall, HD95, and ASSD estimates and intervals;
- `presence_metrics.csv`: target-presence sensitivity, specificity, precision, and F1 summaries;
- `calibration.csv`: count-weighted reader-support reliability bins for the selected scope;
- `history.csv`: ordered histories for all requested runs;
- `paired_comparisons.csv`: patient-paired `T2` Dice differences relative to `A0` with 95% bootstrap confidence intervals;
- `final_results_summary.json`: model, stratum, calibration, coverage, and provenance summaries;
- `run_coverage_report.json`: the validated run and out-of-fold panel inventory; and
- `predictions/`: the verified seed-42 `A4_proposed` out-of-fold prediction set for qualitative figures.

## 9. Checks before calling the results final

Do not copy numbers into the paper until every item below is true.

- The dataset validator reports `PASS` with the expected fingerprint and zero failures.
- Every final run is outside `_smoke`, and every `config.json` records the same dataset and manifest fingerprints.
- All nine variants have five evaluated outer folds for each of seeds 41, 42, and 43.
- Every patient occurs in one outer fold only, and no outer-test case appears in optimization or validation for its run.
- Fold-specific normalization is fitted only on the optimization patients.
- Checkpoint epoch and threshold selection use validation data only; outer-test data is not used to choose an epoch, threshold, architecture, or loss.
- `results/final_agreement/aggregates/full/final_results_summary.json` reports `COMPLETE` and confirms 325 candidates, 247 patients, and 237 target-positive `T2` cases for each complete out-of-fold panel.
- Each model/seed has exactly one held-out prediction per candidate, with no duplicated or missing IDs.
- Primary `T2` Dice values are finite and fall in `[0,1]`; overlap, precision, recall, Brier, and nesting rates satisfy their documented bounds.
- The ordered proposed model has a zero nesting-violation rate within the evaluator tolerance. The independent-head ablation is allowed to violate nesting and should be reported as measured.
- Patient-macro estimates and patient-cluster bootstrap intervals, not pooled slice-level values, are used in the abstract, tables, and claims.
- Paired differences are read from `paired_comparisons.csv`; a confidence interval crossing zero is described as inconclusive rather than as an improvement.
- Surface distances are labeled in voxel units.
- The main model remains `A4_proposed`; `A6_transformer` and `A8_no_brier` remain mechanism ablations even if either point estimate is higher.
- Qualitative cases are selected by the deterministic disagreement-spectrum rule or by a rule fixed before viewing predictions, never by visual quality or Dice.
- The exact code commit, commands, seeds, hardware, software versions, and archived aggregate tables are recorded.

## 10. Generate the paper figures

After aggregation reports `COMPLETE`, generate the figures directly from the validated tables and out-of-fold predictions:

```bash
python scripts/25_export_paper_figures.py \
  --results-root results/final_agreement/aggregates/full \
  --data-root data/final_325_nodules_v3 \
  --output-dir results/final_agreement/aggregates/full/paper_figures \
  --model "Proposed 2.5D ordered agreement model" \
  --qualitative-pool all \
  --num-qualitative 3 \
  --calibration-bins 10 \
  --dpi 600
```

`--qualitative-pool all` is valid here because aggregation has already verified and assembled exactly one out-of-fold seed-42 prediction per case. The default selection sorts prediction-backed, `T2`-positive cases by their source-defined reader-disagreement fraction and chooses evenly spaced ranks. Keep that automatic selection for the main paper. If an explicit set is required by a journal revision, record the rule first and pass the fixed IDs:

```bash
python scripts/25_export_paper_figures.py \
  --results-root results/final_agreement/aggregates/full \
  --data-root data/final_325_nodules_v3 \
  --output-dir results/final_agreement/aggregates/full/paper_figures \
  --model "Proposed 2.5D ordered agreement model" \
  --qualitative-pool all \
  --qualitative-cases nodule_XXX,nodule_YYY,nodule_ZZZ \
  --dpi 600
```

The exporter creates both 600-dpi PNG and vector PDF versions:

1. `figure_01_ablation_t2_dice`: patient-macro `T2` Dice with patient-level 95% intervals;
2. `figure_02_agreement_level_dice`: proposed-model Dice across `T1` through `T4`;
3. `figure_03_reader_support_calibration`: predicted versus observed reader-support fraction;
4. `figure_04_qualitative_candidate_rois`: CT ROI, observed support, `T2` target, and proposed probability; and
5. `figure_05_training_curve`: training and validation objective history.

Open every PNG at 100% and every PDF in a vector viewer. Check labels, axes, confidence intervals, case IDs, color bars, text size, and clipping. The white contour in each qualitative probability panel uses the threshold selected only on that case's fold-specific validation patients and stored in its prediction archive. Use `figure_manifest.csv` to transfer the recorded sources, case IDs, selection rule, and notes into the captions.

Recommended caption skeletons:

- **Fig. 1.** Patient-macro three-dimensional Dice for the target-positive `T2` cohort across the prespecified ablations. Each candidate's out-of-fold metric was first averaged across three seeds, after which candidates were averaged within patient. Error bars indicate 95% patient-cluster bootstrap confidence intervals.
- **Fig. 2.** Patient-macro three-dimensional Dice of the proposed model for cumulative reader-support targets `T1` through `T4`. Each level is evaluated among cases with a nonempty target at that level.
- **Fig. 3.** Reliability of the proposed support estimate \(\hat q\) against the observed four-session support fraction \(q\) within the `T1` union. Marker size is proportional to voxel count; the dashed line denotes perfect calibration.
- **Fig. 4.** Deterministically selected candidate ROIs spanning the source-defined reader-disagreement spectrum. Columns show the source image, observed support fraction, `T2` target, and proposed `T2` probability. The displayed slice has maximal `T2` area, with the middle slice selected for ties.
- **Fig. 5.** Training and validation objectives for the proposed model across the final folds and seeds. Curves show epoch means and the shaded region shows the observed run range.

## 11. Populate the paper draft from final outputs

Use this map so that every reported value remains traceable:

| Paper field | Final source |
|---|---|
| Completed run count and out-of-fold coverage | `run_coverage_report.json` and `final_results_summary.json` |
| `A0`-`A8` primary `T2` Dice and 95% intervals | `ablation_metrics.csv` |
| Each model's paired difference from `A0` and its interval | `paired_comparisons.csv` |
| Proposed `T1`-`T4` Dice and target-positive counts | `agreement_metrics.csv` |
| IoU, precision, recall, HD95, and ASSD | `detailed_metrics.csv` |
| Empty-target support-presence behavior | `presence_metrics.csv` |
| Balanced Brier, nesting rate, contour-count strata, calibration ECE, and aggregation policy | `final_results_summary.json` |
| Reliability-bin values and voxel counts | `calibration.csv` |
| Qualitative case IDs, selection rule, sources, and figure notes | `paper_figures/figure_manifest.csv` |

Keep full-precision tables unchanged. Round only in the manuscript, use one consistent precision for a metric and both confidence bounds, and verify that rounding does not turn a confidence bound near zero into a stronger claim.

## 12. IEEE-style paper draft

Replace every bracketed field only with a value read from the final aggregate outputs. Keep the task name, cohort definition, averaging unit, and confidence-interval method attached to each number.

### Methods

#### A. Dataset and task definition

We studied multi-reader pulmonary-nodule segmentation using a candidate-centered subset derived from LIDC-IDRI [CIT-LIDC]. The finalized cohort contained 325 nodule candidates from 247 patients and 2,000 axial slices. Each candidate was represented by an 8-bit, `128 x 128` image volume and four source mask slots encoding the first through fourth clustered annotations. Consistent with the four-reader review protocol, an empty higher slot was treated as zero support rather than as an unavailable label; individual radiologist identities were not retained. The contour-count distribution was 88, 62, 69, and 106 candidates with one, two, three, and four nonempty clustered contours, respectively. Because the supplied images were candidate-centered PNG-derived regions, the experimental task was segmentation within a provided nodule ROI rather than full-scan candidate detection. No malignancy endpoint was used.

For voxel \(x\), let \(V(x)\in\{0,1,2,3,4\}\) denote the number of review sessions contributing a contour. Four cumulative targets were defined as \(T_k(x)=\mathbb{1}[V(x)\ge k]\), for \(k\in\{1,2,3,4\}\), and the empirical reader-support fraction was \(q(x)=V(x)/4\). Consequently, the targets obeyed \(T_1\supseteq T_2\supseteq T_3\supseteq T_4\). The primary endpoint used the at-least-two-reader target, `T2`; 237 candidates had a nonempty `T2` region.

#### B. Preprocessing and partitioning

Volumes were reconstructed from the preserved source PNGs using numeric slice order. Images remained in the original `128 x 128` frame; no ground-truth contour was used for cropping, recentering, or padding. Raw `uint8` intensities were stored in the finalized dataset. For each outer-fold run, the 1st and 99th intensity percentiles were estimated using only the optimization patients, after which the frozen linear scaling was applied to the validation and outer-test patients. During optimization, a single spatial transform was applied consistently to every image and target slice in a volume: a rotation sampled from 0, 90, 180, or 270 degrees and independent horizontal and vertical flips, each with probability 0.5. The augmentation schedule was deterministic for a given run seed, epoch, and sample index. No augmentation was applied during validation or outer-test evaluation.

We used five patient-disjoint outer folds stratified by the number of nonempty clustered contours. Each fold contained 65 candidates. For outer fold \(f\), fold \((f+1)\bmod 5\) served as the validation fold and the remaining three folds served as optimization data. The outer fold was excluded from normalization fitting, parameter optimization, checkpoint selection, and threshold selection. Each model produced one out-of-fold prediction for every candidate.

#### C. Proposed architecture

The proposed `NestedAgreementResUNet2p5D` used a residual U-Net encoder-decoder [CIT-UNET] with Group Normalization and SiLU activations. The previous, current, and next axial slices formed a three-channel 2.5D input for each predicted slice; the nearest valid slice was replicated at volume boundaries. Encoder widths were 24, 48, 96, and 192 channels. Bilinear decoder upsampling was followed by skip-feature fusion and residual convolution. Dropout was set to 0.10 in the deeper stages.

The decoder produced four cumulative agreement logits. Given an unconstrained first logit \(a\) and three learned gap fields \(g_k\), the output was parameterized as \(z_1=a\) and \(z_k=z_{k-1}-\operatorname{softplus}(g_k)\) for \(k=2,3,4\). Since the sigmoid is monotonic, this construction guaranteed \(p_1(x)\ge p_2(x)\ge p_3(x)\ge p_4(x)\) at every voxel without a post-processing projection or ordering penalty. The predicted reader-support fraction was \(\hat q(x)=\frac{1}{4}\sum_{k=1}^{4}p_k(x)\).

#### D. Objective and optimization

The training objective combined focal binary cross-entropy, target-positive volume-level soft Dice loss, and a reader-support Brier term with weights 0.45, 0.45, and 0.10, respectively. Agreement-head weights for `T1` through `T4` were 0.4, 0.3, 0.2, and 0.1. Focal parameters were \(\alpha=0.75\) and \(\gamma=2\). The Brier component compared \(\hat q\) with \(q\) and averaged foreground and background errors within each candidate to reduce background dominance. Focal loss included all valid voxels and legitimate empty higher-support targets; the Dice component was evaluated only for candidate-head pairs with a nonempty target. In the no-Brier ablation, the Brier weight was set to zero and the focal and Dice terms were renormalized to equal weight while the architecture, inputs, targets, and remaining training protocol were unchanged.

Models were optimized with AdamW using an initial learning rate of \(3\times10^{-4}\), weight decay of \(10^{-4}\), a cosine schedule, batch size one, four-step gradient accumulation, and gradient-norm clipping at 1.0. Training continued for at most 80 epochs with patience 15. Automatic mixed precision was used on CUDA hardware. The checkpoint maximizing validation patient-macro `T2` Dice, with validation loss as the tie-breaker, was retained. The binary operating threshold was then selected on the validation patients from values 0.15 to 0.85 in increments of 0.025.

#### E. Comparators and ablations

We evaluated nine prespecified configurations: a 2D U-Net with a hard `T2` target; a 2D residual U-Net with hard `T2`; its 2.5D counterpart; a 2D ordered four-target model; the full 2.5D ordered model; a four-head 2.5D model without ordering constraints; the ordered model augmented with a two-layer, four-head slice-token transformer at the bottleneck; an architecture-matched exact-count model predicting \(K\in\{0,1,2,3,4\}\); and the proposed ordered model trained without the Brier component. The exact-count comparator used a five-class softmax and minimized `T2` foreground binary cross-entropy plus 0.8 times the ranked probability score [CIT-ORDINAL]. The no-Brier configuration isolated the contribution of reader-support calibration while retaining the proposed architecture and cumulative targets. All configurations used the same all-case optimization and validation cohorts, patient folds, and evaluation implementation; an empty `T2` mask was treated as a valid negative-support target. Three deterministic seeds, 41, 42, and 43, were used for each fold and configuration.

#### F. Evaluation and statistical analysis

Predictions were reconstructed and evaluated as three-dimensional candidate volumes. The primary endpoint was patient-macro Dice for `T2` among the 237 candidates with nonempty `T2` targets. For each model, each candidate's case-level out-of-fold metric was first averaged across seeds 41, 42, and 43; candidates were then averaged within patient, and patients received equal weight. Model predictions themselves were not pooled across seeds. Secondary endpoints included target-positive Dice, intersection over union, precision, recall, 95th-percentile Hausdorff distance, and average symmetric surface distance for `T1` through `T4`; empty-aware all-case summaries; support-fraction Brier score; voxelwise nesting-violation rate; reader-support reliability; and contour-count strata. Surface distances were expressed in index-space voxel units. HD95 was the 95th percentile of the concatenated bidirectional surface-to-surface distances, and ASSD was their mean. A surface distance was undefined when either mask was empty; those cases were evaluated through the separately reported support-presence endpoints. Uncertainty was estimated using 10,000 patient-cluster bootstrap replicates with two-sided 95% percentile intervals. Model differences in primary Dice were computed on paired patient-level out-of-fold results using the same seed-then-patient aggregation and bootstrap design.

### Results

The finalized cohort included 325 candidates from 247 patients, with 65 candidates in each patient-disjoint outer fold. Dataset validation examined all 2,000 slices and 325 case directories and returned zero integrity failures. Exactly 237 candidates contributed to the target-positive primary `T2` endpoint. **[COMPLETED-RUNS]** of 135 prespecified model-seed-fold runs completed, producing **[OOF-COVERAGE-STATEMENT]** [VERIFY-FINAL-COVERAGE].

For the primary endpoint, the proposed 2.5D ordered agreement model achieved a patient-macro `T2` Dice of **[A4-DICE]** (95% CI, **[A4-CI-LOW]–[A4-CI-HIGH]**). The 2D U-Net hard-`T2` baseline achieved **[A0-DICE]** (95% CI, **[A0-CI-LOW]–[A0-CI-HIGH]**). The paired patient-level difference was **[A4-MINUS-A0]** (95% CI, **[DIFF-CI-LOW]–[DIFF-CI-HIGH]**). Report this interval as [supporting a positive difference / inconclusive / supporting a negative difference] according to whether it lies wholly above zero, crosses zero, or lies wholly below zero.

Across cumulative support levels, the proposed model achieved patient-macro Dice values of **[T1-DICE]**, **[T2-DICE]**, **[T3-DICE]**, and **[T4-DICE]** for `T1`, `T2`, `T3`, and `T4`, respectively; the corresponding 95% intervals were **[T1-CI]**, **[T2-CI]**, **[T3-CI]**, and **[T4-CI]**. The target-positive case counts at these levels were 325, 237, 175, and 106, respectively. Precision, recall, IoU, HD95, and ASSD are summarized in Table [TABLE-ID].

The ablation sequence yielded primary Dice values of **[A1-DICE]** for the 2D residual backbone, **[A2-DICE]** after adding adjacent-slice context to the hard target, **[A3-DICE]** for 2D ordered agreement supervision, **[A5-DICE]** for independent agreement heads, **[A6-DICE]** with bottleneck transformer conditioning, **[A7-DICE]** for the exact-count ordinal comparator, and **[A8-DICE]** for the no-Brier objective. The ordered and independent-head configurations produced primary Dice estimates of **[A4-DICE]** and **[A5-DICE]**, respectively, while their nesting-violation rates were **[A4-VIOLATION]** and **[A5-VIOLATION]**. The cumulative proposed model and architecture-matched exact-count comparator produced balanced support Brier scores of **[A4-BRIER]** and **[A7-BRIER]**, respectively. Holding the proposed architecture fixed, including versus omitting the Brier component yielded balanced Brier scores of **[A4-BRIER]** and **[A8-BRIER]**, expected calibration errors of **[A4-ECE]** and **[A8-ECE]**, and primary Dice values of **[A4-DICE]** and **[A8-DICE]**, respectively.

For reader-support estimation within **[CALIBRATION-SCOPE]**, the proposed model obtained a balanced Brier score of **[A4-BRIER]** and an expected calibration error of **[A4-ECE]**. The ordered construction yielded a measured nesting-violation rate of **[A4-VIOLATION]**. Performance by contour-count stratum was **[T2-DICE-C2]**, **[T2-DICE-C3]**, and **[T2-DICE-C4]** for candidates with two, three, and four clustered contours, respectively [VERIFY-AVAILABLE-STRATA].

Figure [QUAL-FIG-ID] displays candidates selected deterministically across the source-defined reader-disagreement spectrum, independent of model Dice. The panels show the candidate ROI, empirical support fraction, `T2` target, and out-of-fold proposed-model probability. Record the displayed case IDs as **[CASE-IDS]** from `figure_manifest.csv`; describe only patterns visible in the exported panels.

### Discussion

The principal finding was that cumulative, structurally ordered agreement modeling produced a patient-macro `T2` Dice of **[A4-DICE]**, corresponding to a paired change of **[A4-MINUS-A0]** relative to the hard-consensus 2D U-Net. Because the paired 95% interval **[DIFF-CI-LOW]–[DIFF-CI-HIGH]** [excluded/included] zero, the experiment provides [evidence/inconclusive evidence] that the proposed formulation changes candidate-ROI segmentation performance beyond the reference baseline.

The controlled ablations separate the effects of backbone, through-plane context, agreement supervision, and output ordering. The `A1` and `A2` primary estimates were **[A1-DICE]** and **[A2-DICE]**, respectively; the point estimate [increased/decreased/remained similar] after adding local adjacent-slice information under otherwise matched hard-target training. The `A3` and `A4` estimates were **[A3-DICE]** and **[A4-DICE]**, respectively, showing the corresponding point-estimate change from 2.5D context when ordered supervision was held fixed. Comparing `A4` with `A5`, the ordered parameterization produced a [higher/lower/similar] primary point estimate while changing the nesting-violation rate from **[A5-VIOLATION]** to **[A4-VIOLATION]**. These comparisons, rather than the proposed-versus-baseline comparison alone, identify which components are consistent with the observed result; inferential language is reserved for comparisons with a paired confidence interval.

The agreement-level analysis showed **[PATTERN-ACROSS-T1-T4]** as the required support increased from one to four readers. This pattern should be interpreted together with the decreasing target prevalence and the target-positive evaluation rule. The support-calibration result of **[A4-BRIER]** and reliability pattern in Fig. [CAL-FIG-ID] indicate **[CALIBRATION-INTERPRETATION]**. The comparison with `A7` further distinguishes direct cumulative-support prediction from an exact vote-count parameterization using the same residual 2.5D backbone. Because `A4` and `A8` share the same architecture and cumulative targets, their balanced Brier values (**[A4-BRIER]** versus **[A8-BRIER]**), expected calibration errors (**[A4-ECE]** versus **[A8-ECE]**), and Dice estimates (**[A4-DICE]** versus **[A8-DICE]**) isolate the empirical effect of the support-calibration term. These values indicate **[A4-VS-A8-INTERPRETATION]**, with lower Brier and ECE indicating better calibration.

The resulting claims are intentionally aligned with the available data: the method models multi-reader boundaries within supplied candidate-centered regions. It does not establish full-scan detection or malignancy prediction. A subsequent study can test those endpoints after candidate-negative scans, native DICOM geometry, and verified nodule-level malignancy mappings are incorporated under a new protocol.

## 13. Student completion checklist

- [ ] I can state the final research question and narrow novelty without claiming full-scan detection, malignancy prediction, or methodological priority over all multi-rater work.
- [ ] I read the dataset, model, loss, metric, trainer, evaluator, aggregator, and figure code and can trace one candidate from `uint8` volume to final patient-macro metric.
- [ ] I preserved the supplied source data and generated `v3.1` as a separate, reproducible release.
- [ ] I reran dataset validation and recorded the expected dataset-content, case-manifest, and slice-manifest fingerprints.
- [ ] I inspected at least three case directories and confirmed image/mask shape, numeric slice order, vote-count values, `q=vote_count/4`, and nested `T1`-`T4` masks.
- [ ] I completed both smoke model families and kept them under `_smoke` only.
- [ ] I ran one full seed-42 fold, estimated GPU time and storage, and inspected its configuration, split assignments, history, checkpoint, predictions, and metrics before scheduling the matrix.
- [ ] I completed all nine variants, five folds, and three seeds without changing the locked protocol after outer-test inspection.
- [ ] I saved per-run logs and recorded code commit, environment, GPU, CUDA, PyTorch, and wall-clock information.
- [ ] I ran the strict aggregator and obtained `status: COMPLETE` with the expected candidate, patient, and positive-target cardinalities.
- [ ] I checked patient-paired differences and confidence intervals before writing comparative language.
- [ ] I generated all figures from aggregate tables and verified both PNG and PDF exports visually.
- [ ] I copied case IDs and figure selection rules from `figure_manifest.csv`, not from memory.
- [ ] I replaced every paper placeholder with a traceable value and independently checked each value against its CSV or JSON source.
- [ ] I kept smoke metrics, pilot metrics, and validation-selection scores out of the abstract and final Results section.
- [ ] I excluded the earlier autoencoder MSE, predicted-positive fractions, and preliminary consensus Dice from the locked final result tables.
- [ ] I archived the validated dataset report, final code commit, complete aggregate result directory, commands, and paper figures together.

The project is ready to move into paper writing when this checklist is complete and the strict aggregator and figure exporter both finish without an exception.
