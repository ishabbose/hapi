#!/usr/bin/env python3
"""Validate and aggregate the locked final agreement-model evaluation runs.

This script is deliberately fail-closed.  It writes paper-facing aggregate files
only after every requested variant/seed/fold is present and the out-of-fold case
panels pass all identity, cardinality, and metric checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "final_agreement"

VARIANT_LABELS: "OrderedDict[str, str]" = OrderedDict(
    [
        ("A0_unet2d_t2", "2D U-Net, hard T2"),
        ("A1_resunet2d_t2", "2D residual U-Net, hard T2"),
        ("A2_resunet2p5d_t2", "2.5D residual U-Net, hard T2"),
        ("A3_resunet2d_nested", "2D residual U-Net, ordered T1-T4"),
        ("A4_proposed", "Proposed 2.5D ordered agreement model"),
        ("A5_independent_heads", "2.5D agreement model, independent heads"),
        ("A6_transformer", "2.5D ordered agreement model + transformer"),
        ("A7_ordinal_rps", "Architecture-matched exact-count ORC-RPS"),
        ("A8_no_brier", "Proposed 2.5D ordered model without Brier term"),
    ]
)

TARGETS = ("T1", "T2", "T3", "T4")
T2_ONLY_VARIANTS = {
    "A0_unet2d_t2",
    "A1_resunet2d_t2",
    "A2_resunet2p5d_t2",
}
DEFAULT_FOLDS = (0, 1, 2, 3, 4)
EXPECTED_CASES = 325
EXPECTED_PATIENTS = 247
EXPECTED_T2_POSITIVE = 237

REQUIRED_CASE_COLUMNS = {
    "target",
    "volume_id",
    "patient_id",
    "outer_fold",
    "seed",
    "model",
    "target_positive",
    "prediction_positive",
    "dice",
    "iou",
    "precision",
    "recall",
    "hd95",
    "assd",
    "surface_defined",
}
REQUIRED_SOFT_ID_COLUMNS = {
    "volume_id",
    "patient_id",
    "outer_fold",
    "seed",
    "model",
}
REQUIRED_CALIBRATION_COLUMNS = {
    "model",
    "seed",
    "outer_fold",
    "scope",
    "bin_lower",
    "bin_upper",
    "mean_predicted",
    "mean_observed",
    "n_voxels",
}
BRIER_EXACT_COLUMNS = {
    "soft_brier",
    "soft_brier_foreground",
    "soft_brier_background",
    "soft_brier_balanced",
    "q_brier",
    "q_brier_foreground",
    "q_brier_background",
    "q_brier_balanced",
}
class CoverageError(RuntimeError):
    """Raised when requested runs or their out-of-fold panels are incomplete."""

    def __init__(self, message: str, report: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.report = dict(report or {})


def _parse_csv_list(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def _parse_int_list(value: str) -> list[int]:
    try:
        values = [int(item) for item in _parse_csv_list(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("duplicate integers are not allowed")
    return values


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_write_text(path, frame.to_csv(index=False, lineterminator="\n"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
    _atomic_write_text(path, text + "\n")


def _seed_from_parts(base_seed: int, *parts: str) -> int:
    material = "\0".join([str(base_seed), *map(str, parts)]).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _require_columns(frame: pd.DataFrame, required: set[str], context: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise CoverageError(
            f"{context}: missing required columns: {', '.join(missing)}",
            {"context": context, "missing_columns": missing},
        )


def _read_csv(path: Path, *, required: set[str] | None = None) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise CoverageError(f"could not read {path}: {exc}") from exc
    if frame.empty:
        raise CoverageError(f"{path} is empty")
    if required:
        _require_columns(frame, required, str(path))
    return frame


def _numeric(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    for column in columns:
        if column not in frame:
            continue
        original = frame[column]
        try:
            converted = pd.to_numeric(original, errors="raise")
        except Exception as exc:
            raise CoverageError(f"{context}: {column!r} must be numeric") from exc
        frame[column] = converted


def _coerce_bool(series: pd.Series, context: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    truth = {"true": True, "1": True, "yes": True, "y": True, "t": True}
    false = {"false": False, "0": False, "no": False, "n": False, "f": False}
    converted: list[bool] = []
    for value in series.tolist():
        if isinstance(value, (bool, np.bool_)):
            converted.append(bool(value))
            continue
        if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            converted.append(bool(value))
            continue
        if isinstance(value, (float, np.floating)) and np.isfinite(value) and value in (0.0, 1.0):
            converted.append(bool(int(value)))
            continue
        key = str(value).strip().lower()
        if key in truth:
            converted.append(True)
        elif key in false:
            converted.append(False)
        else:
            raise CoverageError(f"{context}: invalid boolean value {value!r}")
    return pd.Series(converted, index=series.index, dtype=bool)


def _normalise_ids(frame: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    for column in columns:
        if frame[column].isna().any():
            raise CoverageError(f"{context}: {column!r} contains missing values")
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise CoverageError(f"{context}: {column!r} contains blank values")


def _validate_run_identity(
    frame: pd.DataFrame,
    *,
    variant: str,
    seed: int,
    fold: int,
    context: str,
) -> None:
    allowed_models = {variant, VARIANT_LABELS[variant]}
    observed_models = set(frame["model"].astype(str).str.strip().unique())
    if len(observed_models) != 1 or not observed_models.issubset(allowed_models):
        raise CoverageError(
            f"{context}: model must identify {variant!r}; observed {sorted(observed_models)}"
        )
    _numeric(frame, ("seed", "outer_fold"), context)
    observed_seeds = set(frame["seed"].astype(int).unique())
    observed_folds = set(frame["outer_fold"].astype(int).unique())
    if observed_seeds != {seed}:
        raise CoverageError(f"{context}: expected seed {seed}, observed {sorted(observed_seeds)}")
    if observed_folds != {fold}:
        raise CoverageError(
            f"{context}: expected outer_fold {fold}, observed {sorted(observed_folds)}"
        )


def _brier_columns(frame: pd.DataFrame) -> list[str]:
    exact = [column for column in frame.columns if column in BRIER_EXACT_COLUMNS]
    prefixed = [
        column
        for column in frame.columns
        if column.startswith(("soft_brier_", "q_brier_")) and column not in exact
    ]
    return sorted(exact + prefixed)


def _nesting_column(frame: pd.DataFrame) -> str:
    candidates = [
        "nesting_violation_fraction",
        "nesting_violation_rate",
        "ordering_violation_fraction",
        "ordering_violation_rate",
    ]
    present = [column for column in candidates if column in frame]
    if len(present) != 1:
        raise CoverageError(
            "soft_case_metrics.csv must contain exactly one nesting/order violation "
            f"column from {candidates}; observed {present}"
        )
    return present[0]


def _contour_count_source(case: pd.DataFrame, soft: pd.DataFrame, context: str) -> None:
    column = "n_clustered_contours"
    if column not in case and column not in soft:
        raise CoverageError(
            f"{context}: {column!r} is required in case_metrics.csv or "
            "soft_case_metrics.csv for contour-count stratification"
        )
    for frame, name in ((case, "case_metrics.csv"), (soft, "soft_case_metrics.csv")):
        if column not in frame:
            continue
        _numeric(frame, (column,), f"{context}/{name}")
        values = frame[column].to_numpy(dtype=float)
        if (
            not np.isfinite(values).all()
            or not np.equal(values, np.floor(values)).all()
            or ((values < 1) | (values > 4)).any()
        ):
            raise CoverageError(f"{context}/{name}: {column} must be an integer in [1, 4]")


def _validate_case_metrics(
    frame: pd.DataFrame, *, variant: str, seed: int, fold: int, context: str
) -> pd.DataFrame:
    _require_columns(frame, REQUIRED_CASE_COLUMNS, context)
    frame = frame.copy()
    _normalise_ids(frame, ("volume_id", "patient_id", "target"), context)
    frame["target"] = frame["target"].str.upper()
    unknown_targets = sorted(set(frame["target"]).difference(TARGETS))
    if unknown_targets:
        raise CoverageError(f"{context}: unsupported targets {unknown_targets}; expected T1-T4")
    _validate_run_identity(frame, variant=variant, seed=seed, fold=fold, context=context)
    frame["target_positive"] = _coerce_bool(frame["target_positive"], context)
    frame["prediction_positive"] = _coerce_bool(frame["prediction_positive"], context)
    frame["surface_defined"] = _coerce_bool(frame["surface_defined"], context)
    metric_columns = ("dice", "iou", "precision", "recall", "hd95", "assd")
    _numeric(frame, metric_columns, context)
    for column in ("dice", "iou", "precision", "recall"):
        finite = frame[column].dropna().to_numpy(dtype=float)
        if ((finite < 0.0) | (finite > 1.0)).any():
            raise CoverageError(f"{context}: {column} must lie in [0, 1] when defined")
    for column in ("hd95", "assd"):
        finite = frame[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(finite).all() or (finite < 0.0).any():
            raise CoverageError(
                f"{context}: {column} must be finite and non-negative when defined"
            )
    positive = frame["target_positive"]
    if not np.isfinite(frame.loc[positive, "dice"].to_numpy(dtype=float)).all():
        raise CoverageError(f"{context}: target-positive rows require finite Dice")
    duplicated = frame.duplicated(["volume_id", "target"], keep=False)
    if duplicated.any():
        sample = frame.loc[duplicated, ["volume_id", "target"]].head(10).to_dict("records")
        raise CoverageError(f"{context}: duplicate volume/target rows: {sample}")
    mapping_counts = frame.groupby("volume_id", sort=False)["patient_id"].nunique()
    if (mapping_counts != 1).any():
        raise CoverageError(f"{context}: a volume maps to more than one patient")
    return frame


def _validate_soft_metrics(
    frame: pd.DataFrame, *, variant: str, seed: int, fold: int, context: str
) -> tuple[pd.DataFrame, list[str], str]:
    _require_columns(frame, REQUIRED_SOFT_ID_COLUMNS, context)
    frame = frame.copy()
    _normalise_ids(frame, ("volume_id", "patient_id"), context)
    _validate_run_identity(frame, variant=variant, seed=seed, fold=fold, context=context)
    brier = _brier_columns(frame)
    if not brier:
        raise CoverageError(
            f"{context}: no q/soft Brier column found; expected one of "
            f"{sorted(BRIER_EXACT_COLUMNS)} or a q_brier_*/soft_brier_* column"
        )
    nesting = _nesting_column(frame)
    _numeric(frame, [*brier, nesting], context)
    for column in [*brier, nesting]:
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
            raise CoverageError(f"{context}: {column} must be finite and in [0, 1]")
    duplicated = frame.duplicated(["volume_id"], keep=False)
    if duplicated.any():
        sample = frame.loc[duplicated, ["volume_id"]].head(10).to_dict("records")
        raise CoverageError(f"{context}: duplicate volume rows: {sample}")
    return frame, brier, nesting


def _validate_calibration(
    frame: pd.DataFrame, *, variant: str, seed: int, fold: int, context: str
) -> pd.DataFrame:
    _require_columns(frame, REQUIRED_CALIBRATION_COLUMNS, context)
    frame = frame.copy()
    _normalise_ids(frame, ("scope",), context)
    _validate_run_identity(frame, variant=variant, seed=seed, fold=fold, context=context)
    numeric = (
        "bin_lower",
        "bin_upper",
        "mean_predicted",
        "mean_observed",
        "n_voxels",
    )
    _numeric(frame, numeric, context)
    for column in ("bin_lower", "bin_upper"):
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
            raise CoverageError(f"{context}: {column} must be finite and in [0, 1]")
    if (frame["bin_lower"] >= frame["bin_upper"]).any():
        raise CoverageError(f"{context}: every calibration bin requires lower < upper")
    counts = frame["n_voxels"].to_numpy(dtype=float)
    if (
        not np.isfinite(counts).all()
        or not np.equal(counts, np.floor(counts)).all()
        or (counts < 0).any()
    ):
        raise CoverageError(f"{context}: n_voxels must contain non-negative integers")
    occupied = frame["n_voxels"] > 0
    for column in ("mean_predicted", "mean_observed"):
        occupied_values = frame.loc[occupied, column].to_numpy(dtype=float)
        if (
            not np.isfinite(occupied_values).all()
            or ((occupied_values < 0.0) | (occupied_values > 1.0)).any()
        ):
            raise CoverageError(
                f"{context}: occupied-bin {column} values must be finite and in [0, 1]"
            )
        empty_values = frame.loc[~occupied, column]
        if empty_values.notna().any():
            raise CoverageError(
                f"{context}: empty calibration bins must have NaN {column} values"
            )
    duplicated = frame.duplicated(["scope", "bin_lower", "bin_upper"], keep=False)
    if duplicated.any():
        sample = frame.loc[
            duplicated, ["scope", "bin_lower", "bin_upper"]
        ].head(10).to_dict("records")
        raise CoverageError(f"{context}: duplicate calibration bins: {sample}")
    return frame


def _read_summary(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CoverageError(f"could not parse {path}: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        raise CoverageError(f"{path} must contain a non-empty JSON object")
    return payload


def _discover_seeds(results_root: Path, variants: Sequence[str]) -> list[int]:
    seeds: set[int] = set()
    pattern = re.compile(r"seed_(\d+)$")
    for variant in variants:
        variant_dir = results_root / variant
        if not variant_dir.is_dir():
            continue
        for path in variant_dir.iterdir():
            match = pattern.fullmatch(path.name)
            if path.is_dir() and match:
                seeds.add(int(match.group(1)))
    if not seeds:
        raise CoverageError(
            f"no seed_* directories found for requested variants under {results_root}"
        )
    return sorted(seeds)


def _coverage_scan(
    results_root: Path,
    variants: Sequence[str],
    seeds: Sequence[int],
    folds: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    common_required_files = (
        "config.json",
        "run_summary.json",
        "best.pt",
        "case_metrics.csv",
        "target_summary.csv",
        "bootstrap_cis.csv",
        "history.csv",
    )
    agreement_required_files = ("soft_case_metrics.csv", "calibration.csv")
    summary_candidates = ("evaluation_summary.json", "summary.json")
    runs: list[dict[str, Any]] = []
    missing_runs: list[dict[str, Any]] = []
    incomplete_runs: list[dict[str, Any]] = []
    for variant in variants:
        for seed in seeds:
            for fold in folds:
                run_dir = results_root / variant / f"seed_{seed}" / f"fold_{fold}"
                record = {
                    "variant": variant,
                    "seed": seed,
                    "outer_fold": fold,
                    "run_dir": str(run_dir),
                }
                if not run_dir.is_dir():
                    missing_runs.append(record)
                    continue
                required_files = list(common_required_files)
                if variant not in T2_ONLY_VARIANTS:
                    required_files.extend(agreement_required_files)
                missing_files = [name for name in required_files if not (run_dir / name).is_file()]
                summary_path = next(
                    (run_dir / name for name in summary_candidates if (run_dir / name).is_file()),
                    None,
                )
                if summary_path is None:
                    missing_files.append("evaluation_summary.json|summary.json")
                if missing_files:
                    incomplete_runs.append({**record, "missing_files": missing_files})
                    continue
                bootstrap_path = run_dir / "bootstrap_cis.csv"
                runs.append(
                    {
                        **record,
                        "path": run_dir,
                        "bootstrap_path": bootstrap_path,
                        "summary_path": summary_path,
                    }
                )
    expected = len(variants) * len(seeds) * len(folds)
    report = {
        "status": "COMPLETE" if not missing_runs and not incomplete_runs else "INCOMPLETE",
        "checked_at_utc": _iso_now(),
        "results_root": str(results_root),
        "requested_variants": list(variants),
        "requested_seeds": list(seeds),
        "requested_folds": list(folds),
        "expected_run_count": expected,
        "complete_run_count": len(runs),
        "missing_runs": missing_runs,
        "incomplete_runs": incomplete_runs,
        "common_required_files_per_run": list(common_required_files),
        "agreement_variant_additional_files": list(agreement_required_files),
        "summary_file_alternatives": list(summary_candidates),
        "bootstrap_file": "bootstrap_cis.csv",
    }
    if report["status"] != "COMPLETE":
        raise CoverageError(
            f"run coverage incomplete: {len(runs)}/{expected} requested runs are complete",
            report,
        )
    return runs, report


def _attach_contour_counts(
    case: pd.DataFrame, soft: pd.DataFrame, context: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    column = "n_clustered_contours"
    _contour_count_source(case, soft, context)
    if column in case:
        per_volume = case.groupby("volume_id", sort=False)[column].nunique()
        if (per_volume != 1).any():
            raise CoverageError(f"{context}: contour count changes across targets for a volume")
    if column in case and column in soft:
        left = case[["volume_id", column]].drop_duplicates("volume_id")
        right = soft[["volume_id", column]]
        check = left.merge(right, on="volume_id", suffixes=("_case", "_soft"), validate="one_to_one")
        if len(check) != len(right) or not np.array_equal(
            check[f"{column}_case"].to_numpy(dtype=int),
            check[f"{column}_soft"].to_numpy(dtype=int),
        ):
            raise CoverageError(f"{context}: contour counts disagree between case and soft metrics")
    elif column in soft:
        case = case.merge(
            soft[["volume_id", column]], on="volume_id", how="left", validate="many_to_one"
        )
        if case[column].isna().any():
            raise CoverageError(f"{context}: soft metrics lack contour counts for some cases")
    else:
        mapping = case[["volume_id", column]].drop_duplicates("volume_id")
        soft = soft.merge(mapping, on="volume_id", how="left", validate="one_to_one")
        if soft[column].isna().any():
            raise CoverageError(f"{context}: case metrics lack contour counts for some cases")
    case[column] = case[column].astype(int)
    soft[column] = soft[column].astype(int)
    return case, soft


def _load_runs(
    run_records: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    case_frames: list[pd.DataFrame] = []
    soft_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    canonical_brier: set[str] | None = None
    canonical_nesting: str | None = None
    canonical_calibration_grid: set[tuple[str, float, float]] | None = None
    canonical_training_code: str | None = None
    canonical_evaluation_code: str | None = None
    canonical_dataset_content: str | None = None
    canonical_manifest: str | None = None

    for record in run_records:
        variant = str(record["variant"])
        seed = int(record["seed"])
        fold = int(record["outer_fold"])
        run_dir = Path(record["path"])
        context = f"{variant}/seed_{seed}/fold_{fold}"

        case = _validate_case_metrics(
            _read_csv(run_dir / "case_metrics.csv"),
            variant=variant,
            seed=seed,
            fold=fold,
            context=f"{context}/case_metrics.csv",
        )
        target_summary = _read_csv(run_dir / "target_summary.csv")
        _require_columns(
            target_summary,
            {"model", "seed", "outer_fold", "target", "n_target_positive_cases", "patient_macro_dice"},
            f"{context}/target_summary.csv",
        )
        _validate_run_identity(
            target_summary,
            variant=variant,
            seed=seed,
            fold=fold,
            context=f"{context}/target_summary.csv",
        )
        bootstrap_frame = _read_csv(Path(record["bootstrap_path"]))
        _require_columns(
            bootstrap_frame,
            {
                "model",
                "seed",
                "outer_fold",
                "target",
                "scope",
                "metric",
                "estimate",
                "lower",
                "upper",
                "n_patients",
                "n_bootstrap",
                "confidence",
            },
            f"{context}/bootstrap_cis.csv",
        )
        _validate_run_identity(
            bootstrap_frame,
            variant=variant,
            seed=seed,
            fold=fold,
            context=f"{context}/bootstrap_cis.csv",
        )
        _numeric(
            bootstrap_frame,
            ("n_bootstrap", "confidence"),
            f"{context}/bootstrap_cis.csv",
        )
        if (bootstrap_frame["n_bootstrap"] < 1000).any():
            raise CoverageError(
                f"{context}/bootstrap_cis.csv: final runs require at least 1000 replicates"
            )
        if not np.allclose(bootstrap_frame["confidence"], 0.95):
            raise CoverageError(
                f"{context}/bootstrap_cis.csv: final run confidence must equal 0.95"
            )
        summary = _read_summary(Path(record["summary_path"]))
        status = str(summary.get("status", "")).upper()
        if status not in {"EVALUATION_COMPLETE", "COMPLETE", "PASS"}:
            raise CoverageError(
                f"{context}/{Path(record['summary_path']).name}: non-complete status {status!r}"
            )
        expected_summary = {
            "model": variant,
            "seed": seed,
            "outer_fold": fold,
            "n_outer_test_cases": int(case["volume_id"].nunique()),
            "n_outer_test_patients": int(case["patient_id"].nunique()),
            "n_t2_positive_cases": int(
                case.loc[
                    (case["target"] == "T2") & case["target_positive"], "volume_id"
                ].nunique()
            ),
        }
        for key, expected_value in expected_summary.items():
            if summary.get(key) != expected_value:
                raise CoverageError(
                    f"{context}/{Path(record['summary_path']).name}: {key}="
                    f"{summary.get(key)!r}, expected {expected_value!r}"
                )
        config = _read_summary(run_dir / "config.json")
        run_summary = _read_summary(run_dir / "run_summary.json")
        expected_config_identity = {
            "variant": variant,
            "seed": seed,
            "outer_fold": fold,
            "smoke": False,
        }
        for key, expected_value in expected_config_identity.items():
            if config.get(key) != expected_value:
                raise CoverageError(
                    f"{context}/config.json: {key}={config.get(key)!r}, "
                    f"expected {expected_value!r}"
                )
        if run_summary.get("status") != "TRAINING_COMPLETE":
            raise CoverageError(
                f"{context}/run_summary.json: expected TRAINING_COMPLETE status"
            )
        checkpoint_sha256 = _sha256(run_dir / "best.pt")
        if (
            run_summary.get("checkpoint_sha256") != checkpoint_sha256
            or summary.get("checkpoint_sha256") != checkpoint_sha256
        ):
            raise CoverageError(
                f"{context}: checkpoint hash does not match training/evaluation provenance"
            )
        if summary.get("surface_metrics_available") is not True:
            raise CoverageError(
                f"{context}: prespecified surface metrics were not available"
            )
        try:
            selected_threshold = float(summary["threshold_selected_on_validation"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CoverageError(
                f"{context}: invalid validation-selected threshold"
            ) from exc
        if not np.isfinite(selected_threshold) or not 0.0 <= selected_threshold <= 1.0:
            raise CoverageError(
                f"{context}: validation-selected threshold must be finite and in [0, 1]"
            )
        provenance_fields = {
            "training_code_sha256": config.get("training_code_sha256"),
            "evaluation_code_sha256": summary.get("evaluation_code_sha256"),
            "dataset_content_sha256": config.get("dataset_content_sha256"),
            "manifest_sha256": config.get("manifest_sha256"),
        }
        if any(not isinstance(value, str) or len(value) != 64 for value in provenance_fields.values()):
            raise CoverageError(f"{context}: missing or malformed SHA-256 provenance")
        if canonical_training_code is None:
            canonical_training_code = str(provenance_fields["training_code_sha256"])
            canonical_evaluation_code = str(provenance_fields["evaluation_code_sha256"])
            canonical_dataset_content = str(provenance_fields["dataset_content_sha256"])
            canonical_manifest = str(provenance_fields["manifest_sha256"])
        elif (
            provenance_fields["training_code_sha256"] != canonical_training_code
            or provenance_fields["evaluation_code_sha256"] != canonical_evaluation_code
            or provenance_fields["dataset_content_sha256"] != canonical_dataset_content
            or provenance_fields["manifest_sha256"] != canonical_manifest
        ):
            raise CoverageError(
                f"{context}: code or dataset provenance differs from another final run"
            )
        history = _read_csv(run_dir / "history.csv")
        _require_columns(history, {"epoch", "train_loss", "val_loss"}, f"{context}/history.csv")
        _numeric(history, ("epoch", "train_loss", "val_loss"), f"{context}/history.csv")
        if not np.isfinite(history[["epoch", "train_loss", "val_loss"]].to_numpy(dtype=float)).all():
            raise CoverageError(f"{context}/history.csv: epoch and losses must be finite")

        if variant in T2_ONLY_VARIANTS:
            if "n_clustered_contours" not in case:
                raise CoverageError(
                    f"{context}/case_metrics.csv: n_clustered_contours is required because "
                    "hard-T2 runs intentionally have no soft metrics"
                )
            _contour_count_source(case, pd.DataFrame(), context)
            case["n_clustered_contours"] = case["n_clustered_contours"].astype(int)
            soft = None
            calibration = None
            brier = []
            nesting = None
        else:
            soft, brier, nesting = _validate_soft_metrics(
                _read_csv(run_dir / "soft_case_metrics.csv"),
                variant=variant,
                seed=seed,
                fold=fold,
                context=f"{context}/soft_case_metrics.csv",
            )
            calibration = _validate_calibration(
                _read_csv(run_dir / "calibration.csv"),
                variant=variant,
                seed=seed,
                fold=fold,
                context=f"{context}/calibration.csv",
            )
            calibration_grid = set(
                calibration[["scope", "bin_lower", "bin_upper"]].itertuples(
                    index=False, name=None
                )
            )
            if canonical_calibration_grid is None:
                canonical_calibration_grid = calibration_grid
            elif canonical_calibration_grid != calibration_grid:
                raise CoverageError(
                    f"{context}: calibration scope/bin grid differs from other agreement runs"
                )
            case, soft = _attach_contour_counts(case, soft, context)
            case_volumes = set(case["volume_id"])
            soft_volumes = set(soft["volume_id"])
            if case_volumes != soft_volumes:
                raise CoverageError(
                    f"{context}: case/soft volume sets differ "
                    f"({len(case_volumes)} versus {len(soft_volumes)})"
                )
            patient_check = case[["volume_id", "patient_id"]].drop_duplicates().merge(
                soft[["volume_id", "patient_id"]],
                on="volume_id",
                suffixes=("_case", "_soft"),
                validate="one_to_one",
            )
            if not (patient_check["patient_id_case"] == patient_check["patient_id_soft"]).all():
                raise CoverageError(f"{context}: patient IDs disagree between case and soft metrics")

            brier_set = set(brier)
            if canonical_brier is None:
                canonical_brier = brier_set
            elif canonical_brier != brier_set:
                raise CoverageError(
                    f"{context}: inconsistent Brier columns; expected {sorted(canonical_brier)}, "
                    f"observed {sorted(brier_set)}"
                )
            if canonical_nesting is None:
                canonical_nesting = nesting
            elif canonical_nesting != nesting:
                raise CoverageError(
                    f"{context}: inconsistent nesting column; expected {canonical_nesting}, "
                    f"observed {nesting}"
                )

        case["variant"] = variant
        for frame in (case, soft, calibration):
            if frame is None:
                continue
            frame["variant"] = variant
            frame["model"] = VARIANT_LABELS[variant]
            frame["seed"] = seed
            frame["outer_fold"] = fold
        history = history.copy()
        history["variant"] = variant
        history["model"] = VARIANT_LABELS[variant]
        history["seed"] = seed
        history["outer_fold"] = fold

        case_frames.append(case)
        if soft is not None:
            soft_frames.append(soft)
        if calibration is not None:
            calibration_frames.append(calibration)
        history_frames.append(history)
        metadata.append(
            {
                "variant": variant,
                "seed": seed,
                "outer_fold": fold,
                "case_rows": int(len(case)),
                "soft_case_rows": 0 if soft is None else int(len(soft)),
                "calibration_rows": 0 if calibration is None else int(len(calibration)),
                "target_summary_rows": int(len(target_summary)),
                "bootstrap_source": str(record["bootstrap_path"]),
                "bootstrap_rows": int(len(bootstrap_frame)),
                "summary_source": str(record["summary_path"]),
                "summary_keys": sorted(map(str, summary.keys())),
            }
        )

    return (
        pd.concat(case_frames, ignore_index=True),
        pd.concat(soft_frames, ignore_index=True) if soft_frames else pd.DataFrame(),
        pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame(),
        pd.concat(history_frames, ignore_index=True),
        metadata,
    )


def _expected_targets(variant: str, observed: set[str]) -> set[str]:
    """Accept T2-only baselines or a complete four-head evaluator panel.

    T2 is mandatory for every variant.  Any evaluator that emits an agreement
    target beyond T2 must emit all four targets.  Ordered/agreement variants are
    required to emit all four.
    """
    if "T2" not in observed:
        raise CoverageError(f"{variant}: required target T2 is absent")
    if variant not in T2_ONLY_VARIANTS or observed != {"T2"}:
        return set(TARGETS)
    return {"T2"}


def _validate_oof_panels(
    case: pd.DataFrame,
    soft: pd.DataFrame,
    *,
    variants: Sequence[str],
    seeds: Sequence[int],
    expected_cases: int,
    expected_patients: int,
    expected_t2_positive: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {"panels": []}
    t2_truth_reference: pd.DataFrame | None = None
    multitier_truth_reference: pd.DataFrame | None = None

    for variant in variants:
        for seed in seeds:
            subset = case[(case["variant"] == variant) & (case["seed"] == seed)].copy()
            soft_subset = (
                soft[(soft["variant"] == variant) & (soft["seed"] == seed)].copy()
                if not soft.empty
                else pd.DataFrame()
            )
            observed_targets = set(subset["target"])
            required_targets = _expected_targets(variant, observed_targets)
            if observed_targets != required_targets:
                raise CoverageError(
                    f"{variant}/seed_{seed}: expected targets {sorted(required_targets)}, "
                    f"observed {sorted(observed_targets)}"
                )
            duplicated = subset.duplicated(["volume_id", "target"], keep=False)
            if duplicated.any():
                rows = subset.loc[
                    duplicated, ["volume_id", "target", "outer_fold"]
                ].head(20).to_dict("records")
                raise CoverageError(
                    f"{variant}/seed_{seed}: duplicate out-of-fold predictions: {rows}"
                )
            expected_rows = expected_cases * len(required_targets)
            if len(subset) != expected_rows:
                raise CoverageError(
                    f"{variant}/seed_{seed}: expected {expected_rows} case-target rows, "
                    f"observed {len(subset)}"
                )
            counts = subset.groupby("target")["volume_id"].nunique().to_dict()
            bad_counts = {target: counts.get(target, 0) for target in required_targets if counts.get(target, 0) != expected_cases}
            if bad_counts:
                raise CoverageError(
                    f"{variant}/seed_{seed}: incomplete per-target case coverage: {bad_counts}"
                )
            volumes = subset[["volume_id", "patient_id", "outer_fold", "n_clustered_contours"]].drop_duplicates()
            if volumes["volume_id"].nunique() != expected_cases or len(volumes) != expected_cases:
                raise CoverageError(
                    f"{variant}/seed_{seed}: expected exactly {expected_cases} unique OOF volumes"
                )
            if volumes["patient_id"].nunique() != expected_patients:
                raise CoverageError(
                    f"{variant}/seed_{seed}: expected {expected_patients} patients, "
                    f"observed {volumes['patient_id'].nunique()}"
                )
            if volumes.groupby("volume_id")["outer_fold"].nunique().max() != 1:
                raise CoverageError(f"{variant}/seed_{seed}: a volume occurs in multiple folds")
            fold_counts = {
                int(key): int(value)
                for key, value in volumes.groupby("outer_fold")["volume_id"].nunique().items()
            }
            expected_fold_counts = {fold: expected_cases // len(DEFAULT_FOLDS) for fold in DEFAULT_FOLDS}
            if fold_counts != expected_fold_counts:
                raise CoverageError(
                    f"{variant}/seed_{seed}: expected 65 OOF cases in every fold; "
                    f"observed {fold_counts}"
                )
            t2 = subset[subset["target"] == "T2"]
            t2_positive = int(t2["target_positive"].sum())
            if t2_positive != expected_t2_positive:
                raise CoverageError(
                    f"{variant}/seed_{seed}: expected {expected_t2_positive} target-positive "
                    f"T2 cases, observed {t2_positive}"
                )
            if variant not in T2_ONLY_VARIANTS:
                if (
                    len(soft_subset) != expected_cases
                    or soft_subset["volume_id"].nunique() != expected_cases
                ):
                    raise CoverageError(
                        f"{variant}/seed_{seed}: soft metrics require exactly one row for each "
                        f"of {expected_cases} OOF volumes"
                    )
                if set(soft_subset["volume_id"]) != set(volumes["volume_id"]):
                    raise CoverageError(
                        f"{variant}/seed_{seed}: soft and case OOF volume sets differ"
                    )

            truth = subset[
                [
                    "volume_id",
                    "patient_id",
                    "outer_fold",
                    "target",
                    "target_positive",
                    "n_clustered_contours",
                ]
            ].sort_values(["volume_id", "target"]).reset_index(drop=True)
            t2_truth = truth[truth["target"] == "T2"].reset_index(drop=True)
            if t2_truth_reference is None:
                t2_truth_reference = t2_truth
            elif not t2_truth_reference.equals(t2_truth):
                raise CoverageError(
                    f"{variant}/seed_{seed}: T2 truth/patient/contour mapping differs from "
                    "another complete OOF panel"
                )
            if required_targets == set(TARGETS) and multitier_truth_reference is None:
                multitier_truth_reference = truth
            elif required_targets == set(TARGETS):
                if not multitier_truth_reference.equals(truth):
                    raise CoverageError(
                        f"{variant}/seed_{seed}: T1-T4 truth/patient/contour mapping differs "
                        "from another complete OOF panel"
                    )
            else:
                assert required_targets == {"T2"}

            report["panels"].append(
                {
                    "variant": variant,
                    "seed": seed,
                    "targets": sorted(required_targets),
                    "n_case_target_rows": int(len(subset)),
                    "n_cases": int(volumes["volume_id"].nunique()),
                    "n_patients": int(volumes["patient_id"].nunique()),
                    "n_t2_target_positive": t2_positive,
                    "fold_case_counts": {str(key): value for key, value in fold_counts.items()},
                }
            )

    report["status"] = "PASS"
    return report


def _validate_prediction_npz(path: Path) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            keys = set(payload.files)
            if "probabilities" not in keys:
                raise CoverageError(
                    f"{path}: proposed-model prediction requires a probabilities array"
                )
            probabilities = np.asarray(payload["probabilities"])
            if probabilities.ndim != 4 or probabilities.shape[0] != 4:
                raise CoverageError(
                    f"{path}: probabilities must have shape [4,D,H,W], observed "
                    f"{probabilities.shape}"
                )
            if (
                not np.isfinite(probabilities).all()
                or float(probabilities.min()) < 0.0
                or float(probabilities.max()) > 1.0
            ):
                raise CoverageError(f"{path}: probabilities must be finite and in [0, 1]")
            if "q_hat" in keys:
                q_hat = np.asarray(payload["q_hat"])
                if q_hat.shape != probabilities.shape[1:]:
                    raise CoverageError(
                        f"{path}: q_hat shape {q_hat.shape} does not match probability volume "
                        f"{probabilities.shape[1:]}"
                    )
                if (
                    not np.isfinite(q_hat).all()
                    or float(q_hat.min()) < 0.0
                    or float(q_hat.max()) > 1.0
                ):
                    raise CoverageError(f"{path}: q_hat must be finite and in [0, 1]")
            return {
                "shape": [int(value) for value in probabilities.shape],
                "keys": sorted(keys),
            }
    except CoverageError:
        raise
    except Exception as exc:
        raise CoverageError(f"could not validate prediction archive {path}: {exc}") from exc


def _prediction_sources(
    run_records: Sequence[Mapping[str, Any]],
    case: pd.DataFrame,
    *,
    variant: str,
    seed: int,
    folds: Sequence[int],
    expected_cases: int,
) -> tuple[list[tuple[str, Path]], dict[str, Any]]:
    record_by_fold = {
        int(record["outer_fold"]): record
        for record in run_records
        if record["variant"] == variant and int(record["seed"]) == seed
    }
    missing_folds = sorted(set(folds).difference(record_by_fold))
    if missing_folds:
        raise CoverageError(
            f"qualitative prediction source {variant}/seed_{seed} lacks folds {missing_folds}"
        )

    sources: list[tuple[str, Path]] = []
    seen: dict[str, int] = {}
    fold_counts: dict[str, int] = {}
    shapes: dict[str, int] = {}
    for fold in folds:
        run_dir = Path(record_by_fold[fold]["path"])
        predictions_dir = run_dir / "predictions"
        if not predictions_dir.is_dir():
            raise CoverageError(f"missing proposed-model prediction directory: {predictions_dir}")
        files = sorted(predictions_dir.glob("*.npz"))
        actual = {path.stem: path for path in files}
        if len(actual) != len(files):
            raise CoverageError(f"duplicate prediction file stems in {predictions_dir}")
        expected = set(
            case[
                (case["variant"] == variant)
                & (case["seed"] == seed)
                & (case["outer_fold"] == fold)
            ]["volume_id"].astype(str)
        )
        actual_ids = set(actual)
        if actual_ids != expected:
            missing = sorted(expected.difference(actual_ids))
            extra = sorted(actual_ids.difference(expected))
            raise CoverageError(
                f"{predictions_dir}: prediction/case ID mismatch; missing={missing[:20]}, "
                f"extra={extra[:20]}"
            )
        for case_id in sorted(actual):
            if case_id in seen:
                raise CoverageError(
                    f"prediction {case_id!r} appears in both fold {seen[case_id]} and fold {fold}"
                )
            seen[case_id] = fold
            archive = _validate_prediction_npz(actual[case_id])
            shape_key = "x".join(map(str, archive["shape"]))
            shapes[shape_key] = shapes.get(shape_key, 0) + 1
            sources.append((case_id, actual[case_id]))
        fold_counts[str(fold)] = len(actual)
    if len(sources) != expected_cases or len(seen) != expected_cases:
        raise CoverageError(
            f"qualitative prediction assembly requires {expected_cases} unique cases; "
            f"observed {len(seen)}"
        )
    report = {
        "status": "PASS",
        "source_variant": variant,
        "source_model": VARIANT_LABELS[variant],
        "source_seed": seed,
        "source_folds": list(folds),
        "n_unique_cases": len(seen),
        "fold_case_counts": fold_counts,
        "probability_shapes": shapes,
    }
    return sorted(sources), report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assemble_predictions(
    sources: Sequence[tuple[str, Path]], output_root: Path
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "predictions"
    stage = Path(tempfile.mkdtemp(prefix=".predictions.", dir=output_root))
    try:
        for case_id, source in sources:
            shutil.copy2(source, stage / f"{case_id}.npz")
        staged = {path.stem: path for path in stage.glob("*.npz")}
        expected = {case_id: source for case_id, source in sources}
        if set(staged) != set(expected):
            raise CoverageError("staged qualitative prediction set failed its case-ID check")
        if destination.exists():
            if not destination.is_dir():
                raise CoverageError(f"prediction destination is not a directory: {destination}")
            current_paths = sorted(destination.glob("*.npz"))
            current = {path.stem: path for path in current_paths}
            if len(current) != len(current_paths) or set(current) != set(expected):
                raise CoverageError(
                    f"existing {destination} is not the verified {len(expected)}-case set; "
                    "move it aside before rerunning"
                )
            changed = [
                case_id
                for case_id, source in expected.items()
                if _sha256(source) != _sha256(current[case_id])
            ]
            if changed:
                raise CoverageError(
                    f"existing {destination} differs from verified OOF sources for cases "
                    f"{changed[:20]}; move it aside before rerunning"
                )
            return {
                "destination": str(destination),
                "n_cases": len(expected),
                "action": "verified_existing",
            }
        os.replace(stage, destination)
        return {
            "destination": str(destination),
            "n_cases": len(expected),
            "action": "copied",
        }
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _case_mean_across_seeds(
    frame: pd.DataFrame,
    metric: str,
    *,
    extra_keys: Sequence[str] = (),
) -> pd.DataFrame:
    keys = ["volume_id", "patient_id", *extra_keys]
    consistency = frame.groupby("volume_id", sort=False)["patient_id"].nunique()
    if (consistency != 1).any():
        raise CoverageError("a volume maps to multiple patients during seed aggregation")
    grouped = frame.groupby(keys, as_index=False, observed=True)[metric].mean()
    if not np.isfinite(grouped[metric].to_numpy(dtype=float)).all():
        raise CoverageError(f"non-finite {metric} survived case-level seed aggregation")
    return grouped


def _patient_values(case_means: pd.DataFrame, metric: str) -> pd.Series:
    values = case_means.groupby("patient_id", sort=True)[metric].mean()
    if values.empty or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise CoverageError(f"cannot form finite patient-level values for {metric}")
    return values


def _bootstrap_mean_ci(
    patient_values: pd.Series,
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, float]:
    values = patient_values.to_numpy(dtype=float)
    estimate = float(values.mean())
    rng = np.random.default_rng(seed)
    n_patients = len(values)
    draws = np.empty(replicates, dtype=float)
    chunk_size = max(1, min(replicates, 2048))
    for start in range(0, replicates, chunk_size):
        count = min(chunk_size, replicates - start)
        indices = rng.integers(0, n_patients, size=(count, n_patients))
        draws[start : start + count] = values[indices].mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    # Downstream figure contracts require the reported point estimate to be
    # enclosed even in a rare finite-bootstrap/skewed-distribution edge case.
    return estimate, min(float(low), estimate), max(float(high), estimate)


def _metric_summary(
    frame: pd.DataFrame,
    metric: str,
    *,
    replicates: int,
    confidence: float,
    bootstrap_seed: int,
    seed_parts: Sequence[str],
    extra_keys: Sequence[str] = (),
) -> tuple[dict[str, Any], pd.Series]:
    case_means = _case_mean_across_seeds(frame, metric, extra_keys=extra_keys)
    patient_values = _patient_values(case_means, metric)
    estimate, low, high = _bootstrap_mean_ci(
        patient_values,
        replicates=replicates,
        confidence=confidence,
        seed=_seed_from_parts(bootstrap_seed, *seed_parts),
    )
    summary = {
        "estimate": estimate,
        "ci95_low": low,
        "ci95_high": high,
        "n_cases": int(case_means["volume_id"].nunique()),
        "n_patients": int(len(patient_values)),
        "n_seed_predictions_per_case_min": int(frame.groupby("volume_id")["seed"].nunique().min()),
        "n_seed_predictions_per_case_max": int(frame.groupby("volume_id")["seed"].nunique().max()),
    }
    return summary, patient_values


def _aggregate_calibration_all_scopes(calibration: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["variant", "model", "scope", "bin_lower", "bin_upper"]
    for key, group in calibration.groupby(keys, sort=True, observed=True):
        group = group[group["n_voxels"] > 0]
        if group.empty:
            continue
        counts = group["n_voxels"].to_numpy(dtype=float)
        total = float(counts.sum())
        if total <= 0:
            raise CoverageError(f"calibration group {key} has no voxels")
        rows.append(
            {
                "variant": key[0],
                "model": key[1],
                "scope": key[2],
                "bin_lower": float(key[3]),
                "bin_upper": float(key[4]),
                "mean_predicted": float(np.average(group["mean_predicted"], weights=counts)),
                "mean_observed": float(np.average(group["mean_observed"], weights=counts)),
                "n_voxels": int(total),
                "n_seeds": int(group["seed"].nunique()),
                "n_run_folds": int(group[["seed", "outer_fold"]].drop_duplicates().shape[0]),
            }
        )
    return pd.DataFrame(rows)


def _calibration_summary(frame: pd.DataFrame) -> dict[str, Any]:
    counts = frame["n_voxels"].to_numpy(dtype=float)
    errors = np.abs(
        frame["mean_predicted"].to_numpy(dtype=float)
        - frame["mean_observed"].to_numpy(dtype=float)
    )
    return {
        "ece": float(np.average(errors, weights=counts)),
        "n_voxel_predictions": int(counts.sum()),
        "n_nonempty_bins": int(len(frame)),
    }


def _select_figure_calibration_scope(
    aggregated: pd.DataFrame, requested_scope: str
) -> tuple[pd.DataFrame, str]:
    scopes = sorted(aggregated["scope"].unique())
    if requested_scope != "auto":
        if requested_scope not in scopes:
            raise CoverageError(
                f"requested calibration scope {requested_scope!r} is absent; available: {scopes}"
            )
        selected = requested_scope
    elif len(scopes) == 1:
        selected = scopes[0]
    else:
        preferred = (
            "analysis_region",
            "agreement_analysis_region",
            "t1_union",
            "t1_region",
            "union_region",
            "peri_union",
            "full_roi",
        )
        selected = next((scope for scope in preferred if scope in scopes), "")
        if not selected:
            raise CoverageError(
                "multiple calibration scopes are available and none matches the automatic "
                f"priority list; pass --calibration-scope. Available: {scopes}"
            )
    output = aggregated[aggregated["scope"] == selected].copy()
    return output, selected


def _aggregate_results(
    case: pd.DataFrame,
    soft: pd.DataFrame,
    calibration: pd.DataFrame,
    history: pd.DataFrame,
    *,
    variants: Sequence[str],
    seeds: Sequence[int],
    baseline_variant: str,
    replicates: int,
    confidence: float,
    bootstrap_seed: int,
    calibration_scope: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    ablation_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    presence_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    model_summaries: dict[str, Any] = {}
    t2_patient_values: dict[str, pd.Series] = {}

    brier_columns = _brier_columns(soft)
    nesting_column = _nesting_column(soft)

    for order, variant in enumerate(variants):
        label = VARIANT_LABELS[variant]
        model_case = case[case["variant"] == variant]
        model_soft = soft[soft["variant"] == variant]
        primary = model_case[
            (model_case["target"] == "T2") & model_case["target_positive"]
        ]
        t2_summary, t2_values = _metric_summary(
            primary,
            "dice",
            replicates=replicates,
            confidence=confidence,
            bootstrap_seed=bootstrap_seed,
            seed_parts=(variant, "T2", "dice"),
        )
        t2_patient_values[variant] = t2_values
        ablation_rows.append(
            {
                "variant": variant,
                "model": label,
                "order": order,
                "is_proposed": variant == "A4_proposed",
                "patient_macro_t2_dice": t2_summary["estimate"],
                "ci95_low": t2_summary["ci95_low"],
                "ci95_high": t2_summary["ci95_high"],
                "n_target_positive_cases": t2_summary["n_cases"],
                "n_patients": t2_summary["n_patients"],
                "n_seeds": len(seeds),
            }
        )

        target_summaries: dict[str, Any] = {}
        observed_targets = set(model_case["target"])
        for target in TARGETS:
            if target not in observed_targets:
                continue
            target_positive = model_case[
                (model_case["target"] == target) & model_case["target_positive"]
            ]
            metric_summaries: dict[str, Any] = {}
            for metric in ("dice", "iou", "precision", "recall"):
                summary, _ = _metric_summary(
                    target_positive,
                    metric,
                    replicates=replicates,
                    confidence=confidence,
                    bootstrap_seed=bootstrap_seed,
                    seed_parts=(variant, target, metric),
                )
                metric_summaries[metric] = summary
                detailed_rows.append(
                    {
                        "variant": variant,
                        "model": label,
                        "target": target,
                        "scope": "target_positive",
                        "metric": metric,
                        "estimate": summary["estimate"],
                        "ci95_low": summary["ci95_low"],
                        "ci95_high": summary["ci95_high"],
                        "n_cases": summary["n_cases"],
                        "n_patients": summary["n_patients"],
                        "n_seeds": len(seeds),
                    }
                )

            surface_rows = target_positive[target_positive["surface_defined"]].copy()
            for metric in ("hd95", "assd"):
                if surface_rows.empty:
                    continue
                summary, _ = _metric_summary(
                    surface_rows,
                    metric,
                    replicates=replicates,
                    confidence=confidence,
                    bootstrap_seed=bootstrap_seed,
                    seed_parts=(variant, target, metric, "surface_defined"),
                )
                metric_summaries[metric] = summary
                detailed_rows.append(
                    {
                        "variant": variant,
                        "model": label,
                        "target": target,
                        "scope": "target_positive_and_both_surfaces_defined",
                        "metric": metric,
                        "estimate": summary["estimate"],
                        "ci95_low": summary["ci95_low"],
                        "ci95_high": summary["ci95_high"],
                        "n_cases": summary["n_cases"],
                        "n_patients": summary["n_patients"],
                        "n_seeds": len(seeds),
                    }
                )

            seed_presence: list[dict[str, float]] = []
            for seed in seeds:
                rows = model_case[
                    (model_case["target"] == target) & (model_case["seed"] == seed)
                ]
                truth = rows["target_positive"].to_numpy(dtype=bool)
                prediction = rows["prediction_positive"].to_numpy(dtype=bool)
                tp = int(np.logical_and(truth, prediction).sum())
                tn = int(np.logical_and(~truth, ~prediction).sum())
                fp = int(np.logical_and(~truth, prediction).sum())
                fn = int(np.logical_and(truth, ~prediction).sum())

                def ratio(numerator: int, denominator: int) -> float:
                    return float(numerator / denominator) if denominator else float("nan")

                seed_presence.append(
                    {
                        "tp": float(tp),
                        "tn": float(tn),
                        "fp": float(fp),
                        "fn": float(fn),
                        "sensitivity": ratio(tp, tp + fn),
                        "specificity": ratio(tn, tn + fp),
                        "precision": ratio(tp, tp + fp),
                        "f1": ratio(2 * tp, 2 * tp + fp + fn),
                    }
                )
            presence_frame = pd.DataFrame(seed_presence)
            presence_rows.append(
                {
                    "variant": variant,
                    "model": label,
                    "target": target,
                    "n_cases_per_seed": int(
                        model_case[model_case["target"] == target]["volume_id"].nunique()
                    ),
                    "n_seeds": len(seeds),
                    **{
                        f"mean_{column}": float(presence_frame[column].mean(skipna=True))
                        for column in presence_frame.columns
                    },
                }
            )
            target_summaries[target] = metric_summaries

        if set(TARGETS).issubset(observed_targets):
            for target_order, target in enumerate(TARGETS, start=1):
                summary = target_summaries[target]["dice"]
                agreement_rows.append(
                    {
                        "variant": variant,
                        "model": label,
                        "agreement_level": target,
                        "agreement_order": target_order,
                        "patient_macro_dice": summary["estimate"],
                        "ci95_low": summary["ci95_low"],
                        "ci95_high": summary["ci95_high"],
                        "n_target_positive_cases": summary["n_cases"],
                        "n_patients": summary["n_patients"],
                        "n_seeds": len(seeds),
                    }
                )

        contour_summaries: list[dict[str, Any]] = []
        for contour_count, stratum in primary.groupby("n_clustered_contours", sort=True):
            summary, _ = _metric_summary(
                stratum,
                "dice",
                replicates=replicates,
                confidence=confidence,
                bootstrap_seed=bootstrap_seed,
                seed_parts=(variant, "T2", f"contours_{int(contour_count)}"),
            )
            contour_summaries.append(
                {"n_clustered_contours": int(contour_count), **summary}
            )

        soft_summaries: dict[str, Any] = {}
        if not model_soft.empty:
            for metric in [*brier_columns, nesting_column]:
                summary, _ = _metric_summary(
                    model_soft,
                    metric,
                    replicates=replicates,
                    confidence=confidence,
                    bootstrap_seed=bootstrap_seed,
                    seed_parts=(variant, metric),
                    extra_keys=("n_clustered_contours",),
                )
                soft_summaries[metric] = summary

        model_summaries[variant] = {
            "model": label,
            "primary_t2_target_positive_patient_macro_dice": t2_summary,
            "target_positive_patient_macro_metrics": target_summaries,
            "t2_by_contour_count": contour_summaries,
            "soft_and_nesting_patient_macro": soft_summaries,
        }

    baseline = t2_patient_values[baseline_variant]
    for variant in variants:
        if variant == baseline_variant:
            continue
        candidate = t2_patient_values[variant]
        if not baseline.index.equals(candidate.index):
            baseline_only = sorted(set(baseline.index).difference(candidate.index))
            candidate_only = sorted(set(candidate.index).difference(baseline.index))
            raise CoverageError(
                f"paired comparison {variant} versus {baseline_variant} has different patient "
                f"sets; baseline-only={baseline_only[:10]}, candidate-only={candidate_only[:10]}"
            )
        differences = candidate.to_numpy(dtype=float) - baseline.to_numpy(dtype=float)
        paired_series = pd.Series(differences, index=baseline.index)
        estimate, low, high = _bootstrap_mean_ci(
            paired_series,
            replicates=replicates,
            confidence=confidence,
            seed=_seed_from_parts(bootstrap_seed, "paired", baseline_variant, variant),
        )
        paired_rows.append(
            {
                "baseline_variant": baseline_variant,
                "baseline_model": VARIANT_LABELS[baseline_variant],
                "variant": variant,
                "model": VARIANT_LABELS[variant],
                "metric": "patient_macro_t2_dice",
                "target": "T2",
                "cohort": "target_positive",
                "estimate_difference": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "n_patients": int(len(paired_series)),
                "n_cases": int(
                    case[
                        (case["variant"] == variant)
                        & (case["target"] == "T2")
                        & case["target_positive"]
                    ]["volume_id"].nunique()
                ),
                "n_seeds": len(seeds),
                "bootstrap_replicates": replicates,
                "confidence": confidence,
            }
        )

    all_calibration = _aggregate_calibration_all_scopes(calibration)
    figure_calibration, selected_scope = _select_figure_calibration_scope(
        all_calibration, calibration_scope
    )
    for variant in variants:
        model_calibration: dict[str, Any] = {}
        for scope, group in all_calibration[
            all_calibration["variant"] == variant
        ].groupby("scope", sort=True):
            model_calibration[str(scope)] = _calibration_summary(group)
        model_summaries[variant]["calibration"] = model_calibration

    ordered_history = history.copy()
    ordered_history = ordered_history.sort_values(
        ["variant", "seed", "outer_fold", "epoch"], kind="stable"
    ).reset_index(drop=True)

    outputs = {
        "ablation_metrics.csv": pd.DataFrame(ablation_rows),
        "agreement_metrics.csv": pd.DataFrame(agreement_rows),
        "detailed_metrics.csv": pd.DataFrame(detailed_rows),
        "presence_metrics.csv": pd.DataFrame(presence_rows),
        "calibration.csv": figure_calibration.sort_values(
            ["variant", "bin_lower", "bin_upper"], kind="stable"
        ).reset_index(drop=True),
        "history.csv": ordered_history,
        "paired_comparisons.csv": pd.DataFrame(paired_rows),
    }
    summary = {
        "aggregation": {
            "case_seed_aggregation": "mean each case metric across complete seeds",
            "case_to_patient_aggregation": "mean cases within patient",
            "patient_macro": "unweighted mean across patients",
            "confidence_interval": "percentile patient-cluster bootstrap",
            "paired_comparison": "paired patient bootstrap of candidate-minus-A0 differences",
            "bootstrap_replicates": replicates,
            "bootstrap_seed": bootstrap_seed,
            "confidence": confidence,
            "calibration_pooling": "voxel-count-weighted across complete folds and seeds",
            "figure_calibration_scope": selected_scope,
        },
        "models": model_summaries,
    }
    return outputs, summary


def _build_parser() -> argparse.ArgumentParser:
    description = """Aggregate strict out-of-fold final results into paper/figure tables.

The command refuses to emit aggregate tables unless every requested run is present
and every model/seed has exactly one OOF prediction for each of 325 cases (247
patients), including exactly 237 target-positive T2 cases.
"""
    epilog = """Required evaluator files and columns in every variant/seed/fold directory:

  case_metrics.csv
    target, volume_id, patient_id, outer_fold, seed, model, target_positive,
    prediction_positive, dice, iou, precision, recall, hd95, assd,
    surface_defined

  soft_case_metrics.csv (A3-A8; intentionally absent for A0-A2)
    volume_id, patient_id, outer_fold, seed, model; exactly one nesting/order
    violation fraction/rate column; at least one q_brier*/soft_brier* column

  calibration.csv (A3-A8; intentionally absent for A0-A2)
    model, seed, outer_fold, scope, bin_lower, bin_upper, mean_predicted,
    mean_observed, n_voxels

  target_summary.csv, bootstrap_cis.csv, evaluation_summary.json (or summary.json),
  history.csv
    history.csv requires epoch, train_loss, val_loss

  config.json, run_summary.json, best.pt
    config.json requires the locked variant/seed/fold identity, smoke=false,
    training_code_sha256, dataset_content_sha256, and manifest_sha256;
    run_summary.json requires status=TRAINING_COMPLETE; evaluation_summary.json
    requires evaluation_code_sha256. All four SHA-256 values must be 64-character
    strings and identical across every requested final run.

  n_clustered_contours
    required in case_metrics.csv or soft_case_metrics.csv

Outputs (written only after full validation): ablation_metrics.csv,
agreement_metrics.csv, detailed_metrics.csv, presence_metrics.csv,
calibration.csv, history.csv, paired_comparisons.csv, and
final_results_summary.json. A verified A4_proposed seed-42 OOF prediction set
is copied to predictions/. On failure, run_coverage_report.json is written and
no aggregate output is created or updated.
"""
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"run tree and output directory (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="aggregate output directory (default: --results-root)",
    )
    parser.add_argument(
        "--variants",
        type=_parse_csv_list,
        default=list(VARIANT_LABELS),
        help="comma-separated variant IDs (default: all A0-A8)",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_int_list,
        default=None,
        help="comma-separated seeds; by default discover the union of seed_* directories",
    )
    parser.add_argument(
        "--folds",
        type=_parse_int_list,
        default=list(DEFAULT_FOLDS),
        help="comma-separated outer folds (default: 0,1,2,3,4)",
    )
    parser.add_argument(
        "--baseline-variant",
        default="A0_unet2d_t2",
        help="baseline for paired comparisons (default: A0_unet2d_t2)",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=10_000,
        help="patient-cluster bootstrap replicates (default: 10000)",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260904,
        help="base RNG seed for deterministic aggregation (default: 20260904)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="two-sided bootstrap confidence level (default: 0.95)",
    )
    parser.add_argument(
        "--calibration-scope",
        default="auto",
        help="scope exported for figure calibration.csv; auto uses a documented priority",
    )
    parser.add_argument(
        "--prediction-variant",
        default="A4_proposed",
        help="variant used for the aggregate qualitative prediction set (default: A4_proposed)",
    )
    parser.add_argument(
        "--prediction-seed",
        type=int,
        default=42,
        help="seed used for the aggregate qualitative prediction set (default: 42)",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    unknown = [variant for variant in args.variants if variant not in VARIANT_LABELS]
    if unknown:
        raise CoverageError(
            f"unknown variants {unknown}; valid IDs are {list(VARIANT_LABELS)}"
        )
    if len(args.variants) != len(set(args.variants)):
        raise CoverageError("--variants contains duplicates")
    if args.baseline_variant not in args.variants:
        raise CoverageError("--baseline-variant must be included in --variants")
    if args.prediction_variant not in args.variants:
        raise CoverageError("--prediction-variant must be included in --variants")
    if args.prediction_variant in T2_ONLY_VARIANTS:
        raise CoverageError("--prediction-variant must be a four-tier agreement variant")
    if tuple(sorted(args.folds)) != DEFAULT_FOLDS:
        raise CoverageError(
            f"the locked final protocol requires outer folds {list(DEFAULT_FOLDS)}"
        )
    if not any(variant not in T2_ONLY_VARIANTS for variant in args.variants):
        raise CoverageError(
            "at least one four-tier agreement variant is required for agreement/calibration outputs"
        )
    if args.bootstrap_replicates < 1_000:
        raise CoverageError("--bootstrap-replicates must be at least 1000")
    if not 0.5 < args.confidence < 1.0:
        raise CoverageError("--confidence must lie strictly between 0.5 and 1")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    results_root = args.results_root.resolve()
    output_root = (args.output_root or results_root).resolve()
    report_path = output_root / "run_coverage_report.json"
    try:
        _validate_args(args)
        seeds = sorted(args.seeds) if args.seeds is not None else _discover_seeds(
            results_root, args.variants
        )
        if args.prediction_seed not in seeds:
            raise CoverageError(
                f"prediction seed {args.prediction_seed} is not in the complete requested seed set {seeds}"
            )
        runs, coverage = _coverage_scan(results_root, args.variants, seeds, args.folds)
        case, soft, calibration, history, run_metadata = _load_runs(runs)
        oof_report = _validate_oof_panels(
            case,
            soft,
            variants=args.variants,
            seeds=seeds,
            expected_cases=EXPECTED_CASES,
            expected_patients=EXPECTED_PATIENTS,
            expected_t2_positive=EXPECTED_T2_POSITIVE,
        )
        outputs, aggregate_summary = _aggregate_results(
            case,
            soft,
            calibration,
            history,
            variants=args.variants,
            seeds=seeds,
            baseline_variant=args.baseline_variant,
            replicates=args.bootstrap_replicates,
            confidence=args.confidence,
            bootstrap_seed=args.bootstrap_seed,
            calibration_scope=args.calibration_scope,
        )
        prediction_sources, prediction_validation = _prediction_sources(
            runs,
            case,
            variant=args.prediction_variant,
            seed=args.prediction_seed,
            folds=args.folds,
            expected_cases=EXPECTED_CASES,
        )
        prediction_assembly = _assemble_predictions(prediction_sources, output_root)

        final_summary = {
            "status": "COMPLETE",
            "created_at_utc": _iso_now(),
            "results_root": str(results_root),
            "output_root": str(output_root),
            "coverage": coverage,
            "out_of_fold_validation": oof_report,
            "run_metadata": run_metadata,
            "expected_cardinalities": {
                "cases": EXPECTED_CASES,
                "patients": EXPECTED_PATIENTS,
                "t2_target_positive_cases": EXPECTED_T2_POSITIVE,
            },
            "qualitative_predictions": {
                **prediction_validation,
                **prediction_assembly,
            },
            **aggregate_summary,
        }

        output_root.mkdir(parents=True, exist_ok=True)
        for filename, frame in outputs.items():
            _atomic_write_csv(output_root / filename, frame)
        _write_json(output_root / "final_results_summary.json", final_summary)
        _write_json(
            report_path,
            {
                **coverage,
                "out_of_fold_validation": oof_report,
                "qualitative_predictions": {
                    **prediction_validation,
                    **prediction_assembly,
                },
            },
        )
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "output_root": str(output_root),
                    "runs": len(runs),
                    "variants": list(args.variants),
                    "seeds": seeds,
                    "folds": list(args.folds),
                    "outputs": [*outputs, "final_results_summary.json"],
                },
                indent=2,
            )
        )
        return 0
    except CoverageError as exc:
        failure_report = {
            "status": "INCOMPLETE",
            "checked_at_utc": _iso_now(),
            "error": str(exc),
            "results_root": str(results_root),
            "output_root": str(output_root),
            **exc.report,
        }
        try:
            _write_json(report_path, failure_report)
        except Exception as report_exc:
            print(f"also failed to write coverage report: {report_exc}", file=sys.stderr)
        print(json.dumps(_json_safe(failure_report), indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
