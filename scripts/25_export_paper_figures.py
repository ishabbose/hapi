#!/usr/bin/env python3
"""Export data-backed, publication-quality figures for the final study.

The script intentionally does not contain example scores or fallback values.  It
expects the final evaluator to write aggregate metrics and held-out prediction
files, validates those inputs, and stops with a concrete schema message when a
required result is absent or ambiguous.

Default result layout::

    results/final_agreement/aggregates/screen/
      ablation_metrics.csv
      agreement_metrics.csv
      calibration.csv
      predictions/<subset_nodule_id>.npz
      history.csv                         # optional
      qualitative_cases.csv              # optional

CSV files may be replaced by JSON files with the same stem.  JSON tables may be
a list of row objects or an object containing ``rows``, ``records``, ``data``,
``results``, or ``metrics``.  See ``--help`` and the individual validation
errors for the exact column contracts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


HEADS = ("T1", "T2", "T3", "T4")
DEFAULT_RESULTS_ROOT = Path("results/final_agreement/aggregates/screen")
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#7A7A7A"
LIGHT_GRAY = "#D9D9D9"


class InputContractError(RuntimeError):
    """Raised when a result cannot support the requested figure."""


@dataclass(frozen=True)
class FigureRecord:
    figure_id: str
    title: str
    png_path: Path
    pdf_path: Path
    source_files: tuple[Path, ...]
    model: str
    cases: tuple[str, ...] = ()
    selection_rule: str = ""
    notes: str = ""


@dataclass(frozen=True)
class QualitativeCase:
    case_id: str
    patient_id: str
    n_clustered_contours: int
    disagreement_fraction: float
    z_index: int
    image: np.ndarray
    vote_fraction: np.ndarray
    t2_target: np.ndarray
    t2_probability: np.ndarray
    selected_threshold: float
    prediction_path: Path


def configure_style() -> None:
    """Use restrained typography and strokes suitable for IEEE figures."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.0,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _candidate_table_path(root: Path, stem: str, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise InputContractError(f"Requested result table does not exist: {path}")
        return path
    csv_path = root / f"{stem}.csv"
    json_path = root / f"{stem}.json"
    if csv_path.is_file() and json_path.is_file():
        raise InputContractError(
            f"Both {csv_path} and {json_path} exist. Pass the intended file explicitly "
            f"with --{stem.replace('_', '-')} to avoid reading an ambiguous result."
        )
    if csv_path.is_file():
        return csv_path
    if json_path.is_file():
        return json_path
    return csv_path


def _load_table(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise InputContractError(f"Missing {label}: {path}")
    try:
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        elif path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                frame = pd.DataFrame(payload)
            elif isinstance(payload, dict):
                nested = None
                for key in ("rows", "records", "data", "results", "metrics"):
                    if isinstance(payload.get(key), list):
                        nested = payload[key]
                        break
                if nested is not None:
                    frame = pd.DataFrame(nested)
                elif all(not isinstance(value, (dict, list)) for value in payload.values()):
                    frame = pd.DataFrame([payload])
                else:
                    try:
                        frame = pd.DataFrame(payload)
                    except ValueError as error:
                        raise InputContractError(
                            f"{label} JSON must be a list of rows or contain one of "
                            "rows/records/data/results/metrics as a list."
                        ) from error
            else:
                raise InputContractError(f"{label} JSON must contain an object or list of rows.")
        else:
            raise InputContractError(f"{label} must be CSV or JSON: {path}")
    except (OSError, json.JSONDecodeError, pd.errors.ParserError) as error:
        raise InputContractError(f"Could not read {label} {path}: {error}") from error
    if frame.empty:
        raise InputContractError(f"{label} contains no rows: {path}")
    return frame


def _rename_alias(frame: pd.DataFrame, canonical: str, aliases: Sequence[str]) -> pd.DataFrame:
    if canonical in frame.columns:
        return frame
    matches = [name for name in aliases if name in frame.columns]
    if len(matches) > 1:
        raise InputContractError(
            f"Multiple columns could represent {canonical}: {matches}. Rename the intended "
            f"column to {canonical}."
        )
    if matches:
        return frame.rename(columns={matches[0]: canonical})
    return frame


def _coerce_numeric(frame: pd.DataFrame, columns: Iterable[str], label: str) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (ValueError, TypeError) as error:
            raise InputContractError(f"{label}.{column} must contain only numeric values.") from error
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise InputContractError(f"{label}.{column} contains a non-finite value.")
    return frame


def _coerce_bool(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    invalid = sorted(set(normalized) - set(mapping))
    if invalid:
        raise InputContractError(f"{label} must contain booleans; invalid values: {invalid[:5]}")
    return normalized.map(mapping).astype(bool)


def _validate_unit_interval(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        if np.any((values < 0.0) | (values > 1.0)):
            raise InputContractError(
                f"{label}.{column} must be a proportion in [0, 1], not a percentage."
            )


def load_ablation(path: Path) -> pd.DataFrame:
    frame = _load_table(path, "ablation metrics")
    frame = _rename_alias(frame, "model", ("variant", "configuration", "ablation"))
    frame = _rename_alias(
        frame,
        "patient_macro_t2_dice",
        ("t2_patient_macro_dice", "t2_dice_patient_macro", "patient_macro_dice", "dice"),
    )
    frame = _rename_alias(frame, "ci95_low", ("ci_low", "lower_ci", "dice_ci95_low"))
    frame = _rename_alias(frame, "ci95_high", ("ci_high", "upper_ci", "dice_ci95_high"))
    if "agreement_level" in frame.columns:
        level = frame["agreement_level"].astype(str).str.strip().str.upper()
        frame = frame[level.isin({"T2", "2"})].copy()
    required = {"model", "patient_macro_t2_dice", "ci95_low", "ci95_high"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise InputContractError(
            f"Ablation metrics are missing {missing}. Create {path} with one aggregate row per "
            "ablation and columns model,patient_macro_t2_dice,ci95_low,ci95_high. "
            "All three metric columns must be patient-macro proportions in [0,1], and the "
            "confidence bounds must be a patient-level 95% CI. Optional columns: order,is_proposed."
        )
    if frame.empty:
        raise InputContractError(f"{path} has no T2 ablation rows.")
    frame["model"] = frame["model"].astype(str).str.strip()
    if (frame["model"] == "").any() or frame["model"].duplicated().any():
        duplicates = frame.loc[frame["model"].duplicated(False), "model"].tolist()
        raise InputContractError(
            "Ablation metrics require one non-empty aggregate row per model. "
            f"Duplicate labels: {duplicates[:8]}"
        )
    frame = _coerce_numeric(
        frame,
        ("patient_macro_t2_dice", "ci95_low", "ci95_high"),
        "ablation_metrics",
    )
    _validate_unit_interval(
        frame,
        ("patient_macro_t2_dice", "ci95_low", "ci95_high"),
        "ablation_metrics",
    )
    score = frame["patient_macro_t2_dice"].to_numpy(dtype=float)
    low = frame["ci95_low"].to_numpy(dtype=float)
    high = frame["ci95_high"].to_numpy(dtype=float)
    if np.any(low > score) or np.any(score > high):
        raise InputContractError(
            "Each ablation 95% CI must contain its patient-macro T2 Dice point estimate."
        )
    if "order" in frame.columns:
        frame = _coerce_numeric(frame, ("order",), "ablation_metrics")
        if frame["order"].duplicated().any():
            raise InputContractError("ablation_metrics.order must be unique.")
        frame = frame.sort_values("order", kind="stable")
    if "is_proposed" in frame.columns:
        frame["is_proposed"] = _coerce_bool(frame["is_proposed"], "is_proposed")
    else:
        frame["is_proposed"] = False
    return frame.reset_index(drop=True)


def _normalize_head(value: object) -> str:
    text = str(value).strip().upper().replace("HEAD", "").replace("_", "")
    if text in {"1", "T1", ">=1", "≥1"}:
        return "T1"
    if text in {"2", "T2", ">=2", "≥2"}:
        return "T2"
    if text in {"3", "T3", ">=3", "≥3"}:
        return "T3"
    if text in {"4", "T4", ">=4", "≥4"}:
        return "T4"
    raise InputContractError(f"Unknown agreement level {value!r}; expected T1, T2, T3, or T4.")


def load_agreement(path: Path) -> pd.DataFrame:
    frame = _load_table(path, "agreement-level metrics")
    frame = _rename_alias(frame, "model", ("variant", "configuration"))
    frame = _rename_alias(frame, "agreement_level", ("head", "target", "tier"))
    frame = _rename_alias(
        frame,
        "patient_macro_dice",
        ("dice_patient_macro", "patient_dice", "dice"),
    )
    frame = _rename_alias(frame, "ci95_low", ("ci_low", "lower_ci", "dice_ci95_low"))
    frame = _rename_alias(frame, "ci95_high", ("ci_high", "upper_ci", "dice_ci95_high"))
    required = {"agreement_level", "patient_macro_dice", "ci95_low", "ci95_high"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise InputContractError(
            f"Agreement metrics are missing {missing}. Create {path} with four rows per model "
            "and columns model,agreement_level,patient_macro_dice,ci95_low,ci95_high. "
            "agreement_level must be T1,T2,T3,T4; overlap rows must use the prespecified "
            "target-positive endpoint before patient-macro aggregation."
        )
    if "model" not in frame.columns:
        frame["model"] = ""
    frame["model"] = frame["model"].fillna("").astype(str).str.strip()
    frame["agreement_level"] = frame["agreement_level"].map(_normalize_head)
    frame = _coerce_numeric(
        frame,
        ("patient_macro_dice", "ci95_low", "ci95_high"),
        "agreement_metrics",
    )
    _validate_unit_interval(
        frame,
        ("patient_macro_dice", "ci95_low", "ci95_high"),
        "agreement_metrics",
    )
    score = frame["patient_macro_dice"].to_numpy(dtype=float)
    low = frame["ci95_low"].to_numpy(dtype=float)
    high = frame["ci95_high"].to_numpy(dtype=float)
    if np.any(low > score) or np.any(score > high):
        raise InputContractError(
            "Each agreement-level 95% CI must contain its patient-macro Dice point estimate."
        )
    duplicates = frame.duplicated(["model", "agreement_level"], keep=False)
    if duplicates.any():
        values = frame.loc[duplicates, ["model", "agreement_level"]].astype(str).agg(": ".join, axis=1)
        raise InputContractError(
            "Agreement metrics require one aggregate row per model/head. Duplicate rows: "
            + ", ".join(values.tolist()[:8])
        )
    for model, group in frame.groupby("model", dropna=False):
        present = set(group["agreement_level"])
        if present != set(HEADS):
            raise InputContractError(
                f"Agreement metrics for model {model or '<unspecified>'!r} must contain exactly "
                f"T1-T4; found {sorted(present)}."
            )
    return frame.reset_index(drop=True)


def load_calibration(path: Path, bins: int) -> tuple[pd.DataFrame, str]:
    frame = _load_table(path, "calibration data")
    frame = _rename_alias(frame, "model", ("variant", "configuration"))
    if "model" not in frame.columns:
        frame["model"] = ""
    frame["model"] = frame["model"].fillna("").astype(str).str.strip()

    # Accept either evaluator-produced reliability bins or raw paired q-hat/q rows.
    frame = _rename_alias(
        frame,
        "mean_predicted",
        ("q_hat_mean", "predicted_mean", "mean_q_hat", "confidence"),
    )
    frame = _rename_alias(
        frame,
        "mean_observed",
        ("q_mean", "observed_mean", "mean_q", "accuracy"),
    )
    frame = _rename_alias(frame, "n_voxels", ("count", "n", "bin_count"))
    if {"mean_predicted", "mean_observed", "n_voxels"}.issubset(frame.columns):
        frame = _coerce_numeric(
            frame,
            ("mean_predicted", "mean_observed", "n_voxels"),
            "calibration",
        )
        _validate_unit_interval(frame, ("mean_predicted", "mean_observed"), "calibration")
        if (frame["n_voxels"] <= 0).any():
            raise InputContractError("calibration.n_voxels must be positive for every retained bin.")
        if "bin_lower" in frame.columns or "bin_upper" in frame.columns:
            if not {"bin_lower", "bin_upper"}.issubset(frame.columns):
                raise InputContractError(
                    "Calibration bins must provide both bin_lower and bin_upper, or neither."
                )
            frame = _coerce_numeric(frame, ("bin_lower", "bin_upper"), "calibration")
            _validate_unit_interval(frame, ("bin_lower", "bin_upper"), "calibration")
            if (frame["bin_lower"] >= frame["bin_upper"]).any():
                raise InputContractError("Every calibration bin must satisfy bin_lower < bin_upper.")
        return frame.reset_index(drop=True), "pre-binned evaluator output"

    raw = frame.copy()
    raw = _rename_alias(raw, "q_hat", ("predicted_support", "prediction", "probability"))
    raw = _rename_alias(raw, "q", ("observed_support", "target", "vote_fraction"))
    if not {"q_hat", "q"}.issubset(raw.columns):
        raise InputContractError(
            f"Calibration data must be either binned or paired raw support. Create {path} with "
            "columns model,bin_lower,bin_upper,mean_predicted,mean_observed,n_voxels; or with "
            "model,q_hat,q. q_hat and q must be in [0,1]."
        )
    raw = _coerce_numeric(raw, ("q_hat", "q"), "calibration")
    _validate_unit_interval(raw, ("q_hat", "q"), "calibration")
    if bins < 2:
        raise InputContractError("--calibration-bins must be at least 2.")
    rows: list[dict[str, object]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for model, group in raw.groupby("model", sort=False, dropna=False):
        q_hat = group["q_hat"].to_numpy(dtype=float)
        q = group["q"].to_numpy(dtype=float)
        bin_index = np.minimum(np.searchsorted(edges, q_hat, side="right") - 1, bins - 1)
        bin_index = np.maximum(bin_index, 0)
        for index in range(bins):
            selected = bin_index == index
            if not selected.any():
                continue
            rows.append(
                {
                    "model": str(model),
                    "bin_lower": float(edges[index]),
                    "bin_upper": float(edges[index + 1]),
                    "mean_predicted": float(q_hat[selected].mean()),
                    "mean_observed": float(q[selected].mean()),
                    "n_voxels": int(selected.sum()),
                }
            )
    return pd.DataFrame(rows), f"{bins} equal-width bins computed from paired q_hat/q rows"


def infer_model(
    requested: str | None,
    ablation: pd.DataFrame,
    agreement: pd.DataFrame,
    calibration: pd.DataFrame,
) -> str | None:
    available = set(ablation["model"])
    available.update(value for value in agreement["model"] if value)
    available.update(value for value in calibration["model"] if value)
    if requested is not None:
        if requested not in available:
            raise InputContractError(
                f"--model {requested!r} was not found. Exact available labels: {sorted(available)}"
            )
        return requested
    proposed = ablation.loc[ablation["is_proposed"], "model"].tolist()
    if len(proposed) > 1:
        raise InputContractError(
            "More than one ablation row has is_proposed=true. Mark exactly one or pass --model."
        )
    if len(proposed) == 1:
        return proposed[0]
    named_agreement = sorted(set(value for value in agreement["model"] if value))
    named_calibration = sorted(set(value for value in calibration["model"] if value))
    candidates = set(named_agreement) | set(named_calibration)
    if len(candidates) == 1:
        return next(iter(candidates))
    if len(ablation) == 1:
        return str(ablation.iloc[0]["model"])
    return None


def filter_model(frame: pd.DataFrame, selected_model: str | None, label: str) -> pd.DataFrame:
    named = sorted(set(value for value in frame["model"].astype(str) if value))
    if not named:
        return frame.copy()
    if selected_model is None:
        if len(named) > 1:
            raise InputContractError(
                f"{label} contains multiple models {named}. Pass --model or mark exactly one "
                "ablation row with is_proposed=true."
            )
        return frame[frame["model"] == named[0]].copy()
    selected = frame[frame["model"] == selected_model].copy()
    if selected.empty:
        raise InputContractError(
            f"{label} has no rows for selected model {selected_model!r}; available: {named}"
        )
    return selected


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> tuple[Path, Path]:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return png_path, pdf_path


def plot_ablation(
    frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    selected_model: str | None,
) -> tuple[Path, Path]:
    labels = frame["model"].tolist()
    score = frame["patient_macro_t2_dice"].to_numpy(dtype=float)
    low = frame["ci95_low"].to_numpy(dtype=float)
    high = frame["ci95_high"].to_numpy(dtype=float)
    proposed = frame["is_proposed"].to_numpy(dtype=bool)
    if selected_model is not None:
        proposed |= frame["model"].eq(selected_model).to_numpy()
    colors = [BLUE if value else GRAY for value in proposed]

    height = max(2.0, 0.35 * len(frame) + 0.75)
    fig, axis = plt.subplots(figsize=(7.16, height), constrained_layout=True)
    positions = np.arange(len(frame))
    axis.barh(positions, score, height=0.62, color=colors, edgecolor="none", zorder=2)
    axis.errorbar(
        score,
        positions,
        xerr=np.vstack([score - low, high - score]),
        fmt="none",
        ecolor="black",
        elinewidth=0.8,
        capsize=2.2,
        capthick=0.8,
        zorder=3,
    )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.xaxis.set_major_locator(MultipleLocator(0.1))
    axis.xaxis.set_minor_locator(MultipleLocator(0.05))
    axis.set_xlabel("Patient-macro T2 Dice")
    axis.grid(axis="x", color=LIGHT_GRAY, linewidth=0.55, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    return _save_figure(fig, output_dir, "figure_01_ablation_t2_dice", dpi)


def plot_agreement(
    frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    ordered = frame.set_index("agreement_level").loc[list(HEADS)].reset_index()
    score = ordered["patient_macro_dice"].to_numpy(dtype=float)
    low = ordered["ci95_low"].to_numpy(dtype=float)
    high = ordered["ci95_high"].to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)
    x = np.arange(1, 5)
    axis.errorbar(
        x,
        score,
        yerr=np.vstack([score - low, high - score]),
        color=BLUE,
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.0,
        capsize=2.5,
        elinewidth=0.8,
    )
    axis.set_xticks(x, ["T1 (≥1)", "T2 (≥2)", "T3 (≥3)", "T4 (≥4)"])
    axis.set_xlim(0.7, 4.3)
    axis.set_ylim(0.0, 1.0)
    axis.yaxis.set_major_locator(MultipleLocator(0.1))
    axis.set_ylabel("Patient-macro Dice")
    axis.set_xlabel("Reader-support target")
    axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.55)
    axis.spines[["top", "right"]].set_visible(False)
    return _save_figure(fig, output_dir, "figure_02_agreement_level_dice", dpi)


def plot_calibration(
    frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> tuple[tuple[Path, Path], float]:
    ordered = frame.sort_values("mean_predicted", kind="stable")
    predicted = ordered["mean_predicted"].to_numpy(dtype=float)
    observed = ordered["mean_observed"].to_numpy(dtype=float)
    counts = ordered["n_voxels"].to_numpy(dtype=float)
    ece = float(np.average(np.abs(predicted - observed), weights=counts))
    marker_area = 16.0 + 30.0 * np.sqrt(counts / counts.max())

    fig, axis = plt.subplots(figsize=(3.5, 3.05), constrained_layout=True)
    axis.plot([0, 1], [0, 1], linestyle="--", color=GRAY, linewidth=0.9, label="Ideal")
    axis.plot(predicted, observed, color=BLUE, linewidth=1.2, zorder=2)
    axis.scatter(
        predicted,
        observed,
        s=marker_area,
        color=BLUE,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        label="Observed support",
    )
    axis.text(
        0.04,
        0.94,
        f"ECE = {ece:.3f}",
        transform=axis.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": LIGHT_GRAY},
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal", adjustable="box")
    axis.xaxis.set_major_locator(MultipleLocator(0.2))
    axis.yaxis.set_major_locator(MultipleLocator(0.2))
    axis.set_xlabel(r"Predicted reader support $\hat{q}$")
    axis.set_ylabel(r"Observed reader support $q$")
    axis.grid(color=LIGHT_GRAY, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    paths = _save_figure(fig, output_dir, "figure_03_reader_support_calibration", dpi)
    return paths, ece


def _resolve_dataset_path(data_root: Path, value: object, case_id: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise InputContractError(
            f"Dataset manifest path for {case_id} must be portable and relative: {value}"
        )
    path = (data_root / relative).resolve()
    if not path.is_relative_to(data_root.resolve()):
        raise InputContractError(f"Dataset path escapes --data-root for {case_id}: {value}")
    if not path.is_file():
        raise InputContractError(f"Missing v3 array for {case_id}: {path}")
    return path


def load_dataset_manifest(data_root: Path) -> tuple[pd.DataFrame, Path]:
    path = data_root / "final_325_manifest.csv"
    if not path.is_file():
        raise InputContractError(
            f"Missing v3 manifest: {path}. Run scripts/19_build_final_v3.py and "
            "scripts/20_validate_final_v3.py before exporting qualitative panels."
        )
    frame = pd.read_csv(path)
    required = {
        "subset_nodule_id",
        "patient_id",
        "split",
        "n_review_sessions",
        "n_clustered_contours",
        "majority_voxels",
        "disagreement_fraction_within_union",
        "image_volume_uint8_path",
        "reader_vote_count_path",
        "reader_vote_fraction_path",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise InputContractError(f"v3 manifest is missing qualitative-panel columns: {missing}")
    frame["subset_nodule_id"] = frame["subset_nodule_id"].astype(str)
    if frame["subset_nodule_id"].duplicated().any():
        raise InputContractError("v3 manifest contains duplicate subset_nodule_id values.")
    frame = _coerce_numeric(
        frame,
        (
            "n_review_sessions",
            "n_clustered_contours",
            "majority_voxels",
            "disagreement_fraction_within_union",
        ),
        "v3_manifest",
    )
    if not (frame["n_review_sessions"] == 4).all():
        raise InputContractError("v3 manifest must contain four review-session slots per case.")
    if not frame["n_clustered_contours"].between(1, 4).all():
        raise InputContractError("n_clustered_contours must lie in [1,4].")
    return frame, path


def _prediction_files(predictions_dir: Path) -> dict[str, Path]:
    if not predictions_dir.is_dir():
        raise InputContractError(
            f"Missing prediction directory: {predictions_dir}. Export one held-out NPZ per case "
            "as predictions/<subset_nodule_id>.npz. Each NPZ must contain probabilities or logits "
            "with shape [4,D,H,W] ordered T1-T4, or t2_probability with shape [D,H,W]."
        )
    files = sorted(predictions_dir.glob("*.npz"))
    if not files:
        raise InputContractError(
            f"No prediction NPZ files found in {predictions_dir}. Save held-out case predictions "
            "as <subset_nodule_id>.npz with probabilities/logits [4,D,H,W] or "
            "t2_probability [D,H,W]."
        )
    mapping: dict[str, Path] = {}
    for path in files:
        if path.stem in mapping:
            raise InputContractError(f"Duplicate prediction file stem: {path.stem}")
        mapping[path.stem] = path
    return mapping


def _read_t2_probability(
    path: Path, expected_shape: tuple[int, int, int]
) -> tuple[np.ndarray, float]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            keys = set(payload.files)
            if "threshold" not in keys:
                raise InputContractError(
                    f"{path}: held-out prediction archive must store its validation-selected threshold."
                )
            threshold_values = np.asarray(payload["threshold"], dtype=np.float64).reshape(-1)
            if threshold_values.size != 1:
                raise InputContractError(f"{path}: threshold must be scalar.")
            threshold = float(threshold_values[0])
            if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
                raise InputContractError(f"{path}: threshold must be finite and in (0,1).")
            if "t2_probability" in keys:
                probability = np.asarray(payload["t2_probability"])
            elif "t2_prediction" in keys:
                probability = np.asarray(payload["t2_prediction"])
            elif "probabilities" in keys:
                values = np.asarray(payload["probabilities"])
                if values.ndim == 5 and values.shape[0] == 1:
                    values = values[0]
                if values.ndim != 4:
                    raise InputContractError(
                        f"{path}: probabilities must have shape [4,D,H,W] or [D,4,H,W]."
                    )
                if values.shape[0] == 4:
                    probability = values[1]
                elif values.shape[1] == 4:
                    probability = np.moveaxis(values, 1, 0)[1]
                else:
                    raise InputContractError(
                        f"{path}: probabilities have no four-head axis: {values.shape}"
                    )
            elif "logits" in keys:
                values = np.asarray(payload["logits"])
                if values.ndim == 5 and values.shape[0] == 1:
                    values = values[0]
                if values.ndim != 4:
                    raise InputContractError(
                        f"{path}: logits must have shape [4,D,H,W] or [D,4,H,W]."
                    )
                if values.shape[0] == 4:
                    logits = values[1]
                elif values.shape[1] == 4:
                    logits = np.moveaxis(values, 1, 0)[1]
                else:
                    raise InputContractError(f"{path}: logits have no four-head axis: {values.shape}")
                probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
            else:
                raise InputContractError(
                    f"{path} has keys {sorted(keys)}. Add probabilities or logits [4,D,H,W], "
                    "or t2_probability [D,H,W]."
                )
    except (OSError, ValueError) as error:
        if isinstance(error, InputContractError):
            raise
        raise InputContractError(f"Could not read prediction NPZ {path}: {error}") from error
    probability = np.squeeze(np.asarray(probability))
    if probability.shape != expected_shape:
        raise InputContractError(
            f"{path}: T2 probability shape {probability.shape} != v3 image shape {expected_shape}."
        )
    if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
        raise InputContractError(f"{path}: T2 probabilities must be finite and in [0,1].")
    return probability.astype(np.float32, copy=False), threshold


def _load_selection_file(path: Path) -> list[str]:
    frame = _load_table(path, "qualitative case selection")
    frame = _rename_alias(frame, "subset_nodule_id", ("case_id", "nodule_id"))
    if "subset_nodule_id" not in frame.columns:
        raise InputContractError(
            f"{path} must contain subset_nodule_id (optional column: order)."
        )
    if "order" in frame.columns:
        frame = _coerce_numeric(frame, ("order",), "qualitative_cases")
        frame = frame.sort_values("order", kind="stable")
    identifiers = frame["subset_nodule_id"].astype(str).str.strip().tolist()
    if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise InputContractError("qualitative_cases contains blank or duplicate identifiers.")
    return identifiers


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count >= length:
        return list(range(length))
    values = np.linspace(0, length - 1, num=count)
    chosen: list[int] = []
    for value in values:
        index = int(round(float(value)))
        if index not in chosen:
            chosen.append(index)
    # Rounding can collide for small arrays; fill deterministically if needed.
    for index in range(length):
        if len(chosen) == count:
            break
        if index not in chosen:
            chosen.append(index)
    return sorted(chosen)


def select_qualitative_ids(
    manifest: pd.DataFrame,
    predictions: dict[str, Path],
    requested: Sequence[str] | None,
    selection_path: Path | None,
    count: int,
    pool: str,
) -> tuple[list[str], str, Path | None]:
    if requested:
        identifiers = [value.strip() for value in requested if value.strip()]
        if len(set(identifiers)) != len(identifiers):
            raise InputContractError("--qualitative-cases contains duplicate identifiers.")
        return identifiers, "explicit --qualitative-cases order", None
    if selection_path is not None and selection_path.is_file():
        return _load_selection_file(selection_path), "explicit qualitative_cases table order", selection_path
    if count < 1:
        raise InputContractError("--num-qualitative must be at least 1.")
    eligible = manifest[
        manifest["subset_nodule_id"].isin(predictions)
        & (manifest["majority_voxels"] > 0)
    ].copy()
    if pool == "test":
        eligible = eligible[eligible["split"].astype(str).str.lower() == "test"].copy()
    if eligible.empty:
        suffix = (
            " Export held-out fixed-test predictions, or set --qualitative-pool all only when "
            "the available files are verified out-of-fold predictions."
            if pool == "test"
            else ""
        )
        raise InputContractError(
            "No prediction-backed cases with a positive T2 target are eligible for qualitative "
            f"selection in pool={pool!r}.{suffix}"
        )
    if len(eligible) < count:
        raise InputContractError(
            f"Requested {count} qualitative cases but only {len(eligible)} prediction-backed "
            f"T2-positive cases are eligible in pool={pool!r}. Export additional "
            "held-out predictions, lower --num-qualitative, or provide an explicit valid "
            "--qualitative-cases list."
        )
    eligible = eligible.sort_values(
        ["disagreement_fraction_within_union", "subset_nodule_id"], kind="stable"
    ).reset_index(drop=True)
    indices = _evenly_spaced_indices(len(eligible), count)
    identifiers = eligible.iloc[indices]["subset_nodule_id"].tolist()
    rule = (
        f"deterministic {pool}-pool reader-disagreement spectrum: evenly spaced ranks after "
        "sorting eligible prediction-backed cases by disagreement_fraction_within_union"
    )
    return identifiers, rule, None


def load_qualitative_cases(
    identifiers: Sequence[str],
    manifest: pd.DataFrame,
    data_root: Path,
    predictions: dict[str, Path],
) -> list[QualitativeCase]:
    indexed = manifest.set_index("subset_nodule_id", drop=False)
    cases: list[QualitativeCase] = []
    for case_id in identifiers:
        if case_id not in indexed.index:
            raise InputContractError(f"Qualitative case {case_id!r} is absent from the v3 manifest.")
        if case_id not in predictions:
            raise InputContractError(
                f"Missing qualitative prediction {predictions and next(iter(predictions.values())).parent / (case_id + '.npz')}. "
                "Export this case's held-out NPZ or remove it from the requested selection."
            )
        row = indexed.loc[case_id]
        if isinstance(row, pd.DataFrame):
            raise InputContractError(f"Duplicate v3 manifest rows for {case_id}.")
        n_clustered_contours = int(row["n_clustered_contours"])
        image_path = _resolve_dataset_path(data_root, row["image_volume_uint8_path"], case_id)
        votes_path = _resolve_dataset_path(data_root, row["reader_vote_count_path"], case_id)
        q_path = _resolve_dataset_path(data_root, row["reader_vote_fraction_path"], case_id)
        image = np.load(image_path, allow_pickle=False)
        votes = np.load(votes_path, allow_pickle=False)
        q = np.load(q_path, allow_pickle=False)
        if image.ndim != 3 or votes.shape != image.shape or q.shape != image.shape:
            raise InputContractError(
                f"{case_id}: image, vote count, and vote fraction must share [D,H,W]; found "
                f"{image.shape}, {votes.shape}, {q.shape}."
            )
        if image.dtype != np.uint8:
            raise InputContractError(f"{case_id}: qualitative source image must be uint8.")
        expected_q = votes.astype(np.float32) / 4.0
        if not np.allclose(q, expected_q, atol=1e-6):
            raise InputContractError(f"{case_id}: vote_fraction is inconsistent with vote_count.")
        t2 = votes >= 2
        areas = t2.reshape(t2.shape[0], -1).sum(axis=1)
        if int(areas.max()) <= 0:
            raise InputContractError(
                f"{case_id} has no positive T2 voxel; it cannot show the requested T2 panel."
            )
        max_area = int(areas.max())
        tied = np.flatnonzero(areas == max_area)
        z_index = int(tied[len(tied) // 2])
        t2_probability, selected_threshold = _read_t2_probability(
            predictions[case_id], tuple(image.shape)
        )
        cases.append(
            QualitativeCase(
                case_id=case_id,
                patient_id=str(row["patient_id"]),
                n_clustered_contours=n_clustered_contours,
                disagreement_fraction=float(row["disagreement_fraction_within_union"]),
                z_index=z_index,
                image=np.asarray(image[z_index], dtype=np.float32) / 255.0,
                vote_fraction=np.asarray(q[z_index], dtype=np.float32),
                t2_target=np.asarray(t2[z_index], dtype=bool),
                t2_probability=np.asarray(t2_probability[z_index], dtype=np.float32),
                selected_threshold=selected_threshold,
                prediction_path=predictions[case_id],
            )
        )
    return cases


def _overlay_mask(axis: plt.Axes, mask: np.ndarray, color: tuple[float, float, float], alpha: float) -> None:
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[..., :3] = color
    overlay[..., 3] = mask.astype(np.float32) * alpha
    axis.imshow(overlay, interpolation="nearest")


def _draw_contour(axis: plt.Axes, mask: np.ndarray, color: str, linewidth: float = 0.8) -> None:
    if mask.any() and (~mask).any():
        axis.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=linewidth)


def plot_qualitative(
    cases: Sequence[QualitativeCase],
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    rows = len(cases)
    fig, axes = plt.subplots(
        rows,
        4,
        figsize=(7.16, max(1.8, 1.58 * rows)),
        squeeze=False,
        constrained_layout=True,
    )
    titles = ("CT candidate ROI", "Reader agreement q", "T2 ground truth", "Proposed T2 probability")
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, pad=3.0)
    for row_index, case in enumerate(cases):
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.45)
                spine.set_color("#B0B0B0")
        axes[row_index, 0].imshow(case.image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_index, 1].imshow(case.image, cmap="gray", vmin=0.0, vmax=1.0)
        q_masked = np.ma.masked_where(case.vote_fraction <= 0, case.vote_fraction)
        axes[row_index, 1].imshow(q_masked, cmap="cividis", vmin=0.0, vmax=1.0, alpha=0.72)
        axes[row_index, 2].imshow(case.image, cmap="gray", vmin=0.0, vmax=1.0)
        _overlay_mask(axes[row_index, 2], case.t2_target, (0.0, 0.62, 0.45), 0.58)
        _draw_contour(axes[row_index, 2], case.t2_target, "white", 0.75)
        axes[row_index, 3].imshow(case.image, cmap="gray", vmin=0.0, vmax=1.0)
        probability_masked = np.ma.masked_where(case.t2_probability <= 0.0, case.t2_probability)
        axes[row_index, 3].imshow(
            probability_masked,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            alpha=0.68,
        )
        _draw_contour(
            axes[row_index, 3],
            case.t2_probability >= case.selected_threshold,
            "white",
            0.8,
        )
        axes[row_index, 0].set_ylabel(
            f"{case.case_id}\nz={case.z_index}, C={case.n_clustered_contours}\ndisagr.={case.disagreement_fraction:.2f}",
            rotation=0,
            ha="right",
            va="center",
            labelpad=5.0,
            fontsize=6.5,
        )

    q_map = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="cividis")
    q_map.set_array([])
    q_bar = fig.colorbar(q_map, ax=axes[:, 1].tolist(), fraction=0.035, pad=0.015)
    q_bar.set_label("Vote fraction", fontsize=6.5)
    q_bar.ax.tick_params(labelsize=6.0, length=2)
    probability_map = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="magma")
    probability_map.set_array([])
    probability_bar = fig.colorbar(
        probability_map,
        ax=axes[:, 3].tolist(),
        fraction=0.035,
        pad=0.015,
    )
    probability_bar.set_label("Probability", fontsize=6.5)
    probability_bar.ax.tick_params(labelsize=6.0, length=2)
    return _save_figure(fig, output_dir, "figure_04_qualitative_candidate_rois", dpi)


def discover_histories(results_root: Path, explicit: Path | None) -> list[Path]:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise InputContractError(f"Requested history file does not exist: {path}")
        return [path]
    candidates: list[Path] = []
    for name in ("history.csv", "history.json", "training_history.csv", "training_history.json"):
        path = results_root / name
        if path.is_file():
            candidates.append(path)
    history_dir = results_root / "histories"
    if history_dir.is_dir():
        candidates.extend(sorted(history_dir.glob("*.csv")))
        candidates.extend(sorted(history_dir.glob("*.json")))
    return sorted(set(path.resolve() for path in candidates))


def load_histories(paths: Sequence[Path], selected_model: str | None) -> pd.DataFrame | None:
    if not paths:
        return None
    rows: list[pd.DataFrame] = []
    for path in paths:
        frame = _load_table(path, "training history")
        frame = _rename_alias(frame, "epoch", ("step", "training_epoch"))
        frame = _rename_alias(frame, "train_loss", ("loss_train", "training_loss"))
        frame = _rename_alias(frame, "val_loss", ("loss_val", "validation_loss", "valid_loss"))
        frame = _rename_alias(frame, "model", ("variant", "configuration"))
        if "epoch" not in frame.columns:
            raise InputContractError(
                f"Training history {path} must contain epoch and train_loss and/or val_loss. "
                "Long form split,loss is also accepted."
            )
        if not {"train_loss", "val_loss"}.intersection(frame.columns):
            if {"split", "loss"}.issubset(frame.columns):
                split = frame["split"].astype(str).str.lower().str.strip()
                frame = frame.assign(split=split)
                identifiers = [column for column in ("epoch", "model", "fold", "run") if column in frame.columns]
                pivot = frame.pivot_table(index=identifiers, columns="split", values="loss", aggfunc="first")
                pivot = pivot.reset_index()
                rename = {}
                for column in pivot.columns:
                    if str(column) in {"train", "training"}:
                        rename[column] = "train_loss"
                    elif str(column) in {"val", "valid", "validation"}:
                        rename[column] = "val_loss"
                frame = pivot.rename(columns=rename)
            if not {"train_loss", "val_loss"}.intersection(frame.columns):
                raise InputContractError(
                    f"Training history {path} must contain train_loss and/or val_loss, or long-form "
                    "split,loss rows using train and val split names."
                )
        if "model" not in frame.columns:
            frame["model"] = selected_model or path.stem
        frame["source_path"] = str(path)
        rows.append(frame)
    history = pd.concat(rows, ignore_index=True, sort=False)
    history["model"] = history["model"].fillna("").astype(str)
    if selected_model is not None and (history["model"] == selected_model).any():
        history = history[history["model"] == selected_model].copy()
    elif history["model"].nunique() > 1:
        raise InputContractError(
            "Discovered histories describe multiple models but none exactly matches --model. "
            f"Pass --history for one file or use an exact model label; found {sorted(history['model'].unique())}."
        )
    try:
        history["epoch"] = pd.to_numeric(history["epoch"], errors="raise")
    except (ValueError, TypeError) as error:
        raise InputContractError("training_history.epoch must contain only numeric values.") from error
    if not np.isfinite(history["epoch"].to_numpy(dtype=float)).all():
        raise InputContractError("training_history.epoch contains a non-finite value.")
    for column in ("train_loss", "val_loss"):
        if column not in history.columns:
            continue
        try:
            history[column] = pd.to_numeric(history[column], errors="raise")
        except (ValueError, TypeError) as error:
            raise InputContractError(
                f"training_history.{column} must contain numeric values or blanks."
            ) from error
        finite = history[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(finite).all():
            raise InputContractError(f"training_history.{column} contains a non-finite value.")
    if (history["epoch"] < 0).any():
        raise InputContractError("training_history.epoch must be non-negative.")
    for column in ("train_loss", "val_loss"):
        if column in history.columns and (history[column].dropna() < 0).any():
            raise InputContractError(f"training_history.{column} must be non-negative.")
    return history


def plot_history(
    frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    fig, axis = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)
    specifications = (("train_loss", BLUE, "Training"), ("val_loss", ORANGE, "Validation"))
    plotted = 0
    for column, color, label in specifications:
        if column not in frame.columns or frame[column].dropna().empty:
            continue
        grouped = frame.dropna(subset=[column]).groupby("epoch", sort=True)[column]
        mean = grouped.mean()
        minimum = grouped.min()
        maximum = grouped.max()
        epochs = mean.index.to_numpy(dtype=float)
        axis.plot(epochs, mean.to_numpy(dtype=float), color=color, label=f"{label} mean")
        if int(grouped.size().max()) > 1:
            axis.fill_between(
                epochs,
                minimum.to_numpy(dtype=float),
                maximum.to_numpy(dtype=float),
                color=color,
                alpha=0.14,
                linewidth=0,
                label=f"{label} fold range",
            )
        plotted += 1
    if plotted == 0:
        raise InputContractError("Training history has no finite train_loss or val_loss values.")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Objective loss")
    axis.set_ylim(bottom=0.0)
    axis.grid(color=LIGHT_GRAY, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    return _save_figure(fig, output_dir, "figure_05_training_curve", dpi)


def _parse_case_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not parsed:
        raise InputContractError("--qualitative-cases must contain at least one case identifier.")
    return parsed


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def write_figure_manifest(records: Sequence[FigureRecord], output_dir: Path) -> Path:
    rows = []
    for record in records:
        rows.append(
            {
                "figure_id": record.figure_id,
                "title": record.title,
                "png_path": _relative_or_absolute(record.png_path, output_dir),
                "pdf_path": _relative_or_absolute(record.pdf_path, output_dir),
                "source_files": ";".join(str(path.resolve()) for path in record.source_files),
                "model": record.model,
                "cases": ";".join(record.cases),
                "selection_rule": record.selection_rule,
                "notes": record.notes,
            }
        )
    path = output_dir / "figure_manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export five data-backed IEEE-style figures from final evaluation tables, v3 arrays, "
            "and held-out prediction NPZs. The training curve is omitted when no history exists."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=Path("data/final_325_nodules_v3"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--ablation-metrics", type=Path, default=None)
    parser.add_argument("--agreement-metrics", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--predictions-dir", type=Path, default=None)
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Exact proposed-model label when result tables contain multiple models",
    )
    parser.add_argument(
        "--qualitative-cases",
        type=str,
        default=None,
        help="Comma-separated case IDs; otherwise qualitative_cases.csv or deterministic selection is used",
    )
    parser.add_argument("--num-qualitative", type=int, default=3)
    parser.add_argument(
        "--qualitative-pool",
        choices=("test", "all"),
        default="test",
        help="Use 'all' only for verified out-of-fold prediction files",
    )
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=600)
    return parser


def run(args: argparse.Namespace) -> Path:
    results_root = args.results_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_root / "paper_figures"
    )
    if args.dpi < 150:
        raise InputContractError("--dpi must be at least 150; use 600 for final publication export.")

    ablation_path = _candidate_table_path(results_root, "ablation_metrics", args.ablation_metrics)
    agreement_path = _candidate_table_path(results_root, "agreement_metrics", args.agreement_metrics)
    calibration_path = _candidate_table_path(results_root, "calibration", args.calibration)
    if not ablation_path.is_file():
        raise InputContractError(
            f"Missing ablation metrics: {ablation_path}. Create CSV/JSON columns "
            "model,patient_macro_t2_dice,ci95_low,ci95_high (one row per ablation; optional "
            "order,is_proposed)."
        )
    if not agreement_path.is_file():
        raise InputContractError(
            f"Missing agreement metrics: {agreement_path}. Create CSV/JSON columns "
            "model,agreement_level,patient_macro_dice,ci95_low,ci95_high with exactly T1-T4 rows."
        )
    if not calibration_path.is_file():
        raise InputContractError(
            f"Missing calibration data: {calibration_path}. Create binned CSV/JSON columns "
            "model,bin_lower,bin_upper,mean_predicted,mean_observed,n_voxels; raw model,q_hat,q "
            "rows are also accepted."
        )

    # Complete all table and qualitative preflight checks before writing figures.
    ablation = load_ablation(ablation_path)
    agreement = load_agreement(agreement_path)
    calibration, calibration_rule = load_calibration(calibration_path, args.calibration_bins)
    selected_model = infer_model(args.model, ablation, agreement, calibration)
    agreement_selected = filter_model(agreement, selected_model, "agreement_metrics")
    calibration_selected = filter_model(calibration, selected_model, "calibration")
    model_label = selected_model or (
        str(agreement_selected.iloc[0]["model"]) if agreement_selected.iloc[0]["model"] else "proposed model"
    )

    manifest, manifest_path = load_dataset_manifest(data_root)
    predictions_dir = (
        args.predictions_dir.expanduser().resolve()
        if args.predictions_dir is not None
        else results_root / "predictions"
    )
    predictions = _prediction_files(predictions_dir)
    requested_cases = _parse_case_list(args.qualitative_cases)
    selection_path: Path | None = None
    if requested_cases is None:
        for suffix in ("csv", "json"):
            candidate = results_root / f"qualitative_cases.{suffix}"
            if candidate.is_file():
                if selection_path is not None:
                    raise InputContractError(
                        "Both qualitative_cases.csv and qualitative_cases.json exist; pass "
                        "--qualitative-cases explicitly or remove the obsolete selection file."
                    )
                selection_path = candidate
    qualitative_ids, selection_rule, used_selection_path = select_qualitative_ids(
        manifest,
        predictions,
        requested_cases,
        selection_path,
        args.num_qualitative,
        args.qualitative_pool,
    )
    qualitative_cases = load_qualitative_cases(
        qualitative_ids,
        manifest,
        data_root,
        predictions,
    )

    history_paths = discover_histories(results_root, args.history)
    history = load_histories(history_paths, selected_model)

    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    records: list[FigureRecord] = []

    png, pdf = plot_ablation(ablation, output_dir, args.dpi, selected_model)
    records.append(
        FigureRecord(
            "Fig. 1",
            "Ablation patient-macro T2 Dice with patient-level 95% confidence intervals",
            png,
            pdf,
            (ablation_path,),
            model_label,
            notes="Ablation display order follows the order column when supplied, otherwise source-row order.",
        )
    )
    png, pdf = plot_agreement(agreement_selected, output_dir, args.dpi)
    records.append(
        FigureRecord(
            "Fig. 2",
            "Agreement-level patient-macro Dice from T1 through T4",
            png,
            pdf,
            (agreement_path,),
            model_label,
            notes="All four support heads are defined; Dice uses the target-positive endpoint.",
        )
    )
    (png, pdf), ece = plot_calibration(calibration_selected, output_dir, args.dpi)
    records.append(
        FigureRecord(
            "Fig. 3",
            "Reader-support reliability curve",
            png,
            pdf,
            (calibration_path,),
            model_label,
            selection_rule=calibration_rule,
            notes=f"Displayed ECE is count-weighted absolute calibration error from plotted bins: {ece:.6f}.",
        )
    )
    png, pdf = plot_qualitative(qualitative_cases, output_dir, args.dpi)
    qualitative_sources = [manifest_path]
    if used_selection_path is not None:
        qualitative_sources.append(used_selection_path)
    qualitative_sources.extend(case.prediction_path for case in qualitative_cases)
    records.append(
        FigureRecord(
            "Fig. 4",
            "Qualitative candidate-ROI reader-agreement and T2 segmentation panels",
            png,
            pdf,
            tuple(qualitative_sources),
            model_label,
            cases=tuple(case.case_id for case in qualitative_cases),
            selection_rule=selection_rule + "; displayed z is the middle tie among slices with maximal T2 area",
            notes="White prediction contour uses each fold's validation-selected threshold stored in the NPZ.",
        )
    )
    if history is not None:
        png, pdf = plot_history(history, output_dir, args.dpi)
        records.append(
            FigureRecord(
                "Fig. 5",
                "Training and validation objective history",
                png,
                pdf,
                tuple(history_paths),
                model_label,
                notes="Lines are epoch means; shading, when present, is the observed fold/run range.",
            )
        )
    manifest_output = write_figure_manifest(records, output_dir)
    print(f"Wrote {len(records)} figure sets and {manifest_output}")
    for record in records:
        print(f"  {record.figure_id}: {record.png_path.name}, {record.pdf_path.name}")
    return manifest_output


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except InputContractError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
