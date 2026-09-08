"""Volume-aware metrics for nested four-session segmentation predictions.

Slice predictions are accumulated by nodule and reconstructed into 3D volumes
before overlap and surface metrics are computed.  All T1--T4 heads are valid;
empty higher-order targets encode lack of cross-reader contour support.  The
primary overlap summaries use target-positive cases, with all-case summaries
under an explicit empty-mask policy retained as secondary results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

try:  # Surface metrics remain optional so the core evaluator needs only NumPy.
    from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in environments without SciPy.
    binary_erosion = distance_transform_edt = generate_binary_structure = None
    SCIPY_AVAILABLE = False


HEAD_NAMES = ("T1", "T2", "T3", "T4")
EMPTY_POLICIES = frozenset({"perfect", "zero", "skip"})


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else np.nan


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    output = np.empty_like(logits)
    positive = logits >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    output[~positive] = exp_logits / (1.0 + exp_logits)
    return output


def _as_list(value: Any, batch_size: int, name: str) -> list[Any]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, torch.Tensor):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        values = [value]
    if len(values) != batch_size:
        raise ValueError(f"{name} has {len(values)} values for batch size {batch_size}")
    return values


def _heads_to_array(value: Any, name: str) -> np.ndarray:
    """Normalize a tensor or ``{T1: ..., ..., T4: ...}`` mapping to Bx4xHxW."""

    if isinstance(value, Mapping):
        missing = [head for head in HEAD_NAMES if head not in value]
        if missing:
            raise ValueError(f"{name} is missing heads: {missing}")
        planes = []
        for head in HEAD_NAMES:
            plane = _to_numpy(value[head])
            if plane.ndim == 2:
                plane = plane[None, ...]
            elif plane.ndim == 4 and plane.shape[1] == 1:
                plane = plane[:, 0]
            if plane.ndim != 3:
                raise ValueError(f"{name}[{head}] must have shape BxHxW or Bx1xHxW")
            planes.append(plane)
        array = np.stack(planes, axis=1)
    else:
        array = _to_numpy(value)
        if array.ndim == 3 and array.shape[0] == 4:
            array = array[None, ...]
        if array.ndim != 4 or array.shape[1] != 4:
            raise ValueError(f"{name} must have shape Bx4xHxW")
    return np.asarray(array)


def _masks_to_array(value: Any, name: str) -> np.ndarray:
    """Normalize Bx1xHxW, BxHxW, or a single HxW mask to BxHxW."""

    array = _to_numpy(value)
    if array.ndim == 2:
        array = array[None, ...]
    elif array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape BxHxW or Bx1xHxW")
    return np.asarray(array)


def _overlap_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    empty_policy: str,
) -> tuple[float, float, float, float, str]:
    """Return Dice, IoU, precision, recall/sensitivity, and empty status.

    Precision is set to zero for an empty prediction against a positive
    target.  Recall is undefined when the target is empty but the prediction
    is not.  When both masks are empty, all four scores follow
    ``empty_policy`` so the secondary all-case result remains explicit.
    """

    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    prediction_positive = bool(prediction.any())
    target_positive = bool(target.any())

    if not prediction_positive and not target_positive:
        if empty_policy == "perfect":
            return 1.0, 1.0, 1.0, 1.0, "both_empty"
        if empty_policy == "zero":
            return 0.0, 0.0, 0.0, 0.0, "both_empty"
        return np.nan, np.nan, np.nan, np.nan, "both_empty"
    if not target_positive:
        return 0.0, 0.0, 0.0, np.nan, "target_empty_prediction_nonempty"
    if not prediction_positive:
        return 0.0, 0.0, 0.0, 0.0, "prediction_empty_target_nonempty"

    intersection = int(np.logical_and(prediction, target).sum())
    prediction_sum = int(prediction.sum())
    target_sum = int(target.sum())
    union = prediction_sum + target_sum - intersection
    dice = 2.0 * intersection / float(prediction_sum + target_sum)
    iou = intersection / float(union)
    precision = intersection / float(prediction_sum)
    recall = intersection / float(target_sum)
    return (
        float(dice),
        float(iou),
        float(precision),
        float(recall),
        "both_nonempty",
    )


def _surface_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
    empty_policy: str,
) -> tuple[float, float, bool]:
    if not SCIPY_AVAILABLE:
        return np.nan, np.nan, False

    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not prediction.any() and not target.any():
        if empty_policy == "perfect":
            return 0.0, 0.0, True
        return np.nan, np.nan, False
    if not prediction.any() or not target.any():
        return np.nan, np.nan, False

    structure = generate_binary_structure(prediction.ndim, 1)
    prediction_surface = prediction & ~binary_erosion(
        prediction, structure=structure, border_value=0
    )
    target_surface = target & ~binary_erosion(target, structure=structure, border_value=0)
    if not prediction_surface.any():
        prediction_surface = prediction
    if not target_surface.any():
        target_surface = target

    distance_to_target = distance_transform_edt(~target_surface, sampling=spacing)
    distance_to_prediction = distance_transform_edt(~prediction_surface, sampling=spacing)
    prediction_to_target = distance_to_target[prediction_surface]
    target_to_prediction = distance_to_prediction[target_surface]
    symmetric = np.concatenate((prediction_to_target, target_to_prediction))
    hd95 = float(np.percentile(symmetric, 95.0))
    assd = float(symmetric.mean())
    return hd95, assd, True


def patient_bootstrap_ci(
    case_metrics: pd.DataFrame,
    metric: str,
    *,
    patient_column: str = "patient_id",
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    """Cluster bootstrap a patient-macro mean confidence interval.

    Nodule values are first averaged within patient.  Patients are then sampled
    with replacement, preserving equal patient weighting even when a patient
    contributes multiple nodules.
    """

    if metric not in case_metrics.columns:
        raise ValueError(f"Metric column not found: {metric}")
    if patient_column not in case_metrics.columns:
        raise ValueError(f"Patient column not found: {patient_column}")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    patient_values = (
        case_metrics[[patient_column, metric]]
        .dropna(subset=[metric])
        .groupby(patient_column, sort=False)[metric]
        .mean()
        .to_numpy(dtype=float)
    )
    if patient_values.size == 0:
        return {
            "estimate": np.nan,
            "lower": np.nan,
            "upper": np.nan,
            "n_patients": 0,
            "n_bootstrap": int(n_bootstrap),
            "confidence": float(confidence),
        }

    rng = np.random.default_rng(seed)
    draws = rng.choice(
        patient_values,
        size=(int(n_bootstrap), patient_values.size),
        replace=True,
    ).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": float(patient_values.mean()),
        "lower": float(np.quantile(draws, alpha)),
        "upper": float(np.quantile(draws, 1.0 - alpha)),
        "n_patients": int(patient_values.size),
        "n_bootstrap": int(n_bootstrap),
        "confidence": float(confidence),
    }


@dataclass
class AgreementMetricReport:
    """Structured output returned by :meth:`AgreementMetricsAccumulator.compute`."""

    case_metrics: pd.DataFrame
    patient_metrics: pd.DataFrame
    all_case_patient_metrics: pd.DataFrame
    soft_case_metrics: pd.DataFrame
    soft_patient_metrics: pd.DataFrame
    target_summary: pd.DataFrame
    overall_summary: dict[str, Any]
    bootstrap_cis: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class _VolumeBuffer:
    patient_id: str
    split: str
    valid_heads: np.ndarray
    n_clustered_contours: int | None
    expected_depth: int | None
    spacing: tuple[float, float, float]
    slices: dict[int, tuple[np.ndarray, ...]] = field(default_factory=dict)


class AgreementMetricsAccumulator:
    """Accumulate central-slice predictions and evaluate reconstructed volumes.

    ``expected_split`` defaults to ``"test"`` and rejects any other split,
    preventing accidental train/validation contamination of final reporting.
    Set it explicitly to ``"val"`` for model selection or to ``None`` only for
    a deliberate mixed-split diagnostic.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        expected_split: str | Sequence[str] | None = "test",
        empty_policy: str = "perfect",
        voxel_spacing: Sequence[float] = (1.0, 1.0, 1.0),
        nesting_tolerance: float = 1e-6,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if empty_policy not in EMPTY_POLICIES:
            raise ValueError(f"empty_policy must be one of {sorted(EMPTY_POLICIES)}")
        spacing = tuple(float(value) for value in voxel_spacing)
        if len(spacing) != 3 or any(value <= 0 for value in spacing):
            raise ValueError("voxel_spacing must contain three positive values")

        if expected_split is None:
            allowed_splits = None
        elif isinstance(expected_split, str):
            allowed_splits = frozenset({expected_split})
        else:
            allowed_splits = frozenset(str(value) for value in expected_split)
            if not allowed_splits:
                raise ValueError("expected_split cannot be an empty sequence")

        self.threshold = float(threshold)
        self.allowed_splits = allowed_splits
        self.empty_policy = empty_policy
        self.voxel_spacing = spacing
        self.nesting_tolerance = float(nesting_tolerance)
        self._volumes: dict[str, _VolumeBuffer] = {}

    def reset(self) -> None:
        self._volumes.clear()

    def __len__(self) -> int:
        return sum(len(buffer.slices) for buffer in self._volumes.values())

    @property
    def n_volumes(self) -> int:
        return len(self._volumes)

    def update(
        self,
        *,
        volume_id: Any,
        patient_id: Any,
        split: Any,
        z: Any,
        head_predictions: Any,
        targets: Any,
        valid_heads: Any,
        vote_fraction: Any,
        majority: Any,
        n_clustered_contours: Any | None = None,
        depth: Any | None = None,
        from_logits: bool = True,
        q_prediction: Any | None = None,
        q_from_logits: bool = False,
    ) -> None:
        """Add one evaluation batch.

        ``head_predictions`` and ``targets`` may be Bx4xHxW tensors or mappings
        with keys ``T1`` through ``T4``.  If no separate ``q_prediction`` is
        supplied, the expected reader vote fraction is derived as the sum of
        all four nested-head probabilities divided by four review sessions.
        """

        head_array = _heads_to_array(head_predictions, "head_predictions")
        target_raw = _heads_to_array(targets, "targets")
        if not np.isfinite(target_raw).all() or np.any(
            (target_raw != 0) & (target_raw != 1)
        ):
            raise ValueError("targets must be finite binary values")
        target_array = target_raw.astype(bool)
        if head_array.shape != target_array.shape:
            raise ValueError(
                f"Prediction shape {head_array.shape} != target shape {target_array.shape}"
            )
        batch_size, _, height, width = head_array.shape
        probabilities = _sigmoid(head_array) if from_logits else head_array.astype(np.float32)
        if not np.isfinite(probabilities).all():
            raise ValueError("Head probabilities contain non-finite values")
        if probabilities.min() < 0 or probabilities.max() > 1:
            raise ValueError("Head probabilities must be in [0, 1]")

        valid_array = _to_numpy(valid_heads).astype(bool)
        if valid_array.ndim == 1 and valid_array.shape[0] == 4:
            valid_array = valid_array[None, ...]
        if valid_array.shape != (batch_size, 4):
            raise ValueError(f"valid_heads must have shape Bx4, got {valid_array.shape}")
        if not valid_array.all():
            raise ValueError(
                "All four agreement heads must be valid under four-session semantics"
            )

        q_target = _masks_to_array(vote_fraction, "vote_fraction").astype(np.float32)
        majority_raw = _masks_to_array(majority, "majority")
        if not np.isfinite(majority_raw).all() or np.any(
            (majority_raw != 0) & (majority_raw != 1)
        ):
            raise ValueError("majority must contain finite binary values")
        majority_target = majority_raw.astype(bool)
        if q_target.shape != (batch_size, height, width):
            raise ValueError("vote_fraction spatial shape does not match predictions")
        if majority_target.shape != (batch_size, height, width):
            raise ValueError("majority spatial shape does not match predictions")
        if not np.isfinite(q_target).all() or q_target.min() < 0 or q_target.max() > 1:
            raise ValueError("vote_fraction must be finite in [0, 1]")

        if q_prediction is None:
            q_pred = probabilities.sum(axis=1) / 4.0
        else:
            q_raw = _masks_to_array(q_prediction, "q_prediction")
            q_pred = _sigmoid(q_raw) if q_from_logits else q_raw.astype(np.float32)
            if q_pred.shape != (batch_size, height, width):
                raise ValueError("q_prediction spatial shape does not match predictions")
            if not np.isfinite(q_pred).all() or q_pred.min() < 0 or q_pred.max() > 1:
                raise ValueError("q_prediction must be finite in [0, 1]")

        if np.any(target_array[:, 1:] & ~target_array[:, :-1]):
            raise ValueError("Ground-truth agreement heads are not nested")
        expected_q_target = target_array.astype(np.float32).sum(axis=1) / 4.0
        if not np.allclose(q_target, expected_q_target, atol=1e-6):
            raise ValueError("vote_fraction must equal sum(T1..T4) / 4")
        if not np.array_equal(target_array[:, 1], majority_target):
            raise ValueError("majority target must equal T2 under four-session semantics")

        volume_ids = [str(value) for value in _as_list(volume_id, batch_size, "volume_id")]
        patient_ids = [str(value) for value in _as_list(patient_id, batch_size, "patient_id")]
        splits = [str(value) for value in _as_list(split, batch_size, "split")]
        z_values = [int(value) for value in _as_list(z, batch_size, "z")]
        depth_values = (
            [None] * batch_size
            if depth is None
            else [int(value) for value in _as_list(depth, batch_size, "depth")]
        )
        contour_counts = (
            [None] * batch_size
            if n_clustered_contours is None
            else [
                int(value)
                for value in _as_list(
                    n_clustered_contours,
                    batch_size,
                    "n_clustered_contours",
                )
            ]
        )

        for batch_index in range(batch_size):
            split_value = splits[batch_index]
            if self.allowed_splits is not None and split_value not in self.allowed_splits:
                raise ValueError(
                    f"Refusing split '{split_value}'; expected {sorted(self.allowed_splits)}"
                )
            volume_key = volume_ids[batch_index]
            expected_depth = depth_values[batch_index]
            contour_count = contour_counts[batch_index]
            if contour_count is not None and not 1 <= contour_count <= 4:
                raise ValueError("n_clustered_contours must be between 1 and 4")
            if expected_depth is not None and expected_depth <= 0:
                raise ValueError("depth must be positive")
            buffer = self._volumes.get(volume_key)
            if buffer is None:
                buffer = _VolumeBuffer(
                    patient_id=patient_ids[batch_index],
                    split=split_value,
                    valid_heads=valid_array[batch_index].copy(),
                    n_clustered_contours=contour_count,
                    expected_depth=expected_depth,
                    spacing=self.voxel_spacing,
                )
                self._volumes[volume_key] = buffer
            else:
                if buffer.patient_id != patient_ids[batch_index] or buffer.split != split_value:
                    raise ValueError(f"Inconsistent metadata for volume {volume_key}")
                if not np.array_equal(buffer.valid_heads, valid_array[batch_index]):
                    raise ValueError(f"valid_heads changed within volume {volume_key}")
                if (
                    buffer.n_clustered_contours is not None
                    and contour_count is not None
                    and buffer.n_clustered_contours != contour_count
                ):
                    raise ValueError(
                        f"n_clustered_contours changed within volume {volume_key}"
                    )
                if buffer.n_clustered_contours is None:
                    buffer.n_clustered_contours = contour_count
                if (
                    buffer.expected_depth is not None
                    and expected_depth is not None
                    and buffer.expected_depth != expected_depth
                ):
                    raise ValueError(f"depth changed within volume {volume_key}")
                if buffer.expected_depth is None:
                    buffer.expected_depth = expected_depth

            slice_index = z_values[batch_index]
            if slice_index < 0:
                raise ValueError("z must be non-negative")
            if slice_index in buffer.slices:
                raise ValueError(f"Duplicate prediction for {volume_key} z={slice_index}")
            buffer.slices[slice_index] = (
                probabilities[batch_index].astype(np.float32, copy=True),
                target_array[batch_index].copy(),
                q_pred[batch_index].astype(np.float32, copy=True),
                q_target[batch_index].astype(np.float32, copy=True),
                majority_target[batch_index].copy(),
            )

    def update_from_batch(
        self,
        batch: Mapping[str, Any],
        head_predictions: Any,
        *,
        from_logits: bool = True,
        q_prediction: Any | None = None,
        q_from_logits: bool = False,
    ) -> None:
        """Convenience wrapper for batches from ``AgreementSliceDataset``."""

        self.update(
            volume_id=batch["volume_id"],
            patient_id=batch["patient_id"],
            split=batch["split"],
            z=batch["z"],
            depth=batch.get("depth"),
            head_predictions=head_predictions,
            targets=batch.get("targets", batch.get("nested_targets")),
            valid_heads=batch["valid_heads"],
            vote_fraction=batch.get("vote_fraction", batch.get("q")),
            majority=batch["majority"],
            n_clustered_contours=batch.get("n_clustered_contours"),
            from_logits=from_logits,
            q_prediction=q_prediction,
            q_from_logits=q_from_logits,
        )

    def update_volume_batch(
        self,
        *,
        volume_id: Any,
        patient_id: Any,
        split: Any,
        head_predictions: Any,
        targets: Any,
        valid_heads: Any,
        vote_fraction: Any,
        majority: Any,
        n_clustered_contours: Any | None = None,
        depth: Any | None = None,
        depth_valid: Any | None = None,
        from_logits: bool = True,
        q_prediction: Any | None = None,
        q_from_logits: bool = False,
    ) -> None:
        """Add padded full-volume predictions shaped ``[B,4,D,H,W]``.

        ``depth_valid`` marks the real contiguous depth prefix in each padded
        item.  When it is omitted, ``depth`` supplies each true length; when
        both are omitted every padded position is treated as real.  Each real
        slice is delegated to :meth:`update`, preserving one validation and
        reconstruction path for slice-wise and true-volume inference.
        """

        predictions = _to_numpy(head_predictions)
        target_array = _to_numpy(targets)
        if isinstance(head_predictions, Mapping):
            prediction_planes = []
            for head in HEAD_NAMES:
                if head not in head_predictions:
                    raise ValueError(f"head_predictions is missing head {head}")
                plane = _to_numpy(head_predictions[head])
                if plane.ndim == 4:
                    plane = plane[:, None]
                elif plane.ndim == 5 and plane.shape[1] == 1:
                    pass
                else:
                    raise ValueError(
                        f"head_predictions[{head}] must have shape BxDxHxW "
                        "or Bx1xDxHxW"
                    )
                prediction_planes.append(plane)
            predictions = np.concatenate(prediction_planes, axis=1)
        if isinstance(targets, Mapping):
            target_planes = []
            for head in HEAD_NAMES:
                if head not in targets:
                    raise ValueError(f"targets is missing head {head}")
                plane = _to_numpy(targets[head])
                if plane.ndim == 4:
                    plane = plane[:, None]
                elif plane.ndim == 5 and plane.shape[1] == 1:
                    pass
                else:
                    raise ValueError(
                        f"targets[{head}] must have shape BxDxHxW or Bx1xDxHxW"
                    )
                target_planes.append(plane)
            target_array = np.concatenate(target_planes, axis=1)
        if predictions.ndim != 5 or predictions.shape[1] != 4:
            raise ValueError("head_predictions must have shape Bx4xDxHxW")
        if target_array.shape != predictions.shape:
            raise ValueError(
                f"Prediction shape {predictions.shape} != target shape {target_array.shape}"
            )

        batch_size, _, padded_depth, height, width = predictions.shape
        volume_ids = _as_list(volume_id, batch_size, "volume_id")
        patient_ids = _as_list(patient_id, batch_size, "patient_id")
        splits = _as_list(split, batch_size, "split")
        valid_array = _to_numpy(valid_heads)
        if valid_array.shape != (batch_size, 4):
            raise ValueError(f"valid_heads must have shape Bx4, got {valid_array.shape}")

        contour_counts = (
            [None] * batch_size
            if n_clustered_contours is None
            else [
                int(value)
                for value in _as_list(
                    n_clustered_contours,
                    batch_size,
                    "n_clustered_contours",
                )
            ]
        )

        if depth is None:
            depth_values: list[int] | None = None
        else:
            depth_values = [int(value) for value in _as_list(depth, batch_size, "depth")]
        if depth_valid is None:
            if depth_values is None:
                depth_mask = np.ones((batch_size, padded_depth), dtype=bool)
            else:
                depth_mask = np.arange(padded_depth)[None, :] < np.asarray(depth_values)[:, None]
        else:
            depth_mask = _to_numpy(depth_valid).astype(bool)
            if depth_mask.shape != (batch_size, padded_depth):
                raise ValueError(
                    "depth_valid must have shape "
                    f"[{batch_size}, {padded_depth}], got {depth_mask.shape}"
                )
            lengths = depth_mask.sum(axis=1)
            expected_mask = np.arange(padded_depth)[None, :] < lengths[:, None]
            if not np.array_equal(depth_mask, expected_mask):
                raise ValueError("depth_valid must mark a contiguous valid prefix")
            if depth_values is None:
                depth_values = lengths.astype(int).tolist()
            elif not np.array_equal(np.asarray(depth_values), lengths):
                raise ValueError("depth and depth_valid specify different true lengths")
        if depth_values is None:
            depth_values = [padded_depth] * batch_size
        if any(value < 1 or value > padded_depth for value in depth_values):
            raise ValueError("Every depth must be between 1 and the padded depth")

        q_target = _to_numpy(vote_fraction)
        if q_target.ndim == 5 and q_target.shape[1] == 1:
            q_target = q_target[:, 0]
        majority_target = _to_numpy(majority)
        if majority_target.ndim == 5 and majority_target.shape[1] == 1:
            majority_target = majority_target[:, 0]
        expected_mask_shape = (batch_size, padded_depth, height, width)
        if q_target.shape != expected_mask_shape:
            raise ValueError(
                f"vote_fraction must have shape {expected_mask_shape}, got {q_target.shape}"
            )
        if majority_target.shape != expected_mask_shape:
            raise ValueError(
                f"majority must have shape {expected_mask_shape}, got {majority_target.shape}"
            )

        if q_prediction is None:
            q_predictions = None
        else:
            q_predictions = _to_numpy(q_prediction)
            if q_predictions.ndim == 5 and q_predictions.shape[1] == 1:
                q_predictions = q_predictions[:, 0]
            if q_predictions.shape != expected_mask_shape:
                raise ValueError(
                    f"q_prediction must have shape {expected_mask_shape}, "
                    f"got {q_predictions.shape}"
                )

        # Flatten only real slices; ``update`` will reconstruct and require a
        # complete z=0..depth-1 sequence before metrics are reported.
        flat_predictions = []
        flat_targets = []
        flat_valid = []
        flat_q_target = []
        flat_majority = []
        flat_q_prediction = []
        flat_volume_ids = []
        flat_patient_ids = []
        flat_splits = []
        flat_z = []
        flat_depth = []
        flat_contour_counts = []
        for batch_index, true_depth in enumerate(depth_values):
            for z_index in range(true_depth):
                flat_predictions.append(predictions[batch_index, :, z_index])
                flat_targets.append(target_array[batch_index, :, z_index])
                flat_valid.append(valid_array[batch_index])
                flat_q_target.append(q_target[batch_index, z_index])
                flat_majority.append(majority_target[batch_index, z_index])
                if q_predictions is not None:
                    flat_q_prediction.append(q_predictions[batch_index, z_index])
                flat_volume_ids.append(volume_ids[batch_index])
                flat_patient_ids.append(patient_ids[batch_index])
                flat_splits.append(splits[batch_index])
                flat_z.append(z_index)
                flat_depth.append(true_depth)
                flat_contour_counts.append(contour_counts[batch_index])

        self.update(
            volume_id=flat_volume_ids,
            patient_id=flat_patient_ids,
            split=flat_splits,
            z=flat_z,
            depth=flat_depth,
            head_predictions=np.stack(flat_predictions),
            targets=np.stack(flat_targets),
            valid_heads=np.stack(flat_valid),
            vote_fraction=np.stack(flat_q_target),
            majority=np.stack(flat_majority),
            n_clustered_contours=(
                None
                if n_clustered_contours is None
                else flat_contour_counts
            ),
            from_logits=from_logits,
            q_prediction=(
                None if q_predictions is None else np.stack(flat_q_prediction)
            ),
            q_from_logits=q_from_logits,
        )

    def update_from_volume_batch(
        self,
        batch: Mapping[str, Any],
        head_predictions: Any,
        *,
        from_logits: bool = True,
        q_prediction: Any | None = None,
        q_from_logits: bool = False,
    ) -> None:
        """Convenience wrapper for :func:`pad_agreement_volumes` batches."""

        self.update_volume_batch(
            volume_id=batch["volume_id"],
            patient_id=batch["patient_id"],
            split=batch["split"],
            depth=batch.get("depth"),
            depth_valid=batch.get("depth_valid"),
            head_predictions=head_predictions,
            targets=batch.get("targets", batch.get("nested_targets")),
            valid_heads=batch["valid_heads"],
            vote_fraction=batch.get("vote_fraction", batch.get("q")),
            majority=batch["majority"],
            n_clustered_contours=batch.get("n_clustered_contours"),
            from_logits=from_logits,
            q_prediction=q_prediction,
            q_from_logits=q_from_logits,
        )

    def _stack_volume(self, volume_id: str, buffer: _VolumeBuffer) -> tuple[np.ndarray, ...]:
        z_values = sorted(buffer.slices)
        expected_depth = buffer.expected_depth or (z_values[-1] + 1)
        expected_z = list(range(expected_depth))
        if z_values != expected_z:
            missing = sorted(set(expected_z) - set(z_values))
            extra = sorted(set(z_values) - set(expected_z))
            raise ValueError(
                f"Incomplete volume {volume_id}: missing z={missing[:10]}, extra z={extra[:10]}"
            )
        fields = list(zip(*(buffer.slices[z] for z in z_values)))
        return tuple(
            np.stack(field, axis=1 if index < 2 else 0)
            for index, field in enumerate(fields)
        )

    def compute(
        self,
        *,
        n_bootstrap: int = 0,
        confidence: float = 0.95,
        bootstrap_seed: int = 42,
    ) -> AgreementMetricReport:
        if not self._volumes:
            raise ValueError("No predictions have been accumulated")

        case_rows: list[dict[str, Any]] = []
        soft_rows: list[dict[str, Any]] = []
        for volume_id, buffer in sorted(self._volumes.items()):
            head_probs, head_targets, q_pred, q_target, majority_target = self._stack_volume(
                volume_id, buffer
            )
            majority_head = 1

            for head_index, head_name in enumerate(HEAD_NAMES):
                prediction = head_probs[head_index] >= self.threshold
                target = head_targets[head_index]
                dice, iou, precision, recall, empty_status = _overlap_metrics(
                    prediction, target, self.empty_policy
                )
                hd95, assd, surface_defined = _surface_metrics(
                    prediction, target, buffer.spacing, self.empty_policy
                )
                case_rows.append(
                    {
                        "target": head_name,
                        "volume_id": volume_id,
                        "patient_id": buffer.patient_id,
                        "split": buffer.split,
                        "n_clustered_contours": buffer.n_clustered_contours,
                        "target_positive": bool(target.any()),
                        "prediction_positive": bool(prediction.any()),
                        "target_voxels": int(target.sum()),
                        "prediction_voxels": int(prediction.sum()),
                        "empty_status": empty_status,
                        "dice": dice,
                        "iou": iou,
                        "precision": precision,
                        "recall": recall,
                        "sensitivity": recall,
                        "hd95": hd95,
                        "assd": assd,
                        "surface_defined": surface_defined,
                    }
                )

            majority_prediction = head_probs[majority_head] >= self.threshold
            dice, iou, precision, recall, empty_status = _overlap_metrics(
                majority_prediction, majority_target, self.empty_policy
            )
            hd95, assd, surface_defined = _surface_metrics(
                majority_prediction, majority_target, buffer.spacing, self.empty_policy
            )
            case_rows.append(
                {
                    "target": "majority",
                    "volume_id": volume_id,
                    "patient_id": buffer.patient_id,
                    "split": buffer.split,
                    "n_clustered_contours": buffer.n_clustered_contours,
                    "target_positive": bool(majority_target.any()),
                    "prediction_positive": bool(majority_prediction.any()),
                    "target_voxels": int(majority_target.sum()),
                    "prediction_voxels": int(majority_prediction.sum()),
                    "empty_status": empty_status,
                    "dice": dice,
                    "iou": iou,
                    "precision": precision,
                    "recall": recall,
                    "sensitivity": recall,
                    "hd95": hd95,
                    "assd": assd,
                    "surface_defined": surface_defined,
                }
            )

            error = np.square(q_pred - q_target)
            foreground = q_target > 0
            background = ~foreground
            foreground_brier = float(error[foreground].mean()) if foreground.any() else np.nan
            background_brier = float(error[background].mean()) if background.any() else np.nan
            balanced_components = [
                value for value in (foreground_brier, background_brier) if np.isfinite(value)
            ]
            adjacent_violations = []
            for head_index in range(3):
                adjacent_violations.append(
                    head_probs[head_index + 1]
                    > head_probs[head_index] + self.nesting_tolerance
                )
            nesting_violation = (
                float(np.stack(adjacent_violations).mean())
                if adjacent_violations
                else 0.0
            )
            soft_rows.append(
                {
                    "volume_id": volume_id,
                    "patient_id": buffer.patient_id,
                    "split": buffer.split,
                    "n_clustered_contours": buffer.n_clustered_contours,
                    "soft_brier": float(error.mean()),
                    "soft_brier_foreground": foreground_brier,
                    "soft_brier_background": background_brier,
                    "soft_brier_balanced": float(np.mean(balanced_components)),
                    "nesting_violation_fraction": nesting_violation,
                }
            )

        case_metrics = pd.DataFrame(case_rows)
        soft_case_metrics = pd.DataFrame(soft_rows)
        numeric_metrics = [
            "dice",
            "iou",
            "precision",
            "recall",
            "sensitivity",
            "hd95",
            "assd",
        ]
        target_positive_case_metrics = case_metrics[case_metrics["target_positive"]]
        patient_metrics = (
            target_positive_case_metrics.groupby(
                ["target", "patient_id"], as_index=False, sort=True
            )[numeric_metrics]
            .mean()
        )
        all_case_patient_metrics = (
            case_metrics.groupby(
                ["target", "patient_id"], as_index=False, sort=True
            )[numeric_metrics]
            .mean()
        )
        soft_columns = [
            "soft_brier",
            "soft_brier_foreground",
            "soft_brier_background",
            "soft_brier_balanced",
            "nesting_violation_fraction",
        ]
        soft_patient_metrics = (
            soft_case_metrics.groupby("patient_id", as_index=False, sort=True)[soft_columns]
            .mean()
        )

        summary_rows = []
        for target_name, group in case_metrics.groupby("target", sort=True):
            positive_group = group[group["target_positive"]]
            patient_group = patient_metrics[patient_metrics["target"] == target_name]
            all_patient_group = all_case_patient_metrics[
                all_case_patient_metrics["target"] == target_name
            ]
            status_counts = group["empty_status"].value_counts()
            target_present = group["target_positive"].to_numpy(dtype=bool)
            prediction_present = group["prediction_positive"].to_numpy(dtype=bool)
            presence_tp = int(np.logical_and(target_present, prediction_present).sum())
            presence_tn = int(
                np.logical_and(~target_present, ~prediction_present).sum()
            )
            presence_fp = int(
                np.logical_and(~target_present, prediction_present).sum()
            )
            presence_fn = int(
                np.logical_and(target_present, ~prediction_present).sum()
            )
            summary_rows.append(
                {
                    "target": target_name,
                    "primary_scope": "target_positive",
                    "n_cases": int(group["volume_id"].nunique()),
                    "n_patients": int(group["patient_id"].nunique()),
                    "n_target_positive_cases": int(
                        positive_group["volume_id"].nunique()
                    ),
                    "n_target_positive_patients": int(
                        positive_group["patient_id"].nunique()
                    ),
                    "both_empty_cases": int(status_counts.get("both_empty", 0)),
                    "target_empty_prediction_nonempty_cases": int(
                        status_counts.get("target_empty_prediction_nonempty", 0)
                    ),
                    "prediction_empty_target_nonempty_cases": int(
                        status_counts.get("prediction_empty_target_nonempty", 0)
                    ),
                    "case_macro_dice": float(positive_group["dice"].mean()),
                    "case_macro_iou": float(positive_group["iou"].mean()),
                    "case_macro_precision": float(
                        positive_group["precision"].mean()
                    ),
                    "case_macro_recall": float(positive_group["recall"].mean()),
                    "case_macro_sensitivity": float(
                        positive_group["sensitivity"].mean()
                    ),
                    "patient_macro_dice": float(patient_group["dice"].mean()),
                    "patient_macro_iou": float(patient_group["iou"].mean()),
                    "patient_macro_precision": float(
                        patient_group["precision"].mean()
                    ),
                    "patient_macro_recall": float(patient_group["recall"].mean()),
                    "patient_macro_sensitivity": float(
                        patient_group["sensitivity"].mean()
                    ),
                    "case_macro_hd95": float(positive_group["hd95"].mean()),
                    "case_macro_assd": float(positive_group["assd"].mean()),
                    "surface_defined_target_positive_cases": int(
                        positive_group["surface_defined"].sum()
                    ),
                    "all_case_macro_dice": float(group["dice"].mean()),
                    "all_case_macro_iou": float(group["iou"].mean()),
                    "all_case_macro_precision": float(group["precision"].mean()),
                    "all_case_macro_recall": float(group["recall"].mean()),
                    "all_case_macro_sensitivity": float(
                        group["sensitivity"].mean()
                    ),
                    "all_patient_macro_dice": float(
                        all_patient_group["dice"].mean()
                    ),
                    "all_patient_macro_iou": float(
                        all_patient_group["iou"].mean()
                    ),
                    "all_patient_macro_precision": float(
                        all_patient_group["precision"].mean()
                    ),
                    "all_patient_macro_recall": float(
                        all_patient_group["recall"].mean()
                    ),
                    "all_patient_macro_sensitivity": float(
                        all_patient_group["sensitivity"].mean()
                    ),
                    "all_case_macro_hd95": float(group["hd95"].mean()),
                    "all_case_macro_assd": float(group["assd"].mean()),
                    "presence_tp": presence_tp,
                    "presence_tn": presence_tn,
                    "presence_fp": presence_fp,
                    "presence_fn": presence_fn,
                    "presence_sensitivity": _safe_ratio(
                        presence_tp, presence_tp + presence_fn
                    ),
                    "presence_specificity": _safe_ratio(
                        presence_tn, presence_tn + presence_fp
                    ),
                    "presence_precision": _safe_ratio(
                        presence_tp, presence_tp + presence_fp
                    ),
                    "presence_f1": _safe_ratio(
                        2 * presence_tp,
                        2 * presence_tp + presence_fp + presence_fn,
                    ),
                }
            )
        target_summary = pd.DataFrame(summary_rows)
        majority_summary = target_summary[target_summary["target"] == "majority"].iloc[0]
        overall_summary = {
            "split": sorted({buffer.split for buffer in self._volumes.values()}),
            "n_cases": int(len(self._volumes)),
            "n_patients": int(
                len({buffer.patient_id for buffer in self._volumes.values()})
            ),
            "threshold": self.threshold,
            "primary_overlap_scope": "target_positive_cases",
            "secondary_overlap_scope": "all_cases",
            "empty_policy": self.empty_policy,
            "surface_metrics_available": SCIPY_AVAILABLE,
            "surface_distance_units": (
                "voxels" if self.voxel_spacing == (1.0, 1.0, 1.0) else "spacing units"
            ),
            "majority_case_macro_3d_dice": float(majority_summary["case_macro_dice"]),
            "majority_case_macro_3d_iou": float(majority_summary["case_macro_iou"]),
            "majority_case_macro_3d_precision": float(
                majority_summary["case_macro_precision"]
            ),
            "majority_case_macro_3d_recall": float(
                majority_summary["case_macro_recall"]
            ),
            "majority_patient_macro_3d_dice": float(
                majority_summary["patient_macro_dice"]
            ),
            "majority_patient_macro_3d_iou": float(
                majority_summary["patient_macro_iou"]
            ),
            "majority_patient_macro_3d_precision": float(
                majority_summary["patient_macro_precision"]
            ),
            "majority_patient_macro_3d_recall": float(
                majority_summary["patient_macro_recall"]
            ),
            "majority_all_case_macro_3d_dice": float(
                majority_summary["all_case_macro_dice"]
            ),
            "majority_all_case_macro_3d_iou": float(
                majority_summary["all_case_macro_iou"]
            ),
            "majority_all_case_macro_3d_precision": float(
                majority_summary["all_case_macro_precision"]
            ),
            "majority_all_case_macro_3d_recall": float(
                majority_summary["all_case_macro_recall"]
            ),
            "majority_all_patient_macro_3d_dice": float(
                majority_summary["all_patient_macro_dice"]
            ),
            "majority_all_patient_macro_3d_iou": float(
                majority_summary["all_patient_macro_iou"]
            ),
            "majority_all_patient_macro_3d_precision": float(
                majority_summary["all_patient_macro_precision"]
            ),
            "majority_all_patient_macro_3d_recall": float(
                majority_summary["all_patient_macro_recall"]
            ),
            "majority_presence_tp": int(majority_summary["presence_tp"]),
            "majority_presence_tn": int(majority_summary["presence_tn"]),
            "majority_presence_fp": int(majority_summary["presence_fp"]),
            "majority_presence_fn": int(majority_summary["presence_fn"]),
            "majority_presence_sensitivity": float(
                majority_summary["presence_sensitivity"]
            ),
            "majority_presence_specificity": float(
                majority_summary["presence_specificity"]
            ),
            "majority_presence_precision": float(
                majority_summary["presence_precision"]
            ),
            "majority_presence_f1": float(majority_summary["presence_f1"]),
            "case_macro_soft_brier": float(soft_case_metrics["soft_brier"].mean()),
            "patient_macro_soft_brier": float(
                soft_patient_metrics["soft_brier"].mean()
            ),
            "case_macro_soft_brier_balanced": float(
                soft_case_metrics["soft_brier_balanced"].mean()
            ),
            "patient_macro_soft_brier_balanced": float(
                soft_patient_metrics["soft_brier_balanced"].mean()
            ),
            "case_macro_nesting_violation_fraction": float(
                soft_case_metrics["nesting_violation_fraction"].mean()
            ),
        }

        ci_rows = []
        if n_bootstrap:
            for target_name, group in case_metrics.groupby("target", sort=True):
                scoped_groups = {
                    "target_positive": group[group["target_positive"]],
                    "all_cases": group,
                }
                for scope, scoped_group in scoped_groups.items():
                    for metric in ("dice", "iou", "precision", "recall"):
                        result = patient_bootstrap_ci(
                            scoped_group,
                            metric,
                            n_bootstrap=n_bootstrap,
                            confidence=confidence,
                            seed=bootstrap_seed,
                        )
                        ci_rows.append(
                            {
                                "target": target_name,
                                "scope": scope,
                                "metric": metric,
                                **result,
                            }
                        )
            for metric in ("soft_brier", "soft_brier_balanced"):
                result = patient_bootstrap_ci(
                    soft_case_metrics,
                    metric,
                    n_bootstrap=n_bootstrap,
                    confidence=confidence,
                    seed=bootstrap_seed,
                )
                ci_rows.append(
                    {
                        "target": "soft",
                        "scope": "all_cases",
                        "metric": metric,
                        **result,
                    }
                )

        return AgreementMetricReport(
            case_metrics=case_metrics,
            patient_metrics=patient_metrics,
            all_case_patient_metrics=all_case_patient_metrics,
            soft_case_metrics=soft_case_metrics,
            soft_patient_metrics=soft_patient_metrics,
            target_summary=target_summary,
            overall_summary=overall_summary,
            bootstrap_cis=pd.DataFrame(ci_rows),
        )


__all__ = [
    "AgreementMetricReport",
    "AgreementMetricsAccumulator",
    "EMPTY_POLICIES",
    "HEAD_NAMES",
    "SCIPY_AVAILABLE",
    "patient_bootstrap_ci",
]
