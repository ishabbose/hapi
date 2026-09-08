"""Targets and losses for nested multi-reader agreement segmentation.

The preferred loss API uses volume tensors so Dice is computed over a complete
case. Slice tensors from ``AgreementSliceDataset`` are accepted as a
compatibility mode and treated as depth-one volumes. LIDC-IDRI's four review
sessions are all meaningful: an all-zero mask means that session supplied no
clustered contour, not that the session is unavailable. Therefore all T1--T4
heads are valid and vote fractions always use a denominator of four.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


NUMBER_OF_HEADS = 4


def _validate_depth_valid(
    depth_valid: torch.Tensor | None,
    *,
    batch_size: int,
    depth: int,
    device: torch.device,
) -> torch.Tensor:
    if depth_valid is None:
        return torch.ones((batch_size, depth), dtype=torch.bool, device=device)
    if depth_valid.shape != (batch_size, depth):
        raise ValueError(
            "depth_valid must have shape "
            f"[{batch_size}, {depth}], got {list(depth_valid.shape)}"
        )
    result = depth_valid.to(device=device, dtype=torch.bool)
    if not torch.all(result.any(dim=1)):
        raise ValueError("Every case must contain at least one valid slice.")
    return result


def build_nested_agreement_targets(
    reader_masks: torch.Tensor,
    reader_contour_present: torch.Tensor | None = None,
    depth_valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build fixed four-session T1--T4 targets from reader-mask volumes.

    Parameters
    ----------
    reader_masks:
        Binary or probabilistic masks shaped ``[B, 4, D, H, W]``.  Values are
        thresholded at ``0.5``.
    reader_contour_present:
        Optional ``[B, 4]`` audit matrix indicating which review sessions
        produced non-empty clustered contours. It is checked against the
        masks but does not change head validity or the vote denominator.
    depth_valid:
        Optional ``[B, D]`` boolean matrix for padded variable-depth batches.

    Returns
    -------
    dict
        ``targets`` contains fixed >=k-of-four masks; ``valid_heads`` is true
        for every review-session head; ``vote_fraction`` is ``vote_count / 4``;
        and contour-presence counts are included for auditability.
    """

    slice_batch = reader_masks.ndim == 4
    if slice_batch:
        if depth_valid is not None:
            raise ValueError("depth_valid is only accepted with volume reader masks.")
        reader_masks = reader_masks.unsqueeze(2)
    if reader_masks.ndim != 5:
        raise ValueError(
            "reader_masks must have shape [B, 4, H, W] or [B, 4, D, H, W], "
            f"got {list(reader_masks.shape)}"
        )
    batch_size, readers, depth, _, _ = reader_masks.shape
    if readers != NUMBER_OF_HEADS:
        raise ValueError(f"Expected four reader slots, got {readers}.")
    if not torch.isfinite(reader_masks).all():
        raise ValueError("reader_masks contains NaN or infinite values.")

    valid_depth = _validate_depth_valid(
        depth_valid,
        batch_size=batch_size,
        depth=depth,
        device=reader_masks.device,
    )
    binary = reader_masks > 0.5
    valid_voxels = valid_depth[:, None, :, None, None]
    binary = binary & valid_voxels

    inferred_contour_present = binary.flatten(start_dim=2).any(dim=2)
    if reader_contour_present is None:
        contour_present = inferred_contour_present
    else:
        if reader_contour_present.shape != (batch_size, readers):
            raise ValueError(
                "reader_contour_present must have shape "
                f"[{batch_size}, {readers}], got {list(reader_contour_present.shape)}"
            )
        contour_present = reader_contour_present.to(
            device=reader_masks.device,
            dtype=torch.bool,
        )
        if not torch.equal(contour_present, inferred_contour_present):
            raise ValueError(
                "reader_contour_present does not match the non-empty reader masks."
            )

    n_contours = contour_present.sum(dim=1)
    if torch.any(n_contours < 1):
        raise ValueError("Every case must have at least one non-empty reader mask.")

    # All four positions represent completed review sessions. An all-zero mask
    # casts a meaningful zero vote and is intentionally retained.
    vote_count = binary.sum(dim=1)
    levels = torch.arange(
        1,
        NUMBER_OF_HEADS + 1,
        device=reader_masks.device,
        dtype=vote_count.dtype,
    )
    targets = vote_count[:, None] >= levels[None, :, None, None, None]
    head_valid = torch.ones(
        (batch_size, NUMBER_OF_HEADS),
        device=reader_masks.device,
        dtype=torch.bool,
    )
    targets = targets & head_valid[:, :, None, None, None] & valid_voxels
    output_dtype = reader_masks.dtype if torch.is_floating_point(reader_masks) else torch.float32
    vote_fraction = vote_count.to(dtype=output_dtype) / float(NUMBER_OF_HEADS)
    vote_fraction = vote_fraction * valid_depth[:, :, None, None].to(vote_fraction.dtype)

    targets_out = targets.to(dtype=output_dtype)
    vote_count_out = vote_count
    vote_fraction_out = vote_fraction
    depth_valid_out = valid_depth
    if slice_batch:
        targets_out = targets_out[:, :, 0]
        vote_count_out = vote_count_out[:, 0]
        vote_fraction_out = vote_fraction_out[:, 0].unsqueeze(1)
        depth_valid_out = depth_valid_out[:, 0]

    return {
        "targets": targets_out,
        "valid_heads": head_valid,
        "vote_count": vote_count_out,
        "vote_fraction": vote_fraction_out,
        "n_contours": n_contours,
        "reader_contour_present": contour_present,
        "reader_session_count": torch.full_like(n_contours, NUMBER_OF_HEADS),
        "depth_valid": depth_valid_out,
    }


def _weighted_head_mean(
    values: torch.Tensor,
    included: torch.Tensor,
    head_weights: torch.Tensor,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average ``[B, K]`` values within each head, then across present heads."""

    counts = included.sum(dim=0)
    per_head = torch.where(
        counts > 0,
        (values * included.to(values.dtype)).sum(dim=0) / counts.clamp_min(1).to(values.dtype),
        torch.zeros_like(counts, dtype=values.dtype),
    )
    active_weights = head_weights * (counts > 0).to(head_weights.dtype)
    if bool((active_weights.sum() <= 0).item()):
        return zero, per_head
    return (per_head * active_weights).sum() / active_weights.sum(), per_head


def nested_agreement_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_heads: torch.Tensor,
    vote_fraction: torch.Tensor,
    depth_valid: torch.Tensor | None = None,
    *,
    head_weights: Sequence[float] = (0.4, 0.3, 0.2, 0.1),
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
    focal_weight: float = 0.45,
    dice_weight: float = 0.45,
    brier_weight: float = 0.10,
    epsilon: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Compute the fixed four-session nested-agreement objective.

    Parameters
    ----------
    logits, targets:
        Tensors ``[B, 4, D, H, W]`` ordered T1 through T4.  Slice tensors
        ``[B, 4, H, W]`` are accepted and promoted to depth-one volumes.
    valid_heads:
        Boolean tensor ``[B, 4]`` retained for loader/API consistency. Under
        the v3.1 four-session provenance, every entry must be true. Partial
        validity is rejected to prevent annotation-semantic drift.
    vote_fraction:
        Empirical reader support, shape ``[B, D, H, W]`` or
        ``[B, 1, D, H, W]``, scaled to ``[0, 1]``.
    depth_valid:
        Optional boolean ``[B, D]`` mask for padded slices.

    The focal term uses every voxel from all four heads, including legitimate
    empty higher-support targets. Soft Dice is computed over an entire case
    volume and included only when that head's target is non-empty. The Brier
    term compares empirical support with ``qhat = sum_k(p_k) / 4`` and
    balances foreground and background within each case.

    Returns
    -------
    dict
        Scalar ``loss``, ``focal``, ``dice``, and ``brier`` tensors plus
        per-head diagnostic vectors and valid-case counts.
    """

    slice_batch = logits.ndim == 4
    if slice_batch:
        if depth_valid is not None:
            raise ValueError("depth_valid is only accepted with volume logits.")
        logits = logits.unsqueeze(2)
        targets = targets.unsqueeze(2)
    if logits.ndim != 5 or logits.shape[1] != NUMBER_OF_HEADS:
        raise ValueError("logits must have shape [B, 4, H, W] or [B, 4, D, H, W].")
    if targets.shape != logits.shape:
        raise ValueError(
            f"targets must match logits shape {list(logits.shape)}, got {list(targets.shape)}"
        )
    if not torch.is_floating_point(logits):
        raise TypeError("logits must be floating point.")
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError("logits and targets must be finite.")

    batch_size, heads, depth, height, width = logits.shape
    if valid_heads.shape != (batch_size, heads):
        raise ValueError(
            f"valid_heads must have shape [{batch_size}, {heads}], got {list(valid_heads.shape)}"
        )
    valid_heads = valid_heads.to(device=logits.device, dtype=torch.bool)
    if not torch.all(valid_heads):
        raise ValueError(
            "All T1--T4 heads must be valid for the four completed LIDC review "
            "sessions; no-contour masks are meaningful zero votes."
        )

    valid_depth = _validate_depth_valid(
        depth_valid,
        batch_size=batch_size,
        depth=depth,
        device=logits.device,
    )
    target = targets.to(device=logits.device, dtype=logits.dtype)
    if torch.any((target < 0) | (target > 1)):
        raise ValueError("targets must be in [0, 1].")
    q = vote_fraction.to(device=logits.device, dtype=logits.dtype)
    if slice_batch and q.ndim == 4 and q.shape[1] == 1:
        q = q[:, 0].unsqueeze(1)
    elif slice_batch and q.ndim == 3:
        q = q.unsqueeze(1)
    if q.ndim == 5 and q.shape[1] == 1:
        q = q[:, 0]
    if q.shape != (batch_size, depth, height, width):
        raise ValueError(
            "vote_fraction must have shape "
            f"[{batch_size}, {depth}, {height}, {width}], got {list(q.shape)}"
        )
    if not torch.isfinite(q).all() or torch.any((q < 0) | (q > 1)):
        raise ValueError("vote_fraction must be finite and in [0, 1].")

    weights = torch.as_tensor(head_weights, device=logits.device, dtype=logits.dtype)
    if weights.shape != (heads,) or torch.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("head_weights must contain four non-negative values with positive sum.")
    if not 0.0 <= focal_alpha <= 1.0 or focal_gamma < 0:
        raise ValueError("focal_alpha must be in [0, 1] and focal_gamma must be non-negative.")
    component_weights = (focal_weight, dice_weight, brier_weight)
    if any(weight < 0 for weight in component_weights) or sum(component_weights) <= 0:
        raise ValueError("Loss component weights must be non-negative with positive sum.")

    zero = logits.sum() * 0.0
    probabilities = torch.sigmoid(logits)
    depth_mask = valid_depth[:, None, :, None, None].to(logits.dtype)

    # Focal BCE: first average voxels within each case/head, then cases within
    # each head so long volumes and common T1 labels do not dominate.
    cross_entropy = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_target = probabilities * target + (1.0 - probabilities) * (1.0 - target)
    alpha_target = focal_alpha * target + (1.0 - focal_alpha) * (1.0 - target)
    focal_voxel = alpha_target * (1.0 - p_target).pow(focal_gamma) * cross_entropy
    voxel_denominator = valid_depth.sum(dim=1).to(logits.dtype) * float(height * width)
    focal_per_case_head = (focal_voxel * depth_mask).sum(dim=(2, 3, 4)) / voxel_denominator[:, None]
    focal, per_head_focal = _weighted_head_mean(
        focal_per_case_head,
        valid_heads,
        weights,
        zero,
    )

    # Volume-level soft Dice, excluding valid-but-empty targets by definition.
    masked_probability = probabilities * depth_mask
    masked_target = target * depth_mask
    intersection = (masked_probability * masked_target).sum(dim=(2, 3, 4))
    prediction_mass = masked_probability.sum(dim=(2, 3, 4))
    target_mass = masked_target.sum(dim=(2, 3, 4))
    dice_per_case_head = 1.0 - (2.0 * intersection + epsilon) / (
        prediction_mass + target_mass + epsilon
    )
    dice_included = valid_heads & (target_mass > 0)
    dice, per_head_dice = _weighted_head_mean(
        dice_per_case_head,
        dice_included,
        weights,
        zero,
    )

    # E[K] = sum_k P(K >= k). All cases have four completed review sessions,
    # so division by four yields the predicted reader-support fraction.
    qhat = probabilities.sum(dim=1) / float(NUMBER_OF_HEADS)
    squared_error = (qhat - q).pow(2)
    spatial_valid = valid_depth[:, :, None, None]
    foreground = (q > 0) & spatial_valid
    background = (q <= 0) & spatial_valid

    def region_mean(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = mask.sum(dim=(1, 2, 3))
        mean = (squared_error * mask.to(squared_error.dtype)).sum(dim=(1, 2, 3)) / count.clamp_min(1).to(
            squared_error.dtype
        )
        return mean, count > 0

    foreground_brier, has_foreground = region_mean(foreground)
    background_brier, has_background = region_mean(background)
    region_count = has_foreground.to(logits.dtype) + has_background.to(logits.dtype)
    brier_per_case = (
        foreground_brier * has_foreground.to(logits.dtype)
        + background_brier * has_background.to(logits.dtype)
    ) / region_count.clamp_min(1.0)
    brier = brier_per_case.mean()

    normaliser = float(sum(component_weights))
    total = (
        focal_weight * focal + dice_weight * dice + brier_weight * brier
    ) / normaliser
    return {
        "loss": total,
        "focal": focal,
        "dice": dice,
        "brier": brier,
        "per_head_focal": per_head_focal,
        "per_head_dice": per_head_dice,
        "valid_cases_per_head": valid_heads.sum(dim=0),
        "dice_cases_per_head": dice_included.sum(dim=0),
        "reader_support_mean": qhat.mean(),
    }


__all__ = [
    "NUMBER_OF_HEADS",
    "build_nested_agreement_targets",
    "nested_agreement_loss",
]
