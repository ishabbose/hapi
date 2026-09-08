"""Regression tests for the four-session agreement-model core."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hapi.agreement_losses import (  # noqa: E402
    build_nested_agreement_targets,
    nested_agreement_loss,
)
from hapi.agreement_metrics import AgreementMetricsAccumulator  # noqa: E402
from hapi.agreement_model import (  # noqa: E402
    MonotonicAgreementHead,
    NestedAgreementResUNet2p5D,
    count_logits_to_cumulative_probabilities,
)


def _load_evaluator_module():
    """Load the evaluator script without requiring ``scripts`` as a package."""

    path = REPO_ROOT / "scripts" / "22_evaluate_agreement_model.py"
    specification = importlib.util.spec_from_file_location(
        "hapi_evaluate_agreement_model",
        path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load evaluator module from {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _load_trainer_module():
    path = REPO_ROOT / "scripts" / "21_train_agreement_model.py"
    specification = importlib.util.spec_from_file_location(
        "hapi_train_agreement_model",
        path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load trainer module from {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class AgreementModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_monotonic_head_orders_probabilities(self) -> None:
        head = MonotonicAgreementHead(in_channels=3)
        features = torch.randn(2, 3, 5, 7)
        probabilities = torch.sigmoid(head(features))

        self.assertEqual(tuple(probabilities.shape), (2, 4, 5, 7))
        self.assertTrue(
            bool(torch.all(probabilities[:, :-1] >= probabilities[:, 1:]))
        )

    def test_exact_count_probabilities_convert_to_cumulative_support(self) -> None:
        exact_probabilities = torch.tensor(
            [0.10, 0.20, 0.30, 0.15, 0.25],
            dtype=torch.float64,
        )
        logits = exact_probabilities.log().reshape(1, 5, 1, 1)
        cumulative = count_logits_to_cumulative_probabilities(logits)
        expected = torch.tensor(
            [0.90, 0.70, 0.40, 0.25],
            dtype=torch.float64,
        ).reshape(1, 4, 1, 1)

        torch.testing.assert_close(cumulative, expected)
        self.assertTrue(bool(torch.all(cumulative[:, :-1] >= cumulative[:, 1:])))

        expected_support = (
            exact_probabilities * torch.arange(5, dtype=torch.float64)
        ).sum() / 4.0
        torch.testing.assert_close(cumulative.sum(dim=1).squeeze() / 4.0, expected_support)

    def test_hard_t2_residual_baseline_has_one_output(self) -> None:
        model = NestedAgreementResUNet2p5D(
            in_channels=3,
            base_channels=4,
            ordered_outputs=False,
            output_channels=1,
        )
        logits = model(torch.randn(1, 2, 3, 16, 16), torch.ones(1, 2, dtype=torch.bool))
        self.assertEqual(tuple(logits.shape), (1, 1, 2, 16, 16))


class AgreementTargetAndLossTests(unittest.TestCase):
    def test_ranked_probability_score_matches_published_five_class_scale(self) -> None:
        ordinal_rps_loss = _load_trainer_module().ordinal_rps_loss
        logits = torch.full((1, 5, 1, 1, 1), -100.0)
        logits[:, 0] = 100.0
        batch = {
            "targets": torch.ones((1, 4, 1, 1, 1), dtype=torch.float32),
            "depth_valid": torch.ones((1, 1), dtype=torch.bool),
        }
        parts = ordinal_rps_loss(logits, batch)
        self.assertAlmostEqual(float(parts["dice"]), 0.8, places=6)

    def test_ordinal_loss_has_finite_autocast_backward_gradient(self) -> None:
        ordinal_rps_loss = _load_trainer_module().ordinal_rps_loss
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logits = torch.randn((2, 5, 2, 8, 8), device=device, requires_grad=True)
        targets = torch.zeros((2, 4, 2, 8, 8), device=device)
        targets[0, :2, :, 2:6, 2:6] = 1
        targets[1, :3, 0, 1:5, 1:5] = 1
        batch = {
            "targets": targets,
            "depth_valid": torch.tensor(
                [[True, True], [True, False]],
                device=device,
            ),
        }
        autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=True):
            parts = ordinal_rps_loss(logits, batch)
        parts["loss"].backward()
        for name in ("loss", "focal", "dice"):
            self.assertTrue(bool(torch.isfinite(parts[name])), name)
        self.assertIsNotNone(logits.grad)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_nested_targets_use_four_session_denominator(self) -> None:
        reader_masks = torch.zeros((1, 4, 2, 3), dtype=torch.float32)
        reader_masks[0, 0, 0, :] = 1
        reader_masks[0, 1, 0, 1:] = 1
        reader_masks[0, 2, 0, 2] = 1
        contour_present = torch.tensor([[True, True, True, False]])

        result = build_nested_agreement_targets(
            reader_masks,
            reader_contour_present=contour_present,
        )

        self.assertTrue(bool(result["valid_heads"].all()))
        self.assertEqual(result["n_contours"].tolist(), [3])
        self.assertEqual(result["vote_count"][0, 0].tolist(), [1, 2, 3])
        torch.testing.assert_close(
            result["vote_fraction"][0, 0, 0],
            torch.tensor([0.25, 0.50, 0.75]),
        )
        self.assertEqual(
            result["targets"][0, :, 0].to(torch.int64).tolist(),
            [
                [1, 1, 1],
                [0, 1, 1],
                [0, 0, 1],
                [0, 0, 0],
            ],
        )

    def test_nested_loss_has_finite_backward_gradient(self) -> None:
        reader_masks = torch.zeros((2, 4, 2, 8, 8), dtype=torch.float32)
        reader_masks[0, 0, :, 1:6, 1:6] = 1
        reader_masks[0, 1, 0, 2:6, 2:6] = 1
        reader_masks[1, 0, 0, 3:7, 3:7] = 1
        depth_valid = torch.tensor([[True, True], [True, False]])
        target_data = build_nested_agreement_targets(
            reader_masks,
            depth_valid=depth_valid,
        )
        logits = torch.randn((2, 4, 2, 8, 8), requires_grad=True)

        parts = nested_agreement_loss(
            logits,
            target_data["targets"],
            target_data["valid_heads"],
            target_data["vote_fraction"],
            depth_valid,
        )
        parts["loss"].backward()

        for name in ("loss", "focal", "dice", "brier"):
            self.assertTrue(bool(torch.isfinite(parts[name])), name)
        self.assertIsNotNone(logits.grad)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)


class CalibrationAndMetricTests(unittest.TestCase):
    def test_calibration_pooling_is_unique_and_voxel_weighted(self) -> None:
        aggregate = _load_evaluator_module().aggregate_calibration_rows
        common = {
            "model": "ordered",
            "seed": 11,
            "outer_fold": 2,
            "scope": "full_roi",
            "bin_lower": 0.2,
            "bin_upper": 0.4,
        }
        rows = [
            {**common, "mean_predicted": 0.2, "mean_observed": 0.1, "n_voxels": 2},
            {**common, "mean_predicted": 0.8, "mean_observed": 0.5, "n_voxels": 6},
            {
                **common,
                "scope": "t1_union",
                "mean_predicted": 0.3,
                "mean_observed": 0.2,
                "n_voxels": 5,
            },
        ]

        pooled = aggregate(rows)
        keys = [
            "model",
            "seed",
            "outer_fold",
            "scope",
            "bin_lower",
            "bin_upper",
        ]
        self.assertFalse(bool(pooled.duplicated(keys).any()))
        self.assertEqual(len(pooled), 2)
        full_roi = pooled[pooled["scope"] == "full_roi"].iloc[0]
        self.assertEqual(int(full_roi["n_voxels"]), 8)
        self.assertAlmostEqual(float(full_roi["mean_predicted"]), 0.65)
        self.assertAlmostEqual(float(full_roi["mean_observed"]), 0.40)

    def test_overlap_and_presence_metrics(self) -> None:
        batch_size, depth, height, width = 4, 1, 2, 2
        vote_count = np.zeros(
            (batch_size, depth, height, width),
            dtype=np.uint8,
        )
        vote_count[0, 0, 0, 0] = 2
        vote_count[1, 0, 0, 0] = 2
        vote_count[2, 0, 0, 0] = 1
        vote_count[3, 0, 0, 0] = 1
        targets = vote_count[:, None] >= np.arange(1, 5)[None, :, None, None, None]
        vote_fraction = vote_count.astype(np.float32) / 4.0
        majority = vote_count >= 2

        predictions = targets.astype(np.float32)
        predictions[1, 1] = 0
        predictions[2, 0, 0, 0, 1] = 1
        predictions[2, 1, 0, 0, 1] = 1

        accumulator = AgreementMetricsAccumulator(expected_split="test")
        accumulator.update_volume_batch(
            volume_id=[f"v{index}" for index in range(batch_size)],
            patient_id=[f"p{index}" for index in range(batch_size)],
            split=["test"] * batch_size,
            head_predictions=predictions,
            targets=targets,
            valid_heads=np.ones((batch_size, 4), dtype=bool),
            vote_fraction=vote_fraction,
            majority=majority,
            n_clustered_contours=[2, 2, 1, 1],
            depth=[1] * batch_size,
            from_logits=False,
        )
        report = accumulator.compute(n_bootstrap=8)
        t2 = report.target_summary.set_index("target").loc["T2"]

        for name in ("presence_tp", "presence_tn", "presence_fp", "presence_fn"):
            self.assertEqual(int(t2[name]), 1, name)
        for name in (
            "presence_sensitivity",
            "presence_specificity",
            "presence_precision",
            "presence_f1",
            "case_macro_precision",
            "case_macro_recall",
        ):
            self.assertAlmostEqual(float(t2[name]), 0.5, msg=name)
        self.assertTrue(
            {"precision", "recall", "sensitivity"}.issubset(
                report.case_metrics.columns
            )
        )
        self.assertTrue(
            {"precision", "recall"}.issubset(set(report.bootstrap_cis["metric"]))
        )


if __name__ == "__main__":
    unittest.main()
