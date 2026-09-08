#!/usr/bin/env python3
"""Plan or execute the locked HAPI agreement-model experiment suites.

Dry-run is the default.  Pass ``--execute`` only after reviewing the printed
matrix.  Completed evaluations are skipped; a trained checkpoint without an
evaluation is evaluated; an incomplete training directory fails closed unless
``--force-incomplete`` is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_VARIANTS = (
    "A0_unet2d_t2",
    "A1_resunet2d_t2",
    "A2_resunet2p5d_t2",
    "A3_resunet2d_nested",
    "A4_proposed",
    "A5_independent_heads",
    "A6_transformer",
    "A7_ordinal_rps",
    "A8_no_brier",
)
TRAINING_CODE_FILES = (
    REPO_ROOT / "scripts/21_train_agreement_model.py",
    REPO_ROOT / "src/hapi/agreement_dataset.py",
    REPO_ROOT / "src/hapi/agreement_losses.py",
    REPO_ROOT / "src/hapi/agreement_model.py",
    REPO_ROOT / "src/hapi/models.py",
)
EVALUATION_CODE_FILES = (
    REPO_ROOT / "scripts/22_evaluate_agreement_model.py",
    REPO_ROOT / "src/hapi/agreement_dataset.py",
    REPO_ROOT / "src/hapi/agreement_metrics.py",
    REPO_ROOT / "src/hapi/agreement_model.py",
    REPO_ROOT / "src/hapi/models.py",
)
SUITES: dict[str, dict[str, Any]] = {
    "smoke": {
        "variants": ("A4_proposed", "A7_ordinal_rps"),
        "seeds": (42,),
        "folds": (0,),
        "epochs": 1,
        "smoke": True,
    },
    "screen": {
        "variants": ALL_VARIANTS,
        "seeds": (42,),
        "folds": (0, 1, 2, 3, 4),
        "epochs": 80,
        "smoke": False,
    },
    "confirm": {
        "variants": ("A0_unet2d_t2", "A4_proposed", "A7_ordinal_rps"),
        "seeds": (41, 42, 43),
        "folds": (0, 1, 2, 3, 4),
        "epochs": 80,
        "smoke": False,
    },
    "full": {
        "variants": ALL_VARIANTS,
        "seeds": (41, 42, 43),
        "folds": (0, 1, 2, 3, 4),
        "epochs": 80,
        "smoke": False,
    },
}


def csv_strings(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("duplicate values are not allowed")
    return result


def csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in csv_strings(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Dry-run or execute leakage-safe five-fold ablation suites. "
            "Use screen for all A0-A8 variants at seed 42, confirm for A0/A4/A7 "
            "at three seeds, or full for all variants at three seeds."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    result.add_argument("--suite", choices=sorted(SUITES), default="smoke")
    result.add_argument("--execute", action="store_true", help="run commands; otherwise print plan")
    result.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/final_325_nodules_v3")
    result.add_argument("--results-root", type=Path, default=REPO_ROOT / "results/final_agreement")
    result.add_argument("--variants", type=csv_strings, default=None)
    result.add_argument("--seeds", type=csv_ints, default=None)
    result.add_argument("--folds", type=csv_ints, default=None)
    result.add_argument("--epochs", type=int, default=None)
    result.add_argument("--patience", type=int, default=15)
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--accumulation-steps", type=int, default=4)
    result.add_argument("--base-channels", type=int, default=24)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--device", default="auto")
    result.add_argument("--no-amp", action="store_true")
    result.add_argument("--evaluation-bootstrap", type=int, default=2000)
    result.add_argument("--aggregate-bootstrap", type=int, default=10000)
    result.add_argument(
        "--force-incomplete",
        action="store_true",
        help="restart only an existing run directory that lacks best.pt",
    )
    result.add_argument("--skip-aggregate", action="store_true")
    result.add_argument("--skip-figures", action="store_true")
    return result


def command_text(command: Sequence[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in command)


def run(command: Sequence[str]) -> None:
    print(f"\n$ {command_text(command)}", flush=True)
    subprocess.run(list(command), cwd=REPO_ROOT, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_fingerprint(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = str(path.resolve().relative_to(REPO_ROOT))
        value = sha256_file(path)
        digest.update(relative.encode("utf-8") + b"\0" + value.encode("ascii") + b"\n")
    return digest.hexdigest()


def valid_evaluation(
    run_dir: Path,
    *,
    smoke: bool,
    dataset_content_sha256: str,
    manifest_sha256: str,
    expected_config: dict[str, Any],
    evaluation_code_sha256: str,
    expected_bootstrap: int,
) -> bool:
    path = run_dir / "evaluation_summary.json"
    variant = str(expected_config["variant"])
    required = [
        "config.json",
        "run_summary.json",
        "best.pt",
        "history.csv",
        "case_metrics.csv",
        "target_summary.csv",
        "bootstrap_cis.csv",
        "evaluation_summary.json",
    ]
    if variant not in {"A0_unet2d_t2", "A1_resunet2d_t2", "A2_resunet2p5d_t2"}:
        required.extend(["soft_case_metrics.csv", "calibration.csv"])
    if not all((run_dir / name).is_file() for name in required):
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        run_summary = json.loads(
            (run_dir / "run_summary.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    config_matches = all(config.get(key) == value for key, value in expected_config.items())
    try:
        threshold = float(payload.get("threshold_selected_on_validation"))
        checkpoint_sha256 = sha256_file(run_dir / "best.pt")
    except (OSError, TypeError, ValueError):
        return False
    return payload.get("status") == "EVALUATION_COMPLETE" and (
        payload.get("evaluation_code_sha256") == evaluation_code_sha256
    ) and (
        payload.get("checkpoint_sha256") == checkpoint_sha256
        and run_summary.get("checkpoint_sha256") == checkpoint_sha256
        and 0.0 <= threshold <= 1.0
        and payload.get("surface_metrics_available") is True
    ) and (
        payload.get("bootstrap_replicates") == expected_bootstrap
        and payload.get("confidence") == 0.95
    ) and config_matches and (
        smoke or not config.get("smoke", False)
    ) and (
        config.get("dataset_content_sha256") == dataset_content_sha256
    ) and (
        config.get("manifest_sha256") == manifest_sha256
    )


def valid_training(
    run_dir: Path,
    *,
    smoke: bool,
    dataset_content_sha256: str,
    manifest_sha256: str,
    expected_config: dict[str, Any],
) -> bool:
    if not all(
        (run_dir / name).is_file()
        for name in ("best.pt", "config.json", "run_summary.json")
    ):
        return False
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    try:
        threshold = float(summary.get("selected_threshold"))
        checkpoint_sha256 = sha256_file(run_dir / "best.pt")
    except (OSError, TypeError, ValueError):
        return False
    return (
        summary.get("status") == ("SMOKE_PASS" if smoke else "TRAINING_COMPLETE")
        and summary.get("checkpoint_sha256") == checkpoint_sha256
        and 0.0 <= threshold <= 1.0
        and all(config.get(key) == value for key, value in expected_config.items())
        and bool(config.get("smoke", False)) == smoke
        and config.get("dataset_content_sha256") == dataset_content_sha256
        and config.get("manifest_sha256") == manifest_sha256
    )


def main() -> int:
    args = parser().parse_args()
    suite = SUITES[args.suite]
    variants = tuple(args.variants or suite["variants"])
    seeds = tuple(args.seeds or suite["seeds"])
    folds = tuple(args.folds or suite["folds"])
    epochs = int(suite["epochs"] if args.epochs is None else args.epochs)
    smoke = bool(suite["smoke"])

    if smoke and args.epochs not in (None, 1):
        raise SystemExit("The smoke suite is fixed at one epoch")

    unknown = sorted(set(variants).difference(ALL_VARIANTS))
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}")
    if any(fold not in range(5) for fold in folds):
        raise SystemExit("Every fold must be in 0..4")
    positive = (
        epochs,
        args.patience,
        args.batch_size,
        args.accumulation_steps,
        args.base_channels,
        args.evaluation_bootstrap,
        args.aggregate_bootstrap,
    )
    if any(value < 1 for value in positive):
        raise SystemExit("Epochs, batch settings, channels, and bootstrap counts must be positive")
    if args.base_channels < 4:
        raise SystemExit("--base-channels must be at least 4")
    if args.workers < 0:
        raise SystemExit("--workers must be non-negative")
    aggregation_enabled = not smoke and not args.skip_aggregate
    if aggregation_enabled:
        if tuple(sorted(folds)) != (0, 1, 2, 3, 4):
            raise SystemExit("Final aggregation requires folds 0,1,2,3,4")
        if 42 not in seeds:
            raise SystemExit("Final aggregation requires seed 42 for the qualitative OOF set")
        if "A0_unet2d_t2" not in variants or "A4_proposed" not in variants:
            raise SystemExit(
                "Final aggregation requires A0_unet2d_t2 and A4_proposed; "
                "include them or pass --skip-aggregate"
            )
        if args.evaluation_bootstrap < 1000 or args.aggregate_bootstrap < 1000:
            raise SystemExit("Final aggregation requires both bootstrap counts to be at least 1000")

    data_root = args.data_root.resolve()
    results_root = args.results_root.resolve()
    validation_path = data_root / "validation_report.json"
    if not validation_path.is_file():
        raise SystemExit(f"Missing {validation_path}; build and validate v3.1 first")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise SystemExit("Dataset validation does not report PASS")
    if args.execute:
        run(
            [
                sys.executable,
                "scripts/20_validate_final_v3.py",
                "--data-root",
                str(data_root),
            ]
        )
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation.get("status") != "PASS":
            raise SystemExit("Fresh dataset validation did not report PASS")
    dataset_content_sha256 = str(validation["dataset_content_sha256"])
    manifest_sha256 = sha256_file(data_root / "final_325_manifest.csv")
    if args.device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = str(torch.device(args.device))
    resolved_amp = resolved_device.startswith("cuda") and not args.no_amp
    training_code_sha256 = code_fingerprint(TRAINING_CODE_FILES)
    evaluation_code_sha256 = code_fingerprint(EVALUATION_CODE_FILES)

    tasks: list[dict[str, Any]] = []
    for variant in variants:
        for seed in seeds:
            for fold in folds:
                root = results_root / ("_smoke" if smoke else "")
                run_dir = root / variant / f"seed_{seed}" / f"fold_{fold}"
                expected_config = {
                    "variant": variant,
                    "outer_fold": fold,
                    "seed": seed,
                    "epochs_requested": epochs,
                    "patience": 1 if smoke else args.patience,
                    "batch_size": args.batch_size,
                    "accumulation_steps": 1 if smoke else args.accumulation_steps,
                    "base_channels": 4 if smoke else args.base_channels,
                    "learning_rate": 3e-4,
                    "weight_decay": 1e-4,
                    "dropout": 0.10,
                    "device": resolved_device,
                    "amp": resolved_amp,
                    "training_code_sha256": training_code_sha256,
                }
                complete = valid_evaluation(
                    run_dir,
                    smoke=smoke,
                    dataset_content_sha256=dataset_content_sha256,
                    manifest_sha256=manifest_sha256,
                    expected_config=expected_config,
                    evaluation_code_sha256=evaluation_code_sha256,
                    expected_bootstrap=args.evaluation_bootstrap,
                )
                trained = valid_training(
                    run_dir,
                    smoke=smoke,
                    dataset_content_sha256=dataset_content_sha256,
                    manifest_sha256=manifest_sha256,
                    expected_config=expected_config,
                )
                tasks.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "outer_fold": fold,
                        "run_dir": str(run_dir),
                        "status": "complete" if complete else "trained" if trained else "pending",
                    }
                )

    plan = {
        "suite": args.suite,
        "smoke": smoke,
        "execute": args.execute,
        "variants": list(variants),
        "seeds": list(seeds),
        "folds": list(folds),
        "epochs": epochs,
        "dataset_content_sha256": validation.get("dataset_content_sha256"),
        "n_runs": len(tasks),
        "n_complete": sum(task["status"] == "complete" for task in tasks),
        "n_trained_not_evaluated": sum(task["status"] == "trained" for task in tasks),
        "n_pending": sum(task["status"] == "pending" for task in tasks),
        "tasks": tasks,
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return 0

    for task in tasks:
        run_dir = Path(task["run_dir"])
        if task["status"] == "complete":
            print(f"SKIP complete: {run_dir}")
            continue
        if task["status"] == "pending":
            if run_dir.exists() and any(run_dir.iterdir()) and not args.force_incomplete:
                raise SystemExit(
                    f"Incomplete run exists at {run_dir}; inspect it, then pass "
                    "--force-incomplete to restart this run"
                )
            train = [
                sys.executable,
                "scripts/21_train_agreement_model.py",
                "--variant",
                task["variant"],
                "--data-root",
                str(data_root),
                "--output-root",
                str(results_root),
                "--outer-fold",
                str(task["outer_fold"]),
                "--seed",
                str(task["seed"]),
                "--epochs",
                str(epochs),
                "--patience",
                str(args.patience),
                "--batch-size",
                str(args.batch_size),
                "--accumulation-steps",
                str(args.accumulation_steps),
                "--base-channels",
                str(args.base_channels),
                "--workers",
                str(args.workers),
                "--device",
                args.device,
            ]
            if smoke:
                train.append("--smoke")
            if args.no_amp:
                train.append("--no-amp")
            if args.force_incomplete and run_dir.exists():
                train.append("--force")
            run(train)

        evaluate = [
            sys.executable,
            "scripts/22_evaluate_agreement_model.py",
            "--run-dir",
            str(run_dir),
            "--data-root",
            str(data_root),
            "--device",
            args.device,
            "--workers",
            str(args.workers),
            "--bootstrap",
            str(args.evaluation_bootstrap),
        ]
        if smoke:
            evaluate.append("--allow-smoke")
        if args.no_amp:
            evaluate.append("--no-amp")
        if (run_dir / "case_metrics.csv").exists():
            evaluate.append("--force")
        run(evaluate)

    if smoke or args.skip_aggregate:
        return 0
    aggregate_root = results_root / "aggregates" / args.suite
    aggregate = [
        sys.executable,
        "scripts/24_aggregate_final_results.py",
        "--results-root",
        str(results_root),
        "--output-root",
        str(aggregate_root),
        "--variants",
        ",".join(variants),
        "--seeds",
        ",".join(map(str, seeds)),
        "--folds",
        ",".join(map(str, folds)),
        "--bootstrap-replicates",
        str(args.aggregate_bootstrap),
        "--prediction-variant",
        "A4_proposed",
        "--prediction-seed",
        "42",
    ]
    run(aggregate)

    if not args.skip_figures:
        figures = [
            sys.executable,
            "scripts/25_export_paper_figures.py",
            "--results-root",
            str(aggregate_root),
            "--data-root",
            str(data_root),
            "--qualitative-pool",
            "all",
            "--dpi",
            "600",
        ]
        run(figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
