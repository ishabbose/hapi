"""Leakage-safe slice dataset for the multi-reader LIDC-IDRI v3.1 cohort.

The dataset exposes an axial 2.5D image stack together with the nested reader
vote targets ``T_k = 1[vote_count >= k]``.  All four heads are valid: the four
source mask slots represent the four LIDC-IDRI reader review sessions, and an
empty contour slot is meaningful zero support rather than a missing reader.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


HEAD_NAMES = ("T1", "T2", "T3", "T4")
_REQUIRED_COLUMNS = {
    "subset_nodule_id",
    "patient_id",
    "split",
    "depth",
    "height",
    "width",
    "n_clustered_contours",
    "image_volume_uint8_path",
    "reader_contour_present_path",
    "reader_vote_count_path",
    "reader_vote_fraction_path",
    "mask_majority_path",
}


def _resolve_path(dataset_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else dataset_root / path


def _spatial_transform(
    array: np.ndarray,
    *,
    rotation_k: int,
    flip_vertical: bool,
    flip_horizontal: bool,
) -> np.ndarray:
    """Apply one transform to the final two dimensions of any array."""

    transformed = np.rot90(array, k=rotation_k, axes=(-2, -1))
    if flip_vertical:
        transformed = np.flip(transformed, axis=-2)
    if flip_horizontal:
        transformed = np.flip(transformed, axis=-1)
    # A sequence such as rot180 + two flips can cancel geometrically and
    # expose the original read-only memory map again.  Always materialise an
    # owned writable array before handing data to ``torch.from_numpy``.
    return np.array(transformed, copy=True, order="C")


class AgreementSliceDataset(Dataset):
    """Axial 2.5D samples from ``final_325_manifest.csv``.

    Parameters
    ----------
    manifest:
        Path to the v3 manifest or an already loaded DataFrame.
    dataset_root:
        Root against which manifest array paths are resolved.  It defaults to
        the manifest's parent directory.  It is required for a DataFrame.
    split:
        A split name or sequence of names to retain.  ``None`` retains all.
    context_slices:
        Odd number of adjacent axial slices returned as image channels.  Edge
        indices are replicated, so every sample has the same channel count.
    intensity_low, intensity_high:
        Fold-training intensity bounds used to clip and scale the stored raw
        uint8 images to ``[0,1]``.  These are required so an outer held-out
        fold never influences preprocessing.  Pass ``0`` and ``255``
        explicitly to request identity uint8 scaling.
    augment:
        Enable deterministic flips and 90-degree rotations for every row in
        this dataset instance.  Construct training and evaluation instances
        from their respective DataFrames/splits and enable this flag only for
        the training instance.  This supports outer cross-validation folds
        whose training role is not represented by the legacy ``split`` field.
    augmentation_seed:
        Base seed.  Call :meth:`set_epoch` to obtain a new deterministic
        transform schedule for each epoch.
    cache_size:
        Number of cases whose memory-mapped arrays are retained per worker.
    validate:
        Validate paths, shapes, vote ranges, nesting, and majority semantics
        when a case is first loaded.
    """

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        *,
        dataset_root: str | Path | None = None,
        split: str | Sequence[str] | None = None,
        context_slices: int = 3,
        intensity_low: float | None = None,
        intensity_high: float | None = None,
        augment: bool = False,
        augmentation_seed: int = 42,
        flip_probability: float = 0.5,
        rotate: bool = True,
        cache_size: int = 4,
        validate: bool = True,
    ) -> None:
        if context_slices < 1 or context_slices % 2 == 0:
            raise ValueError("context_slices must be a positive odd integer")
        if not 0.0 <= flip_probability <= 1.0:
            raise ValueError("flip_probability must be in [0, 1]")
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        if intensity_low is None or intensity_high is None:
            raise ValueError(
                "intensity_low and intensity_high are required; derive them only "
                "from the current training fold (or explicitly pass 0 and 255)"
            )
        intensity_low = float(intensity_low)
        intensity_high = float(intensity_high)
        if (
            not np.isfinite(intensity_low)
            or not np.isfinite(intensity_high)
            or not 0.0 <= intensity_low < intensity_high <= 255.0
        ):
            raise ValueError(
                "intensity_low/high must be finite with "
                "0 <= intensity_low < intensity_high <= 255"
            )

        if isinstance(manifest, pd.DataFrame):
            if dataset_root is None:
                raise ValueError("dataset_root is required when manifest is a DataFrame")
            frame = manifest.copy()
            root = Path(dataset_root)
            self.manifest_path: Path | None = None
        else:
            manifest_path = Path(manifest)
            frame = pd.read_csv(manifest_path)
            root = Path(dataset_root) if dataset_root is not None else manifest_path.parent
            self.manifest_path = manifest_path.resolve()

        # Temporary read aliases ease migration from v3.0 manifests.  The
        # arrays are still validated under v3.1 four-session semantics; this
        # cannot make a legacy variable-denominator vote fraction acceptable.
        if "n_clustered_contours" not in frame and "n_available_readers" in frame:
            frame["n_clustered_contours"] = frame["n_available_readers"]
        if (
            "reader_contour_present_path" not in frame
            and "reader_available_path" in frame
        ):
            frame["reader_contour_present_path"] = frame["reader_available_path"]

        missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")

        if split is not None:
            requested = {split} if isinstance(split, str) else set(split)
            frame = frame[frame["split"].astype(str).isin(requested)].copy()
        if frame.empty:
            raise ValueError("No manifest rows remain after split filtering")

        frame["subset_nodule_id"] = frame["subset_nodule_id"].astype(str)
        frame["patient_id"] = frame["patient_id"].astype(str)
        frame["split"] = frame["split"].astype(str)
        if frame["subset_nodule_id"].duplicated().any():
            duplicates = frame.loc[
                frame["subset_nodule_id"].duplicated(False), "subset_nodule_id"
            ].tolist()
            raise ValueError(f"Duplicate subset_nodule_id values: {duplicates[:10]}")

        for column in ("depth", "height", "width", "n_clustered_contours"):
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
        if (frame[["depth", "height", "width"]] <= 0).any().any():
            raise ValueError("All manifest dimensions must be positive")
        if not frame["n_clustered_contours"].between(1, 4).all():
            raise ValueError("n_clustered_contours must be between 1 and 4")

        self.df = frame.reset_index(drop=True)
        self.dataset_root = root.resolve()
        self.context_slices = int(context_slices)
        self.intensity_low = intensity_low
        self.intensity_high = intensity_high
        self.augment = bool(augment)
        self.augmentation_seed = int(augmentation_seed)
        self.flip_probability = float(flip_probability)
        self.rotate = bool(rotate)
        self.cache_size = int(cache_size)
        self.validate = bool(validate)
        # Shared storage lets ``set_epoch`` reach persistent DataLoader worker
        # copies instead of silently repeating the epoch-zero transforms.
        self._epoch_state = torch.zeros((), dtype=torch.int64).share_memory_()
        self._cache: OrderedDict[int, tuple[np.ndarray, ...]] = OrderedDict()
        self._validated_rows: set[int] = set()

        self.index: list[tuple[int, int]] = []
        self.volume_to_indices: dict[str, list[int]] = defaultdict(list)
        for row_index, row in self.df.iterrows():
            volume_id = str(row["subset_nodule_id"])
            for z in range(int(row["depth"])):
                sample_index = len(self.index)
                self.index.append((int(row_index), int(z)))
                self.volume_to_indices[volume_id].append(sample_index)
        self.volume_to_indices = dict(self.volume_to_indices)

    def __len__(self) -> int:
        return len(self.index)

    @property
    def volume_ids(self) -> tuple[str, ...]:
        return tuple(self.volume_to_indices)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used by deterministic sample-level augmentation."""

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch_state.fill_(int(epoch))

    def clear_cache(self) -> None:
        """Release cached memory maps in the current process/worker."""

        self._cache.clear()

    def _load_case(self, row_index: int) -> tuple[np.ndarray, ...]:
        if row_index in self._cache:
            arrays = self._cache.pop(row_index)
            self._cache[row_index] = arrays
            return arrays

        row = self.df.iloc[row_index]
        paths = [
            _resolve_path(self.dataset_root, row["image_volume_uint8_path"]),
            _resolve_path(self.dataset_root, row["reader_contour_present_path"]),
            _resolve_path(self.dataset_root, row["reader_vote_count_path"]),
            _resolve_path(self.dataset_root, row["reader_vote_fraction_path"]),
            _resolve_path(self.dataset_root, row["mask_majority_path"]),
        ]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing v3 case arrays: {missing}")

        image_uint8 = np.load(paths[0], mmap_mode="r")
        contour_present = np.load(paths[1], mmap_mode="r")
        vote_count = np.load(paths[2], mmap_mode="r")
        vote_fraction = np.load(paths[3], mmap_mode="r")
        majority = np.load(paths[4], mmap_mode="r")
        arrays = (image_uint8, contour_present, vote_count, vote_fraction, majority)

        if self.validate and row_index not in self._validated_rows:
            expected_shape = (
                int(row["depth"]),
                int(row["height"]),
                int(row["width"]),
            )
            if image_uint8.shape != expected_shape:
                raise ValueError(
                    f"{row['subset_nodule_id']}: image shape "
                    f"{image_uint8.shape} != {expected_shape}"
                )
            for name, array in (
                ("vote_count", vote_count),
                ("vote_fraction", vote_fraction),
                ("majority", majority),
            ):
                if array.shape != expected_shape:
                    raise ValueError(
                        f"{row['subset_nodule_id']}: {name} shape {array.shape} != {expected_shape}"
                    )
            if image_uint8.dtype != np.uint8:
                raise ValueError(
                    f"{row['subset_nodule_id']}: raw image dtype must be uint8, "
                    f"got {image_uint8.dtype}"
                )
            if contour_present.shape != (4,):
                raise ValueError(
                    f"{row['subset_nodule_id']}: contour_present shape must be (4,)"
                )
            if not np.isin(np.asarray(contour_present), (0, 1)).all():
                raise ValueError(
                    f"{row['subset_nodule_id']}: contour_present must be binary"
                )
            n_contours = int(np.asarray(contour_present).astype(bool).sum())
            if n_contours != int(row["n_clustered_contours"]):
                raise ValueError(
                    f"{row['subset_nodule_id']}: clustered-contour count {n_contours} "
                    f"!= manifest {row['n_clustered_contours']}"
                )
            if not np.issubdtype(vote_count.dtype, np.integer):
                raise ValueError(
                    f"{row['subset_nodule_id']}: vote_count must have integer dtype"
                )
            if vote_count.min() < 0 or vote_count.max() > n_contours:
                raise ValueError(f"{row['subset_nodule_id']}: invalid reader vote count")
            expected_fraction = np.asarray(vote_count, dtype=np.float32) / 4.0
            if not np.allclose(vote_fraction, expected_fraction, atol=1e-6):
                raise ValueError(
                    f"{row['subset_nodule_id']}: vote fraction must use denominator four"
                )
            expected_majority = np.asarray(vote_count) >= 2
            if not np.isin(np.asarray(majority), (0, 1)).all():
                raise ValueError(f"{row['subset_nodule_id']}: majority must be binary")
            if not np.array_equal(np.asarray(majority).astype(bool), expected_majority):
                raise ValueError(f"{row['subset_nodule_id']}: majority mask is inconsistent")
            self._validated_rows.add(row_index)

        if self.cache_size:
            self._cache[row_index] = arrays
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return arrays

    def _augmentation_parameters(self, sample_index: int) -> tuple[int, bool, bool]:
        seed_sequence = np.random.SeedSequence(
            [self.augmentation_seed, int(self._epoch_state.item()), int(sample_index)]
        )
        rng = np.random.default_rng(seed_sequence)
        rotation_k = int(rng.integers(0, 4)) if self.rotate else 0
        flip_vertical = bool(rng.random() < self.flip_probability)
        flip_horizontal = bool(rng.random() < self.flip_probability)
        return rotation_k, flip_vertical, flip_horizontal

    def __getitem__(self, sample_index: int) -> dict[str, Any]:
        row_index, z = self.index[sample_index]
        row = self.df.iloc[row_index]
        image_uint8, contour_present, vote_count, vote_fraction, majority = (
            self._load_case(row_index)
        )

        radius = self.context_slices // 2
        z_indices = np.clip(
            np.arange(z - radius, z + radius + 1),
            0,
            image_uint8.shape[0] - 1,
        )
        image = np.asarray(image_uint8[z_indices], dtype=np.float32)
        image = np.clip(
            (image - self.intensity_low) / (self.intensity_high - self.intensity_low),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
        count = np.asarray(vote_count[z], dtype=np.uint8)
        targets = count[None, ...] >= np.arange(1, 5, dtype=np.uint8)[:, None, None]
        q = np.asarray(vote_fraction[z], dtype=np.float32)
        majority_target = np.asarray(majority[z], dtype=np.float32)

        n_contours = int(row["n_clustered_contours"])
        valid_heads = np.ones(4, dtype=bool)
        if self.validate:
            if np.any(targets[1:] & ~targets[:-1]):
                raise ValueError(f"{row['subset_nodule_id']} z={z}: non-nested targets")
            if not np.array_equal(targets[1], majority_target.astype(bool)):
                raise ValueError(
                    f"{row['subset_nodule_id']} z={z}: majority target mismatch"
                )

        if self.augment:
            rotation_k, flip_vertical, flip_horizontal = self._augmentation_parameters(
                sample_index
            )
            transform = {
                "rotation_k": rotation_k,
                "flip_vertical": flip_vertical,
                "flip_horizontal": flip_horizontal,
            }
            image = _spatial_transform(image, **transform)
            targets = _spatial_transform(targets, **transform)
            q = _spatial_transform(q, **transform)
            majority_target = _spatial_transform(majority_target, **transform)
        else:
            # ``np.load(..., mmap_mode="r")`` produces read-only views.  Give
            # PyTorch owned, writable buffers rather than relying on
            # ``ascontiguousarray``, which may return the original memmap.
            image = np.array(image, copy=True, order="C")
            targets = np.array(targets, copy=True, order="C")
            q = np.array(q, copy=True, order="C")
            majority_target = np.array(majority_target, copy=True, order="C")

        target_tensor = torch.from_numpy(targets.astype(np.float32, copy=False))
        nested_targets = {
            name: target_tensor[index : index + 1]
            for index, name in enumerate(HEAD_NAMES)
        }
        q_tensor = torch.from_numpy(q).unsqueeze(0)
        majority_tensor = torch.from_numpy(majority_target).unsqueeze(0)

        return {
            "volume_id": str(row["subset_nodule_id"]),
            "patient_id": str(row["patient_id"]),
            "split": str(row["split"]),
            "z": int(z),
            "depth": int(row["depth"]),
            "image": torch.from_numpy(image),
            "targets": target_tensor,
            "nested_targets": nested_targets,
            "valid_heads": torch.from_numpy(valid_heads),
            "vote_fraction": q_tensor,
            "q": q_tensor,
            "majority": majority_tensor,
            "contour_present": torch.from_numpy(
                np.asarray(contour_present).astype(bool, copy=True)
            ),
            "n_clustered_contours": n_contours,
            "intensity_low": self.intensity_low,
            "intensity_high": self.intensity_high,
        }


class AgreementVolumeDataset(AgreementSliceDataset):
    """Full variable-depth 2.5D volumes for transformer/volume training.

    Each item contains ``image`` with shape ``[D, C, H, W]``, agreement
    ``targets`` with shape ``[4, D, H, W]``, and ``vote_fraction``/``q`` with
    shape ``[D, H, W]``.  Adjacent-slice image channels use edge replication.
    When ``augment=True``, one deterministic spatial transform is shared by
    every slice, context channel, and target plane in the volume.  As with the
    slice dataset, pass only training rows to an augmented instance.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.index = list(range(len(self.df)))
        self.volume_to_indices = {
            str(row["subset_nodule_id"]): [int(row_index)]
            for row_index, row in self.df.iterrows()
        }

    def __getitem__(self, sample_index: int) -> dict[str, Any]:
        row_index = int(self.index[sample_index])
        row = self.df.iloc[row_index]
        image_uint8, contour_present, vote_count, vote_fraction, majority = (
            self._load_case(row_index)
        )

        depth = int(image_uint8.shape[0])
        radius = self.context_slices // 2
        centers = np.arange(depth)[:, None]
        offsets = np.arange(-radius, radius + 1)[None, :]
        context_indices = np.clip(centers + offsets, 0, depth - 1)
        image = np.asarray(image_uint8[context_indices], dtype=np.float32)
        image = np.clip(
            (image - self.intensity_low) / (self.intensity_high - self.intensity_low),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
        count = np.asarray(vote_count, dtype=np.uint8)
        targets = count[None, ...] >= np.arange(1, 5, dtype=np.uint8)[:, None, None, None]
        q = np.asarray(vote_fraction, dtype=np.float32)
        majority_target = np.asarray(majority, dtype=np.float32)

        n_contours = int(row["n_clustered_contours"])
        valid_heads = np.ones(4, dtype=bool)
        if self.validate:
            if np.any(targets[1:] & ~targets[:-1]):
                raise ValueError(f"{row['subset_nodule_id']}: non-nested volume targets")
            if not np.array_equal(targets[1], majority_target.astype(bool)):
                raise ValueError(f"{row['subset_nodule_id']}: majority volume target mismatch")

        if self.augment:
            rotation_k, flip_vertical, flip_horizontal = self._augmentation_parameters(
                sample_index
            )
            transform = {
                "rotation_k": rotation_k,
                "flip_vertical": flip_vertical,
                "flip_horizontal": flip_horizontal,
            }
            image = _spatial_transform(image, **transform)
            targets = _spatial_transform(targets, **transform)
            q = _spatial_transform(q, **transform)
            majority_target = _spatial_transform(majority_target, **transform)
        else:
            image = np.array(image, copy=True, order="C")
            targets = np.array(targets, copy=True, order="C")
            q = np.array(q, copy=True, order="C")
            majority_target = np.array(majority_target, copy=True, order="C")

        target_tensor = torch.from_numpy(targets.astype(np.float32, copy=False))
        q_tensor = torch.from_numpy(q)
        return {
            "volume_id": str(row["subset_nodule_id"]),
            "patient_id": str(row["patient_id"]),
            "split": str(row["split"]),
            "depth": depth,
            "image": torch.from_numpy(image),
            "targets": target_tensor,
            "nested_targets": {
                name: target_tensor[index : index + 1]
                for index, name in enumerate(HEAD_NAMES)
            },
            "valid_heads": torch.from_numpy(valid_heads),
            "vote_fraction": q_tensor,
            "q": q_tensor,
            "majority": torch.from_numpy(majority_target),
            "contour_present": torch.from_numpy(
                np.asarray(contour_present).astype(bool, copy=True)
            ),
            "n_clustered_contours": n_contours,
            "intensity_low": self.intensity_low,
            "intensity_high": self.intensity_high,
        }


def pad_agreement_volumes(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-depth volume items for a true-volume mini-batch.

    Returns images ``[B,D,C,H,W]``, targets ``[B,4,D,H,W]``, vote fractions
    ``[B,D,H,W]``, and ``depth_valid`` ``[B,D]``.  Padded slices are zero and
    must be excluded from losses/attention with ``depth_valid``.
    """

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(batch)
    depths = [int(item["image"].shape[0]) for item in batch]
    max_depth = max(depths)
    channels, height, width = batch[0]["image"].shape[1:]

    image = torch.zeros(
        (batch_size, max_depth, channels, height, width), dtype=torch.float32
    )
    targets = torch.zeros(
        (batch_size, 4, max_depth, height, width), dtype=torch.float32
    )
    q = torch.zeros((batch_size, max_depth, height, width), dtype=torch.float32)
    majority = torch.zeros_like(q)
    depth_valid = torch.zeros((batch_size, max_depth), dtype=torch.bool)

    for batch_index, (item, depth) in enumerate(zip(batch, depths)):
        if tuple(item["image"].shape[1:]) != (channels, height, width):
            raise ValueError("All volume images must share C, H, and W")
        if tuple(item["targets"].shape) != (4, depth, height, width):
            raise ValueError("Volume targets must have shape [4,D,H,W]")
        if tuple(item["vote_fraction"].shape) != (depth, height, width):
            raise ValueError("Volume vote_fraction must have shape [D,H,W]")
        if tuple(item["majority"].shape) != (depth, height, width):
            raise ValueError("Volume majority must have shape [D,H,W]")
        image[batch_index, :depth] = item["image"].to(dtype=torch.float32)
        targets[batch_index, :, :depth] = item["targets"].to(dtype=torch.float32)
        q[batch_index, :depth] = item["vote_fraction"].to(dtype=torch.float32)
        majority[batch_index, :depth] = item["majority"].to(dtype=torch.float32)
        depth_valid[batch_index, :depth] = True

    intensity_low = float(batch[0]["intensity_low"])
    intensity_high = float(batch[0]["intensity_high"])
    if any(
        float(item["intensity_low"]) != intensity_low
        or float(item["intensity_high"]) != intensity_high
        for item in batch[1:]
    ):
        raise ValueError("All volume items in a batch must use one intensity window")

    nested_targets = {
        name: targets[:, index : index + 1]
        for index, name in enumerate(HEAD_NAMES)
    }
    return {
        "volume_id": [str(item["volume_id"]) for item in batch],
        "patient_id": [str(item["patient_id"]) for item in batch],
        "split": [str(item["split"]) for item in batch],
        "depth": torch.as_tensor(depths, dtype=torch.long),
        "image": image,
        "targets": targets,
        "nested_targets": nested_targets,
        "valid_heads": torch.stack(
            [item["valid_heads"].to(dtype=torch.bool) for item in batch]
        ),
        "vote_fraction": q,
        "q": q,
        "majority": majority,
        "contour_present": torch.stack(
            [item["contour_present"].to(dtype=torch.bool) for item in batch]
        ),
        "n_clustered_contours": torch.as_tensor(
            [int(item["n_clustered_contours"]) for item in batch], dtype=torch.long
        ),
        "intensity_low": intensity_low,
        "intensity_high": intensity_high,
        "depth_valid": depth_valid,
    }


class VolumeAwareSampler(Sampler[int]):
    """Random sampler giving each volume equal expected contribution.

    Every slice receives weight ``1 / depth(volume)``.  With the default
    replacement sampling and ``num_samples=len(dataset)``, volumes are sampled
    uniformly in expectation without discarding the short volumes.  Call
    :meth:`set_epoch` alongside the dataset's method for deterministic epochs.
    """

    def __init__(
        self,
        dataset: AgreementSliceDataset,
        *,
        num_samples: int | None = None,
        replacement: bool = True,
        seed: int = 42,
    ) -> None:
        if not isinstance(dataset, AgreementSliceDataset):
            raise TypeError("dataset must be an AgreementSliceDataset")
        self.dataset = dataset
        self.num_samples = len(dataset) if num_samples is None else int(num_samples)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if not self.replacement and self.num_samples > len(dataset):
            raise ValueError("num_samples cannot exceed dataset length without replacement")

        weights = np.empty(len(dataset), dtype=np.float64)
        for indices in dataset.volume_to_indices.values():
            weight = 1.0 / float(len(indices))
            weights[np.asarray(indices, dtype=int)] = weight
        self.weights = torch.as_tensor(weights, dtype=torch.double)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        sampled = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=self.replacement,
            generator=generator,
        )
        return iter(sampled.tolist())


def make_volume_aware_sampler(
    dataset: AgreementSliceDataset,
    *,
    num_samples: int | None = None,
    seed: int = 42,
) -> VolumeAwareSampler:
    """Convenience constructor for the recommended training sampler."""

    return VolumeAwareSampler(
        dataset,
        num_samples=num_samples,
        replacement=True,
        seed=seed,
    )


__all__ = [
    "AgreementSliceDataset",
    "AgreementVolumeDataset",
    "HEAD_NAMES",
    "VolumeAwareSampler",
    "make_volume_aware_sampler",
    "pad_agreement_volumes",
]
