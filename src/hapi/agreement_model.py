"""Coverage-aware 2.5D model for multi-reader nodule segmentation.

The model consumes one adjacent-slice stack per target slice.  A complete
volume batch has shape ``[B, D, C, H, W]`` where ``C`` is normally three
(``previous, current, next``).  It returns four *ordered* logit volumes with
shape ``[B, 4, D, H, W]``.  The four outputs represent nested targets marked
by at least one, two, three, or four LIDC review sessions.

For compatibility with slice loaders, ``[B, C, H, W]`` input is also accepted
and returns ``[B, 4, H, W]``.  Volume batches should be preferred when the
loss is intended to compute true per-volume Dice.

The ordering is guaranteed by construction rather than encouraged with a
penalty: ``z1 = a`` and ``zk = z(k-1) - softplus(gk)``.  Consequently,
``sigmoid(z1) >= ... >= sigmoid(z4)`` at every voxel.

An optional lightweight slice-token transformer can condition bottleneck
features on the full valid depth of a case.  It is disabled by default; the
conservative model uses only the local 2.5D context supplied in the input.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, maximum: int = 8) -> int:
    """Return the largest useful GroupNorm group count that divides channels."""

    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _normalise_depth_valid(
    depth_valid: torch.Tensor | None,
    *,
    batch_size: int,
    depth: int,
    device: torch.device,
) -> torch.Tensor:
    """Validate or construct a ``[B, D]`` boolean valid-slice mask."""

    if depth_valid is None:
        return torch.ones((batch_size, depth), dtype=torch.bool, device=device)
    if depth_valid.shape != (batch_size, depth):
        raise ValueError(
            "depth_valid must have shape "
            f"[{batch_size}, {depth}], got {list(depth_valid.shape)}"
        )
    depth_valid = depth_valid.to(device=device, dtype=torch.bool)
    if not torch.all(depth_valid.any(dim=1)):
        raise ValueError("Every case must contain at least one valid slice.")
    return depth_valid


def build_adjacent_slice_stacks(
    images: torch.Tensor,
    depth_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert grayscale volumes into previous/current/next 2.5D stacks.

    Parameters
    ----------
    images:
        Grayscale image tensor with shape ``[B, 1, D, H, W]``.
    depth_valid:
        Optional boolean tensor ``[B, D]``.  Valid slices must form a
        contiguous prefix in each padded volume.  At each real volume edge,
        the nearest valid slice is replicated.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``[B, D, 3, H, W]``.
    """

    if images.ndim != 5:
        raise ValueError(
            "images must have shape [B, 1, D, H, W], "
            f"got {list(images.shape)}"
        )
    batch_size, channels, depth, height, width = images.shape
    if channels != 1:
        raise ValueError(f"Expected one grayscale channel, got {channels}.")
    valid = _normalise_depth_valid(
        depth_valid,
        batch_size=batch_size,
        depth=depth,
        device=images.device,
    )

    lengths = valid.sum(dim=1)
    expected = torch.arange(depth, device=images.device).unsqueeze(0) < lengths.unsqueeze(1)
    if not torch.equal(valid, expected):
        raise ValueError("depth_valid must mark a contiguous valid prefix per case.")

    # Gather along D while clamping each case to its own final valid slice.
    positions = torch.arange(depth, device=images.device).unsqueeze(0).expand(batch_size, -1)
    last = (lengths - 1).unsqueeze(1)
    previous_index = torch.maximum(positions - 1, torch.zeros_like(positions))
    next_index = torch.minimum(positions + 1, last)
    current_index = torch.minimum(positions, last)

    source = images[:, 0]

    def gather(index: torch.Tensor) -> torch.Tensor:
        expanded = index[:, :, None, None].expand(-1, -1, height, width)
        return torch.gather(source, dim=1, index=expanded)

    stacks = torch.stack(
        [gather(previous_index), gather(current_index), gather(next_index)],
        dim=2,
    )
    return stacks * valid[:, :, None, None, None].to(dtype=stacks.dtype)


class ResidualBlock2D(nn.Module):
    """Two-convolution residual block using GroupNorm for small batches."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(_group_count(out_channels), out_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        features = F.silu(self.norm1(self.conv1(inputs)), inplace=True)
        features = self.dropout(features)
        features = self.norm2(self.conv2(features))
        return F.silu(features + residual, inplace=True)


class UpBlock2D(nn.Module):
    """Bilinear upsampling followed by skip fusion and a residual block."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.block = ResidualBlock2D(out_channels + skip_channels, out_channels, dropout)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        inputs = self.reduce(inputs)
        return self.block(torch.cat([inputs, skip], dim=1))


def _sinusoidal_depth_encoding(
    depth: int,
    channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a deterministic ``[1, D, C]`` depth-position encoding."""

    positions = torch.arange(depth, device=device, dtype=torch.float32).unsqueeze(1)
    even_channels = torch.arange(0, channels, 2, device=device, dtype=torch.float32)
    scales = torch.exp(-math.log(10000.0) * even_channels / max(channels, 1))
    angles = positions * scales.unsqueeze(0)
    encoding = torch.zeros((depth, channels), device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    if channels > 1:
        encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0).to(dtype=dtype)


class SliceTokenTransformer(nn.Module):
    """Lightweight full-depth context using one spatially pooled token per slice.

    Contextualised tokens generate FiLM scale and bias parameters for the full
    bottleneck maps.  The FiLM projection is zero-initialised, so enabling this
    module starts from the identity transformation.
    """

    def __init__(
        self,
        channels: int,
        *,
        layers: int = 2,
        heads: int = 4,
        feedforward_multiplier: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("transformer layers must be at least one.")
        if heads < 1 or channels % heads != 0:
            raise ValueError("bottleneck channels must be divisible by transformer heads.")
        if feedforward_multiplier < 1:
            raise ValueError("feedforward_multiplier must be at least one.")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=channels * feedforward_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.film = nn.Linear(channels, 2 * channels)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, features: torch.Tensor, depth_valid: torch.Tensor) -> torch.Tensor:
        if features.ndim != 5:
            raise ValueError("features must have shape [B, D, C, H, W].")
        batch_size, depth, channels, _, _ = features.shape
        if depth_valid.shape != (batch_size, depth):
            raise ValueError("depth_valid shape does not match bottleneck features.")
        tokens = features.mean(dim=(-1, -2))
        tokens = tokens + _sinusoidal_depth_encoding(
            depth,
            channels,
            device=features.device,
            dtype=features.dtype,
        )
        tokens = self.encoder(tokens, src_key_padding_mask=~depth_valid)
        gamma, beta = self.film(tokens).chunk(2, dim=-1)
        gamma = gamma[:, :, :, None, None]
        beta = beta[:, :, :, None, None]
        conditioned = features * (1.0 + gamma) + beta
        return conditioned * depth_valid[:, :, None, None, None].to(features.dtype)


class MonotonicAgreementHead(nn.Module):
    """Project decoder features to guaranteed nested T1--T4 logits."""

    number_of_heads = 4

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, self.number_of_heads, 1)
        nn.init.kaiming_normal_(self.projection.weight, nonlinearity="linear")
        with torch.no_grad():
            self.projection.bias[0] = -2.0
            self.projection.bias[1:] = -2.0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.projection(features)
        logits = [raw[:, 0:1]]
        for gap_index in range(1, self.number_of_heads):
            logits.append(logits[-1] - F.softplus(raw[:, gap_index : gap_index + 1]))
        return torch.cat(logits, dim=1)


class IndependentAgreementHead(nn.Module):
    """Project decoder features to one or more unconstrained logits."""

    def __init__(self, in_channels: int, number_of_heads: int = 4) -> None:
        super().__init__()
        if number_of_heads < 1:
            raise ValueError("number_of_heads must be positive.")
        self.number_of_heads = number_of_heads
        self.projection = nn.Conv2d(in_channels, number_of_heads, 1)
        nn.init.kaiming_normal_(self.projection.weight, nonlinearity="linear")
        nn.init.constant_(self.projection.bias, -2.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


class ExactCountHead(nn.Module):
    """Project decoder features to exact reader-count classes K=0, ..., 4."""

    number_of_classes = 5

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, self.number_of_classes, 1)
        nn.init.kaiming_normal_(self.projection.weight, nonlinearity="linear")
        nn.init.zeros_(self.projection.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


def count_logits_to_cumulative_probabilities(count_logits: torch.Tensor) -> torch.Tensor:
    """Convert exact-count logits for K=0..4 into nested P(K>=1)..P(K>=4).

    Both slice logits ``[B, 5, H, W]`` and volume logits
    ``[B, 5, D, H, W]`` are accepted.  The returned tensor has four channels
    and the same remaining dimensions.  Nesting follows directly from the
    cumulative sum of a valid categorical distribution.
    """

    if count_logits.ndim not in {4, 5} or count_logits.shape[1] != 5:
        raise ValueError("count_logits must have shape [B, 5, H, W] or [B, 5, D, H, W].")
    probabilities = torch.softmax(count_logits, dim=1)
    return torch.stack(
        [probabilities[:, threshold:].sum(dim=1) for threshold in range(1, 5)],
        dim=1,
    )


class NestedAgreementResUNet2p5D(nn.Module):
    """Residual 2.5D U-Net with ordered multi-reader agreement outputs.

    Parameters
    ----------
    in_channels:
        Channels per target slice.  Use three with
        :func:`build_adjacent_slice_stacks`.
    base_channels:
        Width of the first encoder stage.  ``24`` provides a conservative
        model for the 325-case cohort.
    dropout:
        Dropout probability used in the deeper residual blocks.
    use_transformer:
        Enable full-depth bottleneck conditioning.  Disabled by default.
    ordered_outputs:
        Guarantee ``T1 >= T2 >= T3 >= T4`` probabilities.  Keep enabled for
        the proposed model; disable only for the controlled output-head
        ablation.
    output_channels:
        Use four for agreement models or one for the hard-``T2`` residual
        baselines. Ordered outputs always require four channels.

    Forward API
    -----------
    ``forward(stacks, depth_valid=None) -> logits`` where a volume ``stacks``
    tensor is ``[B, D, C, H, W]``, ``depth_valid`` is ``[B, D]``, and output
    ``logits`` is ``[B, 4, D, H, W]`` ordered T1 through T4.  Slice input
    ``[B, C, H, W]`` is shape-preserving and returns ``[B, 4, H, W]``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 24,
        dropout: float = 0.10,
        *,
        use_transformer: bool = False,
        ordered_outputs: bool = True,
        output_channels: int = 4,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_feedforward_multiplier: int = 2,
        transformer_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive.")
        if base_channels < 4:
            raise ValueError("base_channels must be at least four.")
        if output_channels not in {1, 4}:
            raise ValueError("output_channels must be one or four.")
        if ordered_outputs and output_channels != 4:
            raise ValueError("ordered outputs require four cumulative channels.")
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.use_transformer = use_transformer
        self.ordered_outputs = ordered_outputs
        self.number_of_outputs = output_channels

        widths = [base_channels, 2 * base_channels, 4 * base_channels, 8 * base_channels]
        self.encoder1 = ResidualBlock2D(in_channels, widths[0], 0.0)
        self.encoder2 = ResidualBlock2D(widths[0], widths[1], dropout * 0.5)
        self.encoder3 = ResidualBlock2D(widths[1], widths[2], dropout)
        self.bottleneck = ResidualBlock2D(widths[2], widths[3], dropout)
        self.pool = nn.MaxPool2d(2)

        if use_transformer:
            self.depth_context: nn.Module | None = SliceTokenTransformer(
                widths[3],
                layers=transformer_layers,
                heads=transformer_heads,
                feedforward_multiplier=transformer_feedforward_multiplier,
                dropout=transformer_dropout,
            )
        else:
            self.depth_context = None

        self.decoder3 = UpBlock2D(widths[3], widths[2], widths[2], dropout)
        self.decoder2 = UpBlock2D(widths[2], widths[1], widths[1], dropout * 0.5)
        self.decoder1 = UpBlock2D(widths[1], widths[0], widths[0], 0.0)
        if ordered_outputs:
            self.agreement_head: nn.Module = MonotonicAgreementHead(widths[0])
        else:
            self.agreement_head = IndependentAgreementHead(widths[0], output_channels)

    def forward(
        self,
        stacks: torch.Tensor,
        depth_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        slice_batch = stacks.ndim == 4
        if slice_batch:
            if depth_valid is not None:
                raise ValueError("depth_valid is only accepted with [B, D, C, H, W] input.")
            stacks = stacks.unsqueeze(1)
        if stacks.ndim != 5:
            raise ValueError(
                "stacks must have shape [B, C, H, W] or [B, D, C, H, W], "
                f"got {list(stacks.shape)}"
            )
        batch_size, depth, channels, height, width = stacks.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {channels}.")
        if min(height, width) < 8:
            raise ValueError("Input height and width must both be at least eight pixels.")
        valid = _normalise_depth_valid(
            depth_valid,
            batch_size=batch_size,
            depth=depth,
            device=stacks.device,
        )

        flat = stacks.reshape(batch_size * depth, channels, height, width)
        encoder1 = self.encoder1(flat)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        bottleneck = self.bottleneck(self.pool(encoder3))

        if self.depth_context is not None:
            _, bottleneck_channels, bottleneck_h, bottleneck_w = bottleneck.shape
            volume_features = bottleneck.reshape(
                batch_size,
                depth,
                bottleneck_channels,
                bottleneck_h,
                bottleneck_w,
            )
            volume_features = self.depth_context(volume_features, valid)
            bottleneck = volume_features.reshape(
                batch_size * depth,
                bottleneck_channels,
                bottleneck_h,
                bottleneck_w,
            )

        decoder3 = self.decoder3(bottleneck, encoder3)
        decoder2 = self.decoder2(decoder3, encoder2)
        decoder1 = self.decoder1(decoder2, encoder1)
        flat_logits = self.agreement_head(decoder1)
        logits = flat_logits.reshape(
            batch_size,
            depth,
            self.number_of_outputs,
            height,
            width,
        )
        logits = logits.permute(0, 2, 1, 3, 4).contiguous()
        return logits[:, :, 0] if slice_batch else logits

    @torch.no_grad()
    def predict_probabilities(
        self,
        stacks: torch.Tensor,
        depth_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ordered T1--T4 probabilities, preserving slice/volume rank."""

        return torch.sigmoid(self(stacks, depth_valid))


class ExactCountResUNet2p5D(NestedAgreementResUNet2p5D):
    """Matched exact vote-count comparator with five categorical outputs.

    This class reuses the proposed model's complete residual encoder, decoder,
    2.5D input convention, and optional bottleneck context.  Only its output
    parameterisation changes: logits represent exactly K=0, 1, 2, 3, or 4
    reader contours at each voxel.

    ``forward`` preserves the parent shape convention and returns either
    ``[B, 5, H, W]`` or ``[B, 5, D, H, W]``.  Use
    :meth:`predict_cumulative_probabilities` (or the standalone helper) for
    directly comparable nested T1--T4 probabilities.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 24,
        dropout: float = 0.10,
        *,
        use_transformer: bool = False,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_feedforward_multiplier: int = 2,
        transformer_dropout: float = 0.10,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            dropout=dropout,
            use_transformer=use_transformer,
            ordered_outputs=False,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_feedforward_multiplier=transformer_feedforward_multiplier,
            transformer_dropout=transformer_dropout,
        )
        self.number_of_outputs = ExactCountHead.number_of_classes
        self.agreement_head = ExactCountHead(base_channels)
        self.output_parameterisation = "exact_count_softmax"

    @torch.no_grad()
    def predict_probabilities(
        self,
        stacks: torch.Tensor,
        depth_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return exact K=0..4 softmax probabilities, preserving input rank."""

        return torch.softmax(self(stacks, depth_valid), dim=1)

    @torch.no_grad()
    def predict_cumulative_probabilities(
        self,
        stacks: torch.Tensor,
        depth_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return nested P(K>=1)..P(K>=4), preserving slice/volume rank."""

        return count_logits_to_cumulative_probabilities(self(stacks, depth_valid))


__all__ = [
    "ExactCountHead",
    "ExactCountResUNet2p5D",
    "IndependentAgreementHead",
    "NestedAgreementResUNet2p5D",
    "MonotonicAgreementHead",
    "ResidualBlock2D",
    "SliceTokenTransformer",
    "build_adjacent_slice_stacks",
    "count_logits_to_cumulative_probabilities",
]
