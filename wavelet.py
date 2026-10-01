import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT3d(nn.Module):
    """
    Single-level 3D Haar Discrete Wavelet Transform.

    Decomposes a volume into 8 frequency sub-bands using fixed orthonormal
    Haar filters. Registered as buffers — move to GPU without contributing
    to the parameter count.

    Sub-band order: LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH
      LLL = coarse approximation (smooth structure)
      *H* = high-pass axis → encodes edges / detail in that dimension

    Input:  [B, 1, D, H, W]
    Output: [B, 8, D/2, H/2, W/2]
    
    """

    def __init__(self):
        super().__init__()
        L = torch.tensor([1.0, 1.0]) / (2 ** 0.5)   # low-pass
        H = torch.tensor([1.0, -1.0]) / (2 ** 0.5)  # high-pass

        filters = []
        for fd in (L, H):
            for fh in (L, H):
                for fw in (L, H):
                    f3d = fd[:, None, None] * fh[None, :, None] * fw[None, None, :]
                    filters.append(f3d)

        weight = torch.stack(filters, dim=0).unsqueeze(1)  # [8, 1, 2, 2, 2]
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(x, self.weight, stride=2, padding=0)


def _max_dwt_levels(dim: int) -> int:
    """
    Maximum number of DWT levels applicable to a spatial dimension.
    DWT requires an even input at each level, so we count how many times
    dim is divisible by 2.

    Examples: 96 → 5,  48 → 4,  24 → 3,  12 → 2,  6 → 1,  3 → 0
    """
    levels = 0
    while dim % 2 == 0:
        dim //= 2
        levels += 1
    return levels


# ---------------------------------------------------------------------------
# Wavelet filter bank registry
# ---------------------------------------------------------------------------
# Analysis (decomposition) 1-D filter pairs  (lo, hi)  for separable 3-D DWT.
# Coefficients sourced from PyWavelets (github.com/PyWavelets/pywt).
#
# Padding rule — ensures output spatial size is exactly D/2 for any even D:
#   padding = (filter_len - 2) // 2
#
# Proof:  out = floor((D + 2·p - k) / 2) + 1
#             = floor((D + (k-2) - k) / 2) + 1
#             = D / 2    (for even D)
#
# Haar   : k=2, p=0  →  96 → 48  ✓
# db2    : k=4, p=1  →  96 → 48  ✓
# sym4   : k=8, p=3  →  96 → 48  ✓
# ---------------------------------------------------------------------------
_WAVELET_FILTERS: dict = {
    # Haar — piecewise-constant, maximum time-frequency localisation
    "haar": (
        [0.7071067811865476,  0.7071067811865476],
        [0.7071067811865476, -0.7071067811865476],
    ),
    # Daubechies-2 (db2) — 4-tap, 2 vanishing moments, minimum-phase
    "db2": (
        [-0.12940952255126034,  0.22414386804201339,
          0.83651630373780772,  0.48296291314453410],
        [-0.48296291314453410,  0.83651630373780772,
         -0.22414386804201339, -0.12940952255126034],
    ),
    # Symlet-4 (sym4) — 8-tap, 4 vanishing moments, near-symmetric phase
    "sym4": (
        [-0.07576571478927333, -0.02963552764599851,
          0.49761866763201545,  0.80373875180591614,
          0.29785779560527736, -0.09921954357684722,
         -0.01260396726203783,  0.03222310060404270],
        [-0.03222310060404270, -0.01260396726203783,
          0.09921954357684722,  0.29785779560527736,
         -0.80373875180591614,  0.49761866763201545,
          0.02963552764599851, -0.07576571478927333],
    ),
}


class DWT3d(nn.Module):
    """
    Generalised 3-D single-level Discrete Wavelet Transform.

    Builds a separable 3-D analysis filter bank from the chosen 1-D wavelet.
    All three spatial axes share the same lo/hi pair, yielding 8 sub-bands in
    the standard order: LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH.

    Filters are registered as non-trainable buffers (identical to HaarDWT3d).
    Padding is computed automatically so that the output size is exactly D/2
    for any even spatial dimension D (see _WAVELET_FILTERS comment above).

    Parameters
    ----------
    wavelet : str
        One of 'haar', 'db2', 'sym4'.

    Input  shape: [B, 1, D, H, W]
    Output shape: [B, 8, D/2, H/2, W/2]
    """

    def __init__(self, wavelet: str = "haar"):
        super().__init__()
        if wavelet not in _WAVELET_FILTERS:
            raise ValueError(
                f"Unknown wavelet '{wavelet}'. "
                f"Available: {list(_WAVELET_FILTERS)}"
            )
        lo_1d, hi_1d = _WAVELET_FILTERS[wavelet]
        L = torch.tensor(lo_1d, dtype=torch.float32)
        H = torch.tensor(hi_1d, dtype=torch.float32)

        filters = []
        for fd in (L, H):
            for fh in (L, H):
                for fw in (L, H):
                    f3d = (fd[:, None, None]
                           * fh[None, :, None]
                           * fw[None, None, :])
                    filters.append(f3d)

        k = len(lo_1d)
        weight = torch.stack(filters, dim=0).unsqueeze(1)  # [8, 1, k, k, k]
        self.register_buffer("weight", weight)
        self.padding: int = (k - 2) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(x, self.weight, stride=2, padding=self.padding)


class SubBandSE(nn.Module):
    """
    Squeeze-and-Excitation over the 8 DWT sub-band channels.

    Learns a per-sub-band importance weight via global average pooling and
    a two-layer bottleneck MLP (8 → 4 → 8) with sigmoid gating.

    Input/Output shape: [B, 8, D, H, W]  (unchanged)
    Parameters: 2 × (8 × 4) = 64  (no bias)
    """

    def __init__(self, channels: int = 8, reduction: int = 2):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        scale = self.squeeze(x).view(b, c)
        scale = self.excite(scale).view(b, c, 1, 1, 1)
        return x * scale


class WaveletPatchEmbed(nn.Module):
    """
    Variant A — single-level wavelet patch embedding.

    Replaces SwinUNETR's standard patch embedding with a single-level 3D
    Haar DWT followed by a learned 1×1×1 projection.

    Pipeline:
        [B, 1, D, H, W]
        → HaarDWT3d  → [B, 8, D/2, H/2, W/2]   (8 frequency sub-bands)
        → Conv3d(8→embed_dim, k=1)  → [B, embed_dim, D/2, H/2, W/2]

    Parameters: 8×embed_dim + embed_dim = 432 for embed_dim=48
    (identical to the baseline Conv3d(1→48, k=2, s=2) = 432 params)
    """

    def __init__(self, in_chans: int = 1, embed_dim: int = 48):
        super().__init__()
        self.dwt  = HaarDWT3d()
        self.proj = nn.Conv3d(8 * in_chans, embed_dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply DWT independently per input channel, then concatenate.
        bands = torch.cat([self.dwt(x[:, c:c+1]) for c in range(x.shape[1])], dim=1)
        return self.proj(bands)


class HaarIDWT3d(nn.Module):
    """
    Single-level 3D Haar Inverse Discrete Wavelet Transform.

    Exact inverse of HaarDWT3d — Haar filters are orthonormal (W^T W = I),
    so IDWT = DWT^T = ConvTranspose3d with the same filter weights.

    Input:  [B, 8, D/2, H/2, W/2]
    Output: [B, 1, D,   H,   W  ]
    """

    def __init__(self):
        super().__init__()
        L = torch.tensor([1.0, 1.0]) / (2 ** 0.5)
        H = torch.tensor([1.0, -1.0]) / (2 ** 0.5)

        filters = []
        for fd in (L, H):
            for fh in (L, H):
                for fw in (L, H):
                    f3d = fd[:, None, None] * fh[None, :, None] * fw[None, None, :]
                    filters.append(f3d)

        weight = torch.stack(filters, dim=0).unsqueeze(1)  # [8, 1, 2, 2, 2]
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose3d(x, self.weight, stride=2)


class WaveletSkipRefinement(nn.Module):
    """
    Variant B — frequency-aware refinement of a decoder skip connection.

    Each feature channel is decomposed independently via the 3D Haar DWT
    into 8 frequency sub-bands. A single SubBandSE block recalibrates
    sub-band importance (e.g. suppressing HHH noise, amplifying edge bands).
    The signal is exactly reconstructed via the IDWT and added back as a
    residual.

    Pipeline per channel:
        [B, C, D, H, W]
        → reshape [B*C, 1, D, H, W]
        → HaarDWT3d   → [B*C, 8, D/2, H/2, W/2]
        → SubBandSE   → frequency recalibration
        → HaarIDWT3d  → [B*C, 1, D, H, W]
        → reshape [B, C, D, H, W]
        → + skip  (residual)

    Parameters: 64 (SE only — DWT/IDWT are fixed buffers)
    """

    def __init__(self):
        super().__init__()
        self.dwt = HaarDWT3d()
        self.se = SubBandSE(channels=8, reduction=2)
        self.idwt = HaarIDWT3d()

    def forward(self, skip: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = skip.shape
        x = skip.reshape(B * C, 1, D, H, W)
        x = self.dwt(x)
        x = self.se(x)
        x = self.idwt(x)
        x = x.reshape(B, C, D, H, W)
        return x + skip


class WaveletSkipDecoder(nn.Module):
    """
    Wraps a SwinUNETR UnetrUpBlock to apply WaveletSkipRefinement on the
    skip connection before it is concatenated with the upsampled features.

    Forward signature mirrors UnetrUpBlock: forward(inp, skip)
    """

    def __init__(self, decoder_block: nn.Module):
        super().__init__()
        self.decoder = decoder_block
        self.refine = WaveletSkipRefinement()

    def forward(self, inp: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.decoder(inp, self.refine(skip))



class WaveletPatchEmbedML(nn.Module):
    """
    Variant A-HL — multi-level wavelet patch embedding with full LLL decomposition.

    Applies the 3D Haar DWT recursively to the LLL (coarse approximation)
    sub-band for `levels` iterations, fully decomposing the volume down to
    its base coefficient. For 96×96×96 patches, levels=5 reduces LLL to 3³.

    Sub-bands from every level are upsampled back to the level-1 spatial
    resolution (D/2 × H/2 × W/2) and recalibrated by a dedicated per-level
    SubBandSE block. All recalibrated bands are concatenated and projected
    to embed_dim, giving the Swin Transformer a multi-resolution frequency
    view at the tokenisation stage:

        level 1 — fine detail bands at D/2    (original token resolution)
        level 2 — mid-scale bands at D/4  → upsampled to D/2
        level 3 — coarse bands at D/8     → upsampled to D/2
        ...
        level L — base approximation at D/2ᴸ  → upsampled to D/2

    Pipeline:
        [B, 1, D, H, W]
        → DWT₁ → [B, 8, D/2, ...]   SE₁ recalibrates
        → DWT₂ → [B, 8, D/4, ...]   SE₂ recalibrates → upsample to D/2
        ...
        → DWT_L → [B, 8, D/2ᴸ, ...] SE_L recalibrates → upsample to D/2
        → concat → [B, 8L, D/2, H/2, W/2]
        → Conv3d(8L→embed_dim, k=1) → [B, embed_dim, D/2, H/2, W/2]

    Parameters (levels=5, embed_dim=48):
        SE:   5 × 64  = 320
        proj: 40×48+48 = 1968
        total: 2288
    """

    def __init__(self, in_chans: int = 1, embed_dim: int = 48, levels: int = 5,
                 wavelet: str = "haar", coeff_aug: bool = False,
                 noise_std: float = 0.05, scale_delta: float = 0.1,
                 legacy_se_recursion: bool = False):
        """
        legacy_se_recursion
            False (default): each level decomposes the raw LLL of the previous
            level; SE only reweights the bands that are stored — a true
            multi-level DWT.
            True: RP2 behaviour — the next level decomposes the SE-rescaled
            LLL. Needed to evaluate RP2 checkpoints (levels >= 2) faithfully.
        """
        super().__init__()
        self.levels = levels
        self.legacy_se_recursion = legacy_se_recursion
        self.coeff_aug = coeff_aug
        self.noise_std = noise_std
        self.scale_delta = scale_delta
        self.dwt = DWT3d(wavelet)
        self.se_blocks = nn.ModuleList(
            [SubBandSE(channels=8, reduction=2) for _ in range(levels)]
        )
        # Each input channel contributes 8*levels sub-bands.
        self.proj = nn.Conv3d(in_chans * 8 * levels, embed_dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_chans = x.shape[1]
        D, H, W  = x.shape[2], x.shape[3], x.shape[4]
        actual_levels = min(
            self.levels,
            _max_dwt_levels(D),
            _max_dwt_levels(H),
            _max_dwt_levels(W),
        )

        target_size = None   # set from first (level=1, channel=0) sub-band
        all_bands   = []

        for c in range(in_chans):
            approx = x[:, c:c+1]   # [B, 1, D, H, W]
            for i in range(actual_levels):
                sub = self.dwt(approx)        # [B, 8, D/2^(i+1), ...]

                if self.training and self.coeff_aug:
                    B_s, device = sub.shape[0], sub.device
                    approx_part = sub[:, :1]   # LLL — low-frequency approximation
                    detail_part = sub[:, 1:]   # LLH…HHH — detail / edge bands
                    # Approx scaling: simulates MRI bias field (low-frequency global shift)
                    if torch.rand(1).item() < 0.5:
                        scale = 1.0 + (torch.rand(B_s, 1, 1, 1, 1, device=device) * 2 - 1) * self.scale_delta
                        approx_part = approx_part * scale
                    # Detail noise: simulates MRI thermal noise (high-frequency, spatially varying)
                    if torch.rand(1).item() < 0.5:
                        noise = torch.randn_like(detail_part) * self.noise_std
                        detail_part = detail_part + noise
                    sub = torch.cat([approx_part, detail_part], dim=1)

                raw_lll = sub[:, :1]          # LLL before SE
                sub = self.se_blocks[i](sub)  # per-level SE recalibration

                if target_size is None:
                    target_size = sub.shape[2:]

                if sub.shape[2:] == target_size:
                    all_bands.append(sub)
                else:
                    all_bands.append(
                        F.interpolate(sub, size=target_size,
                                      mode="trilinear", align_corners=False)
                    )

                # LLL → input for next level
                approx = sub[:, :1] if self.legacy_se_recursion else raw_lll

        x = torch.cat(all_bands, dim=1)   # [B, in_chans*8*actual_levels, D/2, H/2, W/2]
        return self.proj(x)




class WaveletDetailSkip(nn.Module):
    """
    Experiment 2 — extra decoder skip carrying stage-1 detail.

    In SwinUNETR the decoder's 48³ skip receives the patch-embedding output
    from before any attention; the stage-1 attention features only continue
    after patch merging (24³). This module turns those pre-merging stage-1
    features into an extra skip that is added to the decoder's 48³ skip.

    mode
        'plain'   : the stage-1 features themselves (control).
        'haar_hf' : only their high-frequency part — Haar DWT per channel,
                    LLL band zeroed, inverse DWT. Equals the features minus
                    their 2×2×2 block mean, i.e. exactly the detail that patch
                    merging has to compress.

    Both modes have identical parameters: a residual conv block followed by a
    zero-initialised 1×1×1 conv, so at initialisation the model is exactly the
    baseline and any contribution of the skip is learned.

    Input/Output shape: [B, C, D, H, W]
    """

    def __init__(self, channels: int, mode: str = "haar_hf"):
        super().__init__()
        from monai.networks.blocks import UnetrBasicBlock
        if mode not in ("plain", "haar_hf"):
            raise ValueError(f"Unknown detail skip mode '{mode}'")
        self.mode = mode
        if mode == "haar_hf":
            self.dwt  = HaarDWT3d()
            self.idwt = HaarIDWT3d()
        self.block = UnetrBasicBlock(spatial_dims=3, in_channels=channels, out_channels=channels,
                                     kernel_size=3, stride=1, norm_name="instance", res_block=True)
        self.out = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def high_pass(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        bands = self.dwt(x.reshape(B * C, 1, D, H, W))
        bands = torch.cat([torch.zeros_like(bands[:, :1]), bands[:, 1:]], dim=1)  # drop LLL
        return self.idwt(bands).reshape(B, C, D, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "haar_hf":
            x = self.high_pass(x)
        return self.out(self.block(x))
