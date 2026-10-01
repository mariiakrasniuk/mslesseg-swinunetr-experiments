import torch.nn as nn
from monai.networks.nets import SwinUNETR

from wavelet import (
    WaveletDetailSkip,
    WaveletPatchEmbed,
    WaveletPatchEmbedML,
)


def _base_swinunetr(in_channels: int, out_channels: int, feature_size: int,
                    use_checkpoint: bool, use_v2: bool = False) -> nn.Module:
    return SwinUNETR(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
        use_v2=use_v2,
    )


class _CaptureInput(nn.Module):
    """Wraps a module and keeps a reference to its last input."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.captured = None

    def forward(self, x):
        self.captured = x
        return self.inner(x)


class SwinUNETRDetailSkip(SwinUNETR):
    """
    SwinUNETR with an extra skip from the stage-1 transformer features
    (before patch merging, 48³ for a 96³ patch) to the decoder's 48³ level.
    See wavelet.WaveletDetailSkip for the 'plain' / 'haar_hf' modes.
    """

    def __init__(self, detail_mode: str, **kwargs):
        super().__init__(**kwargs)
        if self.swinViT.use_v2:
            raise ValueError("detail skip is implemented for SwinUNETR v1 only")
        stage1 = self.swinViT.layers1[0]
        stage1.downsample = _CaptureInput(stage1.downsample)
        self.detail_skip = WaveletDetailSkip(kwargs["feature_size"], mode=detail_mode)

    def forward(self, x_in):
        self._check_input_size(x_in.shape[2:])
        hidden_states_out = self.swinViT(x_in, self.normalize)
        # stage-1 features before patch merging: [B, D, H, W, C] -> [B, C, D, H, W]
        stage1 = self.swinViT.layers1[0].downsample.captured.permute(0, 4, 1, 2, 3).contiguous()
        stage1 = self.swinViT.proj_out(stage1, self.normalize)

        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0]) + self.detail_skip(stage1)
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        return self.out(out)


def build_model(
    variant: str,
    in_channels: int = 1,
    out_channels: int = 1,
    feature_size: int = 48,
    use_checkpoint: bool = False,
    roi_size: int = 96,
    wavelet: str = "haar",
    levels: int = 1,
    use_v2: bool = False,
    coeff_aug: bool = False,
    legacy_se_recursion: bool = False,
) -> nn.Module:
    """
    Factory that returns the model for a given variant name.

    Variants
    --------
    baseline
        Standard MONAI SwinUNETR, unchanged.  wavelet/levels are ignored.

    wavelet_a
        Patch embedding replaced by a single-level 3D Haar DWT followed by
        a 1x1x1 learned projection. Parameter count identical to baseline (432).

    wavelet_ml  [ablation variant]
        Patch embedding replaced by WaveletPatchEmbedML parameterised by the
        `wavelet` and `levels` arguments. Used for the family × depth ablation.

        Supported wavelet families : 'haar', 'db2', 'sym4'
        Supported levels           : 1, 2, 3
        legacy_se_recursion=True reproduces RP2 (next level decomposes the
        SE-rescaled LLL); needed only to evaluate RP2 checkpoints.

    detail_skip_plain / detail_skip_haar  [Experiment 2]
        Baseline SwinUNETR plus an extra skip from the stage-1 transformer
        features (before patch merging) to the decoder's 48³ level: the plain
        features (control) or only their Haar high-frequency part.
        +127k parameters; identical to the baseline at initialisation.

        Parameter count (embed_dim=48):
            levels=1 :  64  (SE)  + 432 (proj) =  496
            levels=2 : 128  (SE)  + 816 (proj) =  944
            levels=3 : 192  (SE)  + 1200 (proj) = 1392

    """
    if variant == "baseline":
        return _base_swinunetr(in_channels, out_channels, feature_size, use_checkpoint, use_v2)

    elif variant == "wavelet_a":
        model = _base_swinunetr(in_channels, out_channels, feature_size, use_checkpoint, use_v2)
        model.swinViT.patch_embed = WaveletPatchEmbed(
            in_chans=in_channels, embed_dim=feature_size
        )
        return model

    elif variant == "wavelet_ml":
        model = _base_swinunetr(in_channels, out_channels, feature_size, use_checkpoint, use_v2)
        model.swinViT.patch_embed = WaveletPatchEmbedML(
            in_chans=in_channels, embed_dim=feature_size,
            levels=levels, wavelet=wavelet, coeff_aug=coeff_aug,
            legacy_se_recursion=legacy_se_recursion,
        )
        return model

    elif variant in ("detail_skip_plain", "detail_skip_haar"):
        return SwinUNETRDetailSkip(
            detail_mode="plain" if variant == "detail_skip_plain" else "haar_hf",
            spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
            feature_size=feature_size, use_checkpoint=use_checkpoint, use_v2=use_v2,
        )

    else:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from: "
            f"baseline, wavelet_a, wavelet_ml, detail_skip_plain, detail_skip_haar"
        )
