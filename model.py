import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

from wavelet import (
    WaveletDetailSkip,
    WaveletPatchEmbed,
    WaveletPatchEmbedML,
    WaveletUpsample,
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


class SwinUNETRWaveUp(SwinUNETR):
    """
    SwinUNETR whose 5 decoder upsamplings use wavelet synthesis
    (wavelet.WaveletUpsample) instead of transposed convolutions. Detail bands
    come from the encoder features at the target resolution that the standard
    decoder never receives: each transformer stage's output BEFORE patch
    merging (48³, 24³, 12³, 6³ for a 96³ patch) and the full-resolution conv
    features enc0 (96³).
    """

    def __init__(self, filters: str, **kwargs):
        super().__init__(**kwargs)
        if self.swinViT.use_v2:
            raise ValueError("wavelet upsampling is implemented for SwinUNETR v1 only")
        fs = kwargs["feature_size"]
        for i in range(1, 5):
            stage = getattr(self.swinViT, f"layers{i}")[0]
            stage.downsample = _CaptureInput(stage.downsample)
        # (decoder block, decoder channels in, encoder channels at target resolution)
        spec = [("decoder5", 16 * fs, 8 * fs), ("decoder4", 8 * fs, 4 * fs),
                ("decoder3", 4 * fs, 2 * fs), ("decoder2", 2 * fs, fs), ("decoder1", fs, fs)]
        self.wave_up = nn.ModuleDict()
        for name, dec_ch, enc_ch in spec:
            getattr(self, name).transp_conv = nn.Identity()      # replaced by wave_up[name]
            self.wave_up[name] = WaveletUpsample(dec_ch, enc_ch, filters=filters)

    def _pre_merge(self, i):
        x = getattr(self.swinViT, f"layers{i}")[0].downsample.captured     # [B, D, H, W, C]
        return self.swinViT.proj_out(x.permute(0, 4, 1, 2, 3).contiguous(), self.normalize)

    def _up(self, name, dec, enc_detail, skip):
        block = getattr(self, name)
        out = self.wave_up[name](dec, enc_detail)
        return block.conv_block(torch.cat((out, skip), dim=1))

    def forward(self, x_in):
        self._check_input_size(x_in.shape[2:])
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self._up("decoder5", dec4, self._pre_merge(4), hidden_states_out[3])
        dec2 = self._up("decoder4", dec3, self._pre_merge(3), enc3)
        dec1 = self._up("decoder3", dec2, self._pre_merge(2), enc2)
        dec0 = self._up("decoder2", dec1, self._pre_merge(1), enc1)
        out = self._up("decoder1", dec0, enc0, enc0)
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

    waveup_haar / waveup_haar_learn / waveup_rand_learn  [Experiment 5]
        All 5 decoder upsamplings use wavelet synthesis: decoder → coarse
        band, encoder features before patch merging → 7 detail bands.
        Filters fixed Haar / trainable from Haar / trainable from random
        (control with identical tensor flow and parameter count).

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

    elif variant in ("waveup_haar", "waveup_haar_learn", "waveup_rand_learn"):
        return SwinUNETRWaveUp(
            filters=variant[len("waveup_"):],
            spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
            feature_size=feature_size, use_checkpoint=use_checkpoint, use_v2=use_v2,
        )

    else:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from: baseline, wavelet_a, wavelet_ml, "
            f"detail_skip_plain, detail_skip_haar, waveup_haar, waveup_haar_learn, waveup_rand_learn"
        )
