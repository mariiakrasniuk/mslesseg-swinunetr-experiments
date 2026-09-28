import torch.nn as nn
from monai.networks.nets import SwinUNETR

from wavelet import (
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

    else:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from: "
            f"baseline, wavelet_a, wavelet_ml"
        )
