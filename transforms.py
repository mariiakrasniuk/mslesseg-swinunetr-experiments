import numpy as np
import torch
from scipy import ndimage
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, NormalizeIntensityd, RandCropByPosNegLabeld, RandWeightedCropd,
    RandFlipd, RandScaleIntensityd, RandShiftIntensityd,
    RandGaussianNoised, EnsureTyped, ConcatItemsd, DeleteItemsd, MapTransform,
)

_MM = ["flair", "t1", "t2"]   # multimodal key names


class LesionBalancedWeightd(MapTransform):
    """
    Crop-centre weight map for RandWeightedCropd.

    Half of the sampling mass goes to lesions, split EQUALLY between lesion
    components (26-connected), so a 10-voxel lesion is as likely to be a crop
    centre as a 10,000-voxel one. The other half is spread uniformly over
    brain voxels (image != 0 after nonzero normalisation) that are not lesion.

    RandCropByPosNegLabeld instead picks a random lesion VOXEL, so lesions are
    sampled in proportion to their volume: on MSLesSeg small lesions (< 40 mm³)
    are 33% of lesions but < 3% of lesion volume.
    """

    def __init__(self, label_key="label", image_key="image", weight_key="weight"):
        super().__init__(keys=[label_key, image_key])
        self.label_key, self.image_key, self.weight_key = label_key, image_key, weight_key

    def __call__(self, data):
        d = dict(data)
        lab = np.asarray(d[self.label_key].detach().cpu())[0] > 0
        brain = np.asarray(d[self.image_key].detach().cpu()).any(axis=0)
        w = np.zeros(lab.shape, dtype=np.float32)
        comp, n = ndimage.label(lab, structure=np.ones((3, 3, 3), dtype=bool))
        bg = brain & ~lab
        if n:
            sizes = np.bincount(comp.ravel())
            per_voxel = (0.5 / n) / np.maximum(sizes, 1)
            w[lab] = per_voxel[comp[lab]]
            w[bg] = 0.5 / max(int(bg.sum()), 1)
        else:
            w[bg] = 1.0 / max(int(bg.sum()), 1)
        d[self.weight_key] = torch.from_numpy(w[None])
        return d


def _crop_transforms(sampler: str):
    """Random 4×96³ crops: 'posneg' (RP2/Phase 0) or 'lesion' (lesion-balanced)."""
    if sampler == "posneg":
        return [RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                                       spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=4)]
    if sampler == "lesion":
        return [LesionBalancedWeightd(),
                RandWeightedCropd(keys=["image", "label"], w_key="weight",
                                  spatial_size=(96, 96, 96), num_samples=4),
                DeleteItemsd(keys="weight")]
    raise ValueError(f"Unknown sampler '{sampler}'")


# ------------------------------------------------------------------
# Single-modality (FLAIR only)
# ------------------------------------------------------------------

def get_train_transforms(intensity_aug: bool = False, sampler: str = "posneg"):
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1, 1, 1),
                 mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True),
        *_crop_transforms(sampler),
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
    ]
    if intensity_aug:
        transforms += [
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            RandGaussianNoised(keys="image", prob=0.2, std=0.05),
        ]
    transforms.append(EnsureTyped(keys=["image", "label"]))
    return Compose(transforms)


def get_val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1, 1, 1),
                 mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True),
        EnsureTyped(keys=["image", "label"]),
    ])


# ------------------------------------------------------------------
# Multi-modal (FLAIR + T1 + T2)
# Each modality is loaded and normalised independently, then
# concatenated along the channel dim into a single "image" tensor.
# ------------------------------------------------------------------

def get_train_transforms_mm(intensity_aug: bool = False, sampler: str = "posneg"):
    transforms = [
        LoadImaged(keys=_MM + ["label"]),
        EnsureChannelFirstd(keys=_MM + ["label"]),
        Orientationd(keys=_MM + ["label"], axcodes="RAS"),
        Spacingd(keys=_MM + ["label"], pixdim=(1, 1, 1),
                 mode=["bilinear", "bilinear", "bilinear", "nearest"]),
        NormalizeIntensityd(keys=_MM, nonzero=True),
        ConcatItemsd(keys=_MM, name="image", dim=0),  # [3, D, H, W]
        DeleteItemsd(keys=_MM),
        *_crop_transforms(sampler),
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
    ]
    if intensity_aug:
        transforms += [
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            RandGaussianNoised(keys="image", prob=0.2, std=0.05),
        ]
    transforms.append(EnsureTyped(keys=["image", "label"]))
    return Compose(transforms)


def get_val_transforms_mm():
    return Compose([
        LoadImaged(keys=_MM + ["label"]),
        EnsureChannelFirstd(keys=_MM + ["label"]),
        Orientationd(keys=_MM + ["label"], axcodes="RAS"),
        Spacingd(keys=_MM + ["label"], pixdim=(1, 1, 1),
                 mode=["bilinear", "bilinear", "bilinear", "nearest"]),
        NormalizeIntensityd(keys=_MM, nonzero=True),
        ConcatItemsd(keys=_MM, name="image", dim=0),  # [3, D, H, W]
        DeleteItemsd(keys=_MM),
        EnsureTyped(keys=["image", "label"]),
    ])
