from torch.utils.data import DataLoader
from monai.data import CacheDataset, Dataset
from build_datalist import build_train_list, build_train_list_mm
from transforms import (
    get_train_transforms, get_val_transforms,
    get_train_transforms_mm, get_val_transforms_mm,
)


def get_loaders(root, train_patients, val_patients, multimodal: bool = False,
                intensity_aug: bool = False, seed: int | None = None,
                sampler: str = "posneg", cache: bool = False):
    if multimodal:
        build = build_train_list_mm
        train_tf = get_train_transforms_mm(intensity_aug=intensity_aug, sampler=sampler)
        val_tf   = get_val_transforms_mm()
    else:
        build = build_train_list
        train_tf = get_train_transforms(intensity_aug=intensity_aug, sampler=sampler)
        val_tf   = get_val_transforms()

    if cache:
        # Deterministic preprocessing (load, resample, normalise, crop weights)
        # is computed once and kept in RAM; random transforms still run per epoch.
        train_ds = CacheDataset(build(root, train_patients), transform=train_tf, num_workers=4)
        val_ds   = CacheDataset(build(root, val_patients),   transform=val_tf,   num_workers=4)
    else:
        train_ds = Dataset(build(root, train_patients), transform=train_tf)
        val_ds   = Dataset(build(root, val_patients),   transform=val_tf)
    if seed is not None:
        train_tf.set_random_state(seed=seed)   # crops / flips / intensity aug

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=0)

    return train_loader, val_loader
