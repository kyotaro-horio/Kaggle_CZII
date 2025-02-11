import torch
import math

from monai.data import DataLoader, Dataset, CacheDataset
from monai.transforms import (
    Compose, 
    EnsureChannelFirstd, 
    NormalizeIntensityd,
    Orientationd,  
    AsDiscrete,  
    RandCropByLabelClassesd,
    RandFlipd, 
    RandRotate90d, 
    RandAffined,
    RandAdjustContrastd,  
    RandStdShiftIntensityd, 
    RandGaussianSmoothd, 
    RandCropByPosNegLabeld, 
    RandGaussianNoised, 
    RandZoomd, 
    Rand3DElasticd, 
    RandShiftIntensityd, 
    OneOf, 
)

def generate_trn_val_dataloader(trn_files, val_files, cfg):

    non_random_transforms = Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS")
    ])

    #https://www.kaggle.com/competitions/blood-vessel-segmentation/discussion/475522
    random_transforms = Compose([
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(cfg.patch_size[0], cfg.patch_size[1], cfg.patch_size[2]),  # adapt to your GPU memory, patch size
            pos=1,
            neg=1,
            num_samples=cfg.batch_size,  # how many patches to generate per volume
            image_key="image",
            image_threshold=0
        ),
        # RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 2]),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[1, 2]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0), 
        Rand3DElasticd(
            keys=["image", "label"], prob=0.2,
            sigma_range=(2, 4), magnitude_range=(1, 2),
            mode=("bilinear", "nearest"), rotate_range=(0, 0, 0)  # or small angles if you want random rotations here
        ),
        RandGaussianSmoothd(
            keys=["image"], prob=0.5,  
            sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5), 
        ), 
        RandStdShiftIntensityd(keys=["image"], prob=0.5, factors=0.1),
        RandAdjustContrastd(keys=["image"], prob=0.7, gamma=[0.8, 1.2]), 
    ])

    trn_ds = CacheDataset(data=trn_files, transform=non_random_transforms, cache_rate=1.0)
    trn_ds = Dataset(data=trn_ds, transform=random_transforms)
    trn_loader = DataLoader(
        trn_ds,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    val_transforms = Compose([
        # EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        # NormalizeIntensityd(keys="image"),
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=[cfg.patch_size[0], cfg.patch_size[1], cfg.patch_size[2]],
            num_classes=7,
            num_samples=cfg.batch_size,  # Use 1 to get a single, consistent crop per image
        ),
    ])

    val_ds = CacheDataset(data=val_files, transform=non_random_transforms, cache_rate=1.0)
    # val_ds = Dataset(data=val_ds, transform=random_transforms)
    val_ds = Dataset(data=val_ds, transform=val_transforms) # w/o transform in validation
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,  # Ensure the data order remains consistent
    )

    return trn_loader, val_loader