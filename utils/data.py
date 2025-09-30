import torch
import numpy as np
import albumentations as A
import os
from fastai.vision.all import (
    Transform, PILImage, ImageDataLoaders, Resize, aug_transforms, 
    Normalize, imagenet_stats
)

class AlbumentationsTransform(Transform):
    def __init__(self, aug): self.aug = aug
    def encodes(self, img: PILImage):
        aug_img = self.aug(image=np.array(img))['image']
        return PILImage.create(aug_img)

def get_augmentations():
    """
    Mendefinisikan dan mengembalikan pipeline augmentasi Albumentations.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=25, p=0.8),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.2, p=1.0),
            A.GridDistortion(p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
        ], p=0.3),
        A.OneOf([
            A.RandomRain(brightness_coefficient=0.8, drop_width=1, blur_value=3, p=1.0),
            A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, p=1.0),
            A.RandomSunFlare(p=1.0)
        ], p=0.2),
        A.RandomBrightnessContrast(p=0.8),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5),
        A.Solarize(p=0.1),
        A.OneOf([A.GaussianBlur(p=1.0), A.MotionBlur(p=1.0)], p=0.2),
        A.GridDropout(ratio=0.5, p=0.3)
    ])

def create_dataloaders(train_path, img_size, batch_size=32, valid_pct=0.2, seed=42):
    """
    Membuat dan mengembalikan DataLoaders untuk ukuran gambar tertentu.

    Args:
        train_path (Path): Path ke data training.
        img_size (int): Ukuran gambar (misal: 128, 224, 384).
        batch_size (int): Ukuran batch.
        valid_pct (float): Persentase data untuk validasi.
        seed (int): Seed untuk reproduktifitas.

    Returns:
        DataLoaders: Objek DataLoaders Fastai.
    """
    item_resize = int(img_size * 1.5)
    batch_tfms = [
        *aug_transforms(size=img_size, min_scale=0.5),
        AlbumentationsTransform(get_augmentations()),
        Normalize.from_stats(*imagenet_stats)
    ]
    
    dls = ImageDataLoaders.from_folder(
        train_path,
        valid_pct=valid_pct,
        seed=seed,
        bs=batch_size,
        item_tfms=Resize(item_resize),
        batch_tfms=batch_tfms
    )
    return dls
