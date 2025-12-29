from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

import monai
from monai.data import CacheDataset, Dataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)

import pytorch_lightning as pl
from typing import Optional


MSD_TAR_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar"


@dataclass(frozen=True)
class ProstateDataMeta:
    root_dir: str
    task_dir: str
    dataset_json: str
    train_cases: int
    test_cases: int


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_and_extract_msd_prostate(root_dir: str, url: str = MSD_TAR_URL) -> str:

    _ensure_dir(root_dir)
    task_dir = os.path.join(root_dir, "Task05_Prostate")
    tar_path = os.path.join(root_dir, "Task05_Prostate.tar")

    if os.path.exists(task_dir) and os.path.exists(os.path.join(task_dir, "dataset.json")):
        return task_dir

    if not os.path.exists(tar_path):
        print(f"[dataloader.py] Downloading: {url}")
        urllib.request.urlretrieve(url, tar_path)
        print(f"[dataloader.py] Saved: {tar_path}")

    print(f"[dataloader.py] Extracting tar to: {root_dir}")
    ret = os.system(f'tar -xf "{tar_path}" -C "{root_dir}"')
    if ret != 0:
        raise RuntimeError(
            "Extraction failed. If you're on Windows without tar, "
            "extract Task05_Prostate.tar manually into root_dir."
        )

    if not os.path.exists(os.path.join(task_dir, "dataset.json")):
        raise FileNotFoundError(f"Expected dataset.json in {task_dir} after extraction.")

    return task_dir


def _make_abs(task_dir: str, path_rel: str) -> str:
    path_rel = path_rel.replace("./", "")
    return os.path.join(task_dir, path_rel)


def load_file_lists(task_dir: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:

    dataset_json = os.path.join(task_dir, "dataset.json")
    with open(dataset_json, "r", encoding="utf-8") as f:
        ds = json.load(f)

    train_files = [
        {"image": _make_abs(task_dir, item["image"]), "label": _make_abs(task_dir, item["label"])}
        for item in ds.get("training", [])
    ]
    test_files = [{"image": _make_abs(task_dir, p)} for p in ds.get("test", [])]
    return train_files, test_files


def split_train_val(
    train_files: Sequence[Dict[str, str]],
    val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if not (0.0 < val_frac < 1.0):
        raise ValueError("val_frac must be in (0, 1).")

    rng = torch.Generator().manual_seed(seed)
    n_total = len(train_files)
    n_val = int(round(n_total * val_frac))
    n_val = max(1, n_val)
    n_train = n_total - n_val
    if n_train <= 0:
        raise ValueError("val_frac too large; would leave no training samples.")

    perm = torch.randperm(n_total, generator=rng).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_list = [train_files[i] for i in train_idx]
    val_list = [train_files[i] for i in val_idx]
    return train_list, val_list


def build_transforms(
    spacing: Tuple[float, float, float] = (1.5, 1.5, 2.0),
    roi_size: Tuple[int, int, int] = (128, 128, 64),
    augment: bool = True,
) -> Tuple[Compose, Compose]:

    base = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]



    pad = [SpatialPadd(keys=["image", "label"], spatial_size=roi_size)]

    crop = [
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            pos=1,
            neg=1,
            num_samples=1,  
            image_key="image",
            image_threshold=0,
        )
    ]

    aug = []
    if augment:
        aug = [
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
            RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.5),
        ]

    train_transforms = Compose(base + pad + crop + aug + [EnsureTyped(keys=["image", "label"])])
    val_transforms = Compose(base + [EnsureTyped(keys=["image", "label"])])

    return train_transforms, val_transforms


def build_test_transforms(
    spacing: Tuple[float, float, float] = (1.5, 1.5, 2.0),
) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=spacing, mode=("bilinear",)),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image"], source_key="image"),
            EnsureTyped(keys=["image"]),
        ]
    )


def create_dataloaders(
    root_dir: str = "/content/msd",
    spacing: Tuple[float, float, float] = (1.5, 1.5, 2.0),
    roi_size: Tuple[int, int, int] = (128, 128, 64),
    batch_size: int = 2,
    num_workers: int = 2,
    val_frac: float = 0.2,
    augment: bool = True,
    cache_dataset: bool = False,
    cache_rate: float = 1.0,
    seed: int = 42,
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, ProstateDataMeta]:

    monai.utils.set_determinism(seed=seed)

    task_dir = download_and_extract_msd_prostate(root_dir=root_dir)
    train_files, test_files = load_file_lists(task_dir)

    train_list, val_list = split_train_val(train_files, val_frac=val_frac, seed=seed)

    train_tfms, val_tfms = build_transforms(spacing=spacing, roi_size=roi_size, augment=augment)
    test_tfms = build_test_transforms(spacing=spacing)

    ds_cls = CacheDataset if cache_dataset else Dataset

    if cache_dataset:
        train_ds = ds_cls(train_list, transform=train_tfms, cache_rate=cache_rate, num_workers=num_workers)
        val_ds = ds_cls(val_list, transform=val_tfms, cache_rate=cache_rate, num_workers=num_workers)
        test_ds = ds_cls(test_files, transform=test_tfms, cache_rate=cache_rate, num_workers=num_workers)
    else:
        train_ds = ds_cls(train_list, transform=train_tfms)
        val_ds = ds_cls(val_list, transform=val_tfms)
        test_ds = ds_cls(test_files, transform=test_tfms)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    meta = ProstateDataMeta(
        root_dir=root_dir,
        task_dir=task_dir,
        dataset_json=os.path.join(task_dir, "dataset.json"),
        train_cases=len(train_files),
        test_cases=len(test_files),
    )

    return train_loader, val_loader, test_loader, meta



class ProstateDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int = 2,
        val_frac: float = 0.2,
        num_workers: int = 4,
        augment: bool = True,
        roi_size=(128, 128, 64),
        spacing=(1.5, 1.5, 2.0),
        cache_dataset=False,
        cache_rate=1.0,
        seed: int = 42
    ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.num_workers = num_workers
        self.augment = augment
        self.roi_size = roi_size
        self.spacing = spacing
        self.cache_dataset = cache_dataset
        self.cache_rate = cache_rate
        self.seed = seed

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.meta = None

    def setup(self, stage: Optional[str] = None):
        train, val, test, meta = create_dataloaders(
            root_dir=self.root_dir,
            batch_size=self.batch_size,
            val_frac=self.val_frac,
            num_workers=self.num_workers,
            augment=self.augment,
            roi_size=self.roi_size,
            spacing=self.spacing,
            cache_dataset=self.cache_dataset,
            cache_rate=self.cache_rate,
            seed=self.seed
        )
        self.train_loader, self.val_loader, self.test_loader, self.meta = train, val, test, meta

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader