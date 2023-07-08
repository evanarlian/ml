import json
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def get_imagenet100_mappings(json_path: Path) -> tuple[dict, dict]:
    code2id = {}
    id2name = {}
    with open(json_path, "r") as f:
        code2name = json.load(f)
    for i, (code, name) in enumerate(code2name.items()):
        code2id[code] = i
        id2name[i] = name.split(",")[0]  # shorten name
    return code2id, id2name


def get_train_data(dataset_root: Path) -> tuple[list, list]:
    image_paths = []
    image_codes = []
    for path in dataset_root.iterdir():
        if "train" not in path.name:
            continue
        for folder in path.iterdir():
            curr_images = list(folder.iterdir())
            image_paths += curr_images
            image_codes += [folder.name] * len(curr_images)
    return image_paths, image_codes


def get_val_data(dataset_root: Path) -> tuple[list, list]:
    image_paths = []
    image_codes = []
    for folder in (dataset_root / "val.X").iterdir():
        curr_images = list(folder.iterdir())
        image_paths += curr_images
        image_codes += [folder.name] * len(curr_images)
    return image_paths, image_codes


def get_train_val_aug() -> tuple:
    train_aug = A.Compose(
        [
            A.SmallestMaxSize(256),
            A.RandomCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    val_aug = A.Compose(
        [
            A.SmallestMaxSize(256),
            A.CenterCrop(224, 224),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    return train_aug, val_aug


class ImageNet100(Dataset):
    def __init__(
        self,
        image_paths: list,
        image_codes: list,
        code2id: dict,
        id2name: dict,
        aug: A.Compose,
    ):
        self.image_paths = image_paths
        self.image_codes = image_codes
        self.code2id = code2id
        self.id2name = id2name
        self.aug = aug

    def __len__(self):
        return len(self.image_paths)

    def __repr__(self):
        return f"ImageNet100(n={len(self)})"

    def __getitem__(self, i, include_pil=False):
        image_path = str(self.image_paths[i])
        class_code = self.image_codes[i]
        class_id = self.code2id[class_code]
        class_name = self.id2name[class_id]
        d = {
            "image_path": image_path,
            "class_code": class_code,
            "class_id": class_id,
            "class_name": class_name,
        }
        image_pil = Image.open(image_path).convert("RGB")
        image_tensor = self.aug(image=np.array(image_pil))["image"]
        if include_pil:
            d["image_pil"] = image_pil
        d["image_tensor"] = image_tensor
        return d

    def create_dataloader(self, batch_size: int, shuffle: bool, num_workers: int):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
        )
