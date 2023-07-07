import json
from pathlib import Path

import torch
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


def get_train(dataset_root: Path) -> tuple[list, list]:
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


def get_val(dataset_root: Path) -> tuple[list, list]:
    image_paths = []
    image_codes = []
    for folder in (dataset_root / "val.X").iterdir():
        curr_images = list(folder.iterdir())
        image_paths += curr_images
        image_codes += [folder.name] * len(curr_images)
    return image_paths, image_codes


class ImageNet100(Dataset):
    def __init__(
        self, image_paths: list, image_codes: list, code2id: dict, id2name: dict
    ):
        self.image_paths = image_paths
        self.image_codes = image_codes
        self.code2id = code2id
        self.id2name = id2name

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        class_code = self.image_codes[i]
        class_id = self.code2id[class_code]
        class_name = self.id2name[class_id]
        d = {
            "image_path": image_path,
            "class_code": class_code,
            "class_id": class_id,
            "class_name": class_name,
        }
        # TODO load the actual image + aug
        return d

    def create_dataloader(self):
        raise NotImplementedError("TODO")
