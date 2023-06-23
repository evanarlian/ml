from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


class ImageNetMini(Dataset):
    def __init__(self, root_folder: str | Path, aug) -> None:
        super().__init__()
        self.root_folder = Path(root_folder)
        self.aug = aug
        self.class_names = []
        self.classes = []
        self.image_paths = []
        for i, folder in enumerate(sorted(self.root_folder.iterdir())):
            content = list(folder.iterdir())
            self.classes += [i] * len(content)
            self.image_paths += content
            self.class_names.append(folder.name)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, i):
        im = Image.open(self.image_paths[i]).convert("RGB")
        inp = self.aug(im)
        return inp, self.classes[i]

    def create_dataloader(self, batch_size: int, num_workers: int, shuffle: bool):
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
        )


def get_dataset_mean(root_folder: str | Path, cache_file: str | Path) -> Tensor:
    # for subtracting mean over the whole dataset
    # resulting tensor will be 3x256x256 (chw)
    root_folder = Path(root_folder)
    cache_file = Path(cache_file)
    if cache_file.exists():
        print(f"Loading dataset mean cache from {cache_file}")
        return torch.load(cache_file)
    cropper = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(256),
            T.ToTensor(),
        ]
    )
    buffer = torch.zeros(3, 256, 256)
    imgs = list(root_folder.rglob("*.*"))
    for image_path in tqdm(imgs, desc="Calculating dataset mean"):
        im = Image.open(image_path).convert("RGB")
        imtensor = cropper(im)
        buffer += imtensor
    buffer = buffer / len(imgs)
    print(f"Saving dataset mean cache to {cache_file}")
    torch.save(buffer, cache_file)
    return buffer


def make_train_aug(train_dataset_mean: Tensor):
    return T.Compose(
        [
            T.Resize(256),
            T.RandomCrop(256),
            T.RandomHorizontalFlip(),
            T.ColorJitter(),
            T.ToTensor(),
            T.Lambda(lambda im: im - train_dataset_mean),  # alexnet scaling
            T.RandomCrop(224),
        ]
    )


def make_val_aug(train_dataset_mean: Tensor):
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(256),
            T.ToTensor(),
            T.Lambda(lambda im: im - train_dataset_mean),  # alexnet scaling
            T.CenterCrop(224),
        ]
    )
