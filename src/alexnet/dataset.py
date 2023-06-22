from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


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
