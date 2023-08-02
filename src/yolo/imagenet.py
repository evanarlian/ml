import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


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


class ImageNet(Dataset):
    def __init__(self, dataset: HFDataset, aug: A.Compose):
        self.dataset = dataset
        self.classnames = dataset.features["label"]
        self.aug = aug

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return f"ImageNet(n={len(self)})"

    def __getitem__(self, i, include_pil=False):
        sample = self.dataset[i]
        class_id = sample["label"]
        class_name = self.classnames.int2str(class_id)
        d = {
            "class_id": class_id,
            "class_name": class_name,
            "image_tensor": self.aug(image=np.array(sample["image"]))["image"],
        }
        if include_pil:
            d["image_pil"] = sample["image"]
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


def build_imagenet():
    train_aug, val_aug = get_train_val_aug()
    ds = load_dataset("evanarlian/imagenet_1k_resized_256")
    train_dataset = ImageNet(ds["train"], train_aug)
    val_dataset = ImageNet(ds["val"], val_aug)
    return train_dataset, val_dataset


def main():
    train_dataset, val_dataset = build_imagenet()
    print(train_dataset)
    print(train_dataset[0])
    print(val_dataset)
    print(val_dataset[0])


if __name__ == "__main__":
    main()
