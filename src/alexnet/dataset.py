import torchvision.transforms as T
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset


class ImageNet(Dataset):
    def __init__(self, dataset: HFDataset, aug) -> None:
        super().__init__()
        self.dataset = dataset
        self.aug = aug

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i):
        sample = self.dataset[i]
        image_tensor = self.aug(sample["image"])
        return image_tensor, sample["label"]

    def create_dataloader(self, batch_size: int, num_workers: int, shuffle: bool):
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
        )


def make_train_aug():
    return T.Compose(
        [
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ColorJitter(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def make_val_aug():
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
