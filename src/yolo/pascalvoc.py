import xml.etree.ElementTree as ET
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def get_pascal_voc_mapping(annotations_dir: Path) -> tuple[dict, dict]:
    classes = set()
    for xml_path in annotations_dir.iterdir():
        root = ET.parse(xml_path).getroot()
        for detection in root.findall("object"):
            classes.add(detection.find("name").text),
    class2id = {c: i for i, c in enumerate(sorted(classes))}
    id2class = {i: c for c, i in class2id.items()}
    return class2id, id2class


def get_train_val_test_aug() -> tuple:
    train_aug = A.Compose(
        [
            A.Resize(448, 448),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_names", "is_difficult"],
        ),
    )
    val_aug = A.Compose(
        [
            A.Resize(448, 448),
            A.Normalize(),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_names", "is_difficult"],
        ),
    )
    # test_aug is the same as val_aug, but no bbox settings
    test_aug = A.Compose(
        [
            A.Resize(448, 448),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    return train_aug, val_aug, test_aug


class PascalVoc(Dataset):
    def __init__(
        self,
        annotation_paths: list,
        images_dir: Path,
        class2id: dict,
        id2class: dict,
        aug: A.Compose,
        include_difficult: bool,
        S: int,
        B: int,
        C: int,
    ):
        self.annotation_paths = annotation_paths
        self.images_dir = Path(images_dir)
        self.class2id = class2id
        self.id2class = id2class
        self.aug = aug
        self.include_difficult = include_difficult
        self.S = S
        self.B = B
        self.C = C
        assert self.C == len(self.class2id)

    def __len__(self):
        return len(self.annotation_paths)

    def __repr__(self):
        return f"PascalVoc(n={len(self)})"

    def _pascalvoc_to_yolo_format(self, pascalvoc_bbox: tuple, image_size: tuple):
        # pascalvoc: x_min, y_min, x_max, y_max (absolute)
        x_min, y_min, x_max, y_max = pascalvoc_bbox
        im_width, im_height = image_size
        # yolo: x_center, y_center, w, h (normalized)
        x_center = (x_min + x_max) / 2 / im_width
        y_center = (y_min + y_max) / 2 / im_height
        bbox_width = (x_max - x_min) / im_width
        bbox_height = (y_max - y_min) / im_height
        return x_center, y_center, bbox_width, bbox_height

    def _yolo_to_pascalvoc_format(self, yolo_bbox: tuple, image_size: tuple):
        # yolo format
        x_center, y_center, bbox_width, bbox_height = yolo_bbox
        im_width, im_height = image_size
        # pascalvoc format, this is also pil rect format
        x_min = (x_center - (bbox_width / 2)) * im_width
        y_min = (y_center - (bbox_height / 2)) * im_height
        x_max = (x_center + (bbox_width / 2)) * im_width
        y_max = (y_center + (bbox_height / 2)) * im_height
        return x_min, y_min, x_max, y_max

    def _parse_xml(self, xml_path: Path):
        root = ET.parse(xml_path).getroot()
        d = {
            "filename": root.find("filename").text,
            "width": int(root.find("size").find("width").text),
            "height": int(root.find("size").find("height").text),
            "class_names": [],
            "bboxes": [],
            "is_difficult": [],
        }
        for detection in root.findall("object"):
            is_difficult = bool(int(detection.find("difficult").text))
            if not self.include_difficult and is_difficult:
                continue
            d["class_names"].append(detection.find("name").text)
            d["is_difficult"].append(is_difficult)
            bbox = detection.find("bndbox")
            d["bboxes"].append(
                (
                    float(bbox.find("xmin").text),
                    float(bbox.find("ymin").text),
                    float(bbox.find("xmax").text),
                    float(bbox.find("ymax").text),
                )
            )
        return d

    def collate_fn(self, batch: list) -> dict:
        d = {}
        d["image_tensor"] = torch.stack([b["image_tensor"] for b in batch])
        d["bbox_label"] = torch.stack([b["bbox_label"] for b in batch])
        d["objectness_label"] = torch.stack([b["objectness_label"] for b in batch])
        d["class_label"] = torch.stack([b["class_label"] for b in batch])
        d["annot_path"] = [b["annot_path"] for b in batch]
        d["image_path"] = [b["image_path"] for b in batch]
        d["image_pil"] = [b["image_pil"] for b in batch]
        d["class_names"] = [b["class_names"] for b in batch]
        d["class_ids"] = [b["class_ids"] for b in batch]
        d["pascalvoc_bboxes"] = [b["pascalvoc_bboxes"] for b in batch]
        d["yolo_bboxes"] = [b["yolo_bboxes"] for b in batch]
        d["is_difficult"] = [b["is_difficult"] for b in batch]
        return d

    def __getitem__(self, i):
        # parse pascalvoc xml
        annot_path = self.annotation_paths[i]
        parsed = self._parse_xml(annot_path)
        # load image-level data
        image_path = self.images_dir / parsed["filename"]
        image_pil = Image.open(image_path)
        image_width, image_height = image_pil.size
        assert (image_width, image_height) == (parsed["width"], parsed["height"])
        # augment image and bbox
        transformed = self.aug(
            image=np.array(image_pil),
            bboxes=parsed["bboxes"],
            class_names=parsed["class_names"],
            is_difficult=parsed["is_difficult"],
        )
        class_names = transformed["class_names"]
        class_ids = [self.class2id[c] for c in parsed["class_names"]]
        pascalvoc_bboxes = transformed["bboxes"]
        yolo_bboxes = [
            self._pascalvoc_to_yolo_format(bbox, (448, 448))
            for bbox in pascalvoc_bboxes
        ]
        is_difficult = transformed["is_difficult"]
        # construct separate tensors for class, objectness, and bbox label
        # there are extra dimension with length 1 to enable easy broadcasting
        # bbox label: (S x S x 1 x 4), just need 1 bbox because of 1 bbox per grid
        # objectness label: (S x S x 1), object present or not
        # class label: (S x S x C), need to be one-hot because of mse loss
        bbox_label = torch.zeros(self.S, self.S, 1, 4)
        objectness_label = torch.zeros(self.S, self.S, 1)
        class_label = torch.zeros(self.S, self.S, self.C)
        for class_id, bbox in zip(class_ids, yolo_bboxes):
            # fill into the responsible grid
            x_center, y_center, bbox_width, bbox_height = bbox
            grid_x = int(x_center * self.S)
            grid_y = int(y_center * self.S)
            # NOTE the bbox can be overwritten, since only one per grid
            bbox_label[grid_y, grid_x, 0] = torch.tensor(bbox)
            objectness_label[grid_y, grid_x, 0] = 1.0
            class_label[grid_y, grid_x, class_id] = 1.0
        # most items are transformed, i.e. might be changed because of augmentations
        d = {
            "image_tensor": transformed["image"],
            "bbox_label": bbox_label,
            "objectness_label": objectness_label,
            "class_label": class_label,
            "annot_path": annot_path,  # NOT transformed
            "image_path": image_path,  # NOT transformed
            "image_pil": image_pil,  # NOT transformed
            "class_names": class_names,
            "class_ids": class_ids,
            "pascalvoc_bboxes": pascalvoc_bboxes,
            "yolo_bboxes": yolo_bboxes,
            "is_difficult": is_difficult,
        }
        return d

    def create_dataloader(self, batch_size: int, shuffle: bool, num_workers: int):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )


def build_pascalvoc(
    pascalvoc_dir: Path | str,
    include_difficult=False,
    S=7,
    B=2,
    C=20,
):
    pascalvoc_dir = Path(pascalvoc_dir)
    annot_dir = Path(pascalvoc_dir / "Annotations")
    image_dir = Path(pascalvoc_dir / "JPEGImages")
    class2id, id2class = get_pascal_voc_mapping(annot_dir)
    annot_paths = sorted(list(annot_dir.iterdir()))
    train_annots, val_annots = train_test_split(
        annot_paths, test_size=0.3, shuffle=False
    )
    train_aug, val_aug, _ = get_train_val_test_aug()
    train_ds = PascalVoc(
        train_annots,
        image_dir,
        class2id,
        id2class,
        train_aug,
        include_difficult=include_difficult,
        S=S,
        B=B,
        C=C,
    )
    val_ds = PascalVoc(
        val_annots,
        image_dir,
        class2id,
        id2class,
        val_aug,
        include_difficult=include_difficult,
        S=S,
        B=B,
        C=C,
    )
    return train_ds, val_ds, class2id, id2class


def main():
    train_dataset, val_dataset, class2id, id2class = build_pascalvoc("data/VOC2012")
    print(train_dataset)
    print(val_dataset)
    print(class2id)
    print(id2class)
    print(train_dataset[0].keys())


if __name__ == "__main__":
    main()
