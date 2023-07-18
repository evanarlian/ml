import xml.etree.ElementTree as ET
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def get_pascal_voc_mapping(annotations_dir: Path) -> dict:
    classes = set()
    for xml_path in annotations_dir.iterdir():
        root = ET.parse(xml_path).getroot()
        for detection in root.findall("object"):
            classes.add(detection.find("name").text),
    mapping = {v: k for k, v in enumerate(sorted(classes))}
    return mapping


def get_train_val_aug() -> tuple:
    train_aug = A.Compose(
        [
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_names"]),
    )  # TODO see more BboxParams
    val_aug = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    return train_aug, val_aug


class PascalVoc(Dataset):
    def __init__(
        self,
        annotation_paths: list,
        images_dir: Path,
        mapping: dict,
        aug: A.Compose,
        S: int,
        B: int,
        C: int,
    ):
        self.annotation_paths = annotation_paths
        self.images_dir = Path(images_dir)
        self.mapping = mapping
        self.aug = aug
        self.S = S  # S by S grid
        self.B = B  # bboxes per grid
        self.C = C  # classes per grid
        assert self.C == len(self.mapping)

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
        }
        for detection in root.findall("object"):
            d["class_names"].append(detection.find("name").text)
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

    def __getitem__(self, i, include_all=False):
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
        )
        class_names = transformed["class_names"]
        class_ids = [self.mapping[c] for c in parsed["class_names"]]
        pascalvoc_bboxes = transformed["bboxes"]
        yolo_bboxes = [
            self._pascalvoc_to_yolo_format(bbox, image_pil.size)
            for bbox in pascalvoc_bboxes
        ]
        # construct separate tensors for class label and bbox label
        # bbox label: (S x S x 4), just need 1 bbox because of 1 bbox per grid
        # objectness label: (S x S), object present or not
        # class label: (S x S x C), need to be one-hot because of mse loss
        bbox_label = torch.zeros(self.S, self.S, 4)
        objectness_label = torch.zeros(self.S, self.S)
        class_label = torch.zeros(self.S, self.S, self.C)
        grid_sz = 1.0 / self.S
        for class_id, bbox in zip(class_ids, yolo_bboxes):
            # fill into the responsible grid
            x_center, y_center, bbox_width, bbox_height = bbox
            grid_x = int(x_center / grid_sz)
            grid_y = int(y_center / grid_sz)
            # NOTE the bbox can be overwritten, only one per grid!
            class_label[grid_y, grid_x, class_id] = 1.0
            objectness_label[grid_y, grid_x] = 1
            bbox_label[grid_y, grid_x] = torch.tensor(bbox)
        d = {
            "image_tensor": transformed["image"],
            "bbox_label": bbox_label,
            "objectness_label": objectness_label,
            "class_label": class_label,
        }
        if include_all:
            d |= {
                "annot_path": annot_path,
                "image_path": image_path,
                "image_pil": image_pil,  # NOT transformed
                "class_names": class_names,  # transformed
                "class_ids": class_ids,  # transformed
                "pascalvoc_bboxes": pascalvoc_bboxes,  # transformed
                "yolo_bboxes": yolo_bboxes,  # transformed
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
        )


def build_pascalvoc(pascalvoc_dir: Path | str, S=7, B=2, C=20):
    pascalvoc_dir = Path(pascalvoc_dir)
    annot_dir = Path(pascalvoc_dir / "Annotations")
    image_dir = Path(pascalvoc_dir / "JPEGImages")
    mapping = get_pascal_voc_mapping(annot_dir)
    annot_paths = list(annot_dir.iterdir())
    train_annots, val_annots = train_test_split(annot_paths, test_size=0.3)
    train_aug, val_aug = get_train_val_aug()
    train_ds = PascalVoc(train_annots, image_dir, mapping, train_aug, S=S, B=B, C=C)
    val_ds = PascalVoc(val_annots, image_dir, mapping, val_aug, S=S, B=B, C=C)
    return train_ds, val_ds


def main():
    train_dataset, val_dataset = build_pascalvoc("data/pascalvoc2012/voc2012/VOC2012")
    print(train_dataset)
    print(val_dataset)


if __name__ == "__main__":
    main()
