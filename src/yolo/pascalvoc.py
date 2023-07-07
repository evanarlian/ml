import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


def get_pascal_voc_mapping(annotations_dir: Path) -> dict:
    classes = set()
    for xml_path in annotations_dir.iterdir():
        root = ET.parse(xml_path).getroot()
        for detection in root.findall("object"):
            classes.add(detection.find("name").text),
    mapping = {k: v for k, v in enumerate(sorted(classes))}
    return mapping


class PascalVocDataset(Dataset):
    def __init__(self, annotation_paths: list, images_dir: Path, mapping: dict):
        self.annotation_paths = annotation_paths
        self.images_dir = Path(images_dir)
        self.mapping = mapping

    def __len__(self):
        return len(self.annotation_paths)

    def _parse_xml(self, xml_path: Path):
        root = ET.parse(xml_path).getroot()
        d = {
            "filename": root.find("filename").text,
            "width": root.find("size").find("width").text,
            "height": root.find("size").find("height").text,
            "detections": [],
        }
        for detection in root.findall("object"):
            bbox = detection.find("bndbox")
            d["detections"].append(
                {
                    "classname": detection.find("name").text,
                    "xmin": float(bbox.find("xmin").text),
                    "ymin": float(bbox.find("ymin").text),
                    "xmax": float(bbox.find("xmax").text),
                    "ymax": float(bbox.find("ymax").text),
                }
            )
        return d

    def __getitem__(self, i):
        annot_path = self.annotation_paths[i]
        parsed = self._parse_xml(annot_path)
        image_path = self.images_dir / parsed["filename"]
        return parsed, image_path
