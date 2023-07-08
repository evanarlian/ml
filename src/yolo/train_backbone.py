import math
import random
from pathlib import Path

import accelerate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from backbone_trainer import BackboneTrainer
from imagenet100 import (
    ImageNet100,
    get_imagenet100_mappings,
    get_train_data,
    get_train_val_aug,
    get_val_data,
)
from torch import nn, optim
from yolo_model import YoloPretraining


def main():
    accelerate.utils.set_seed(14)
    accelerator = Accelerator(mixed_precision="fp16", log_with=None)

    # load data
    imagenet100_path = Path("data/imagenet100")
    code2id, id2name = get_imagenet100_mappings(imagenet100_path / "Labels.json")
    train_paths, train_codes = get_train_data(imagenet100_path)
    val_paths, val_codes = get_val_data(imagenet100_path)
    train_aug, val_aug = get_train_val_aug()
    train_dataset = ImageNet100(train_paths, train_codes, code2id, id2name, train_aug)
    val_dataset = ImageNet100(val_paths, val_codes, code2id, id2name, val_aug)
    train_loader = train_dataset.create_dataloader(32, True, 8)
    val_loader = val_dataset.create_dataloader(32, False, 8)

    # load model
    n_epochs = 5
    total_steps = len(train_loader) * n_epochs
    model = YoloPretraining(n_classes=100)  # 100 because of imagenet100
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, total_steps=total_steps
    )

    # accelerator stuff
    (
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
    ) = accelerator.prepare(
        model, criterion, optimizer, scheduler, train_loader, val_loader
    )

    trainer = BackboneTrainer(accelerator, model, criterion, optimizer, scheduler)
    trainer.overfit_one_batch(train_loader)


if __name__ == "__main__":
    main()
