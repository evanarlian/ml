# use accelerate, wandb, torchmetrics
# overfit one batch
# maybe just pass the config inside the trainer so it will become clean af (since there are 2 trainers)
# accelerate config??
# please please just use fp16 as default and also CUDA, that is the best

import pickle

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from IPython import embed
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class BackboneTrainer:
    def __init__(
        self,
        accelerator: Accelerator,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
    ):
        self.accelerator = accelerator
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        # TODO some torchmetrics stuff OR hf evaluate OR accelerate logger

    def train(self, train_loader: DataLoader):
        self.model.train()
        for batch in (pbar := tqdm(train_loader, desc="train", leave=False)):
            image, label = batch["image_tensor"], batch["class_id"]
            logits = self.model(image)
            loss = self.criterion(logits, label)
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            pbar.set_postfix({"loss": loss.item()})

    @torch.no_grad()
    def val(self, val_loader: DataLoader):
        self.model.eval()
        for batch in (pbar := tqdm(val_loader, desc="val", leave=False)):
            image, label = batch["image_tensor"], batch["class_id"]
            logits = self.model(image)
            loss = self.criterion(logits, label)
            pbar.set_postfix({"loss": loss.item()})

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
    ):
        for epoch in range(1, n_epochs + 1):
            print(f"{epoch}/{n_epochs}")
            self.train(train_loader)
            self.val(val_loader)

    def overfit_one_batch(self, train_loader: DataLoader):
        print("🔥 overfitting for one batch 🔥")
        self.model.train()
        batch = next(iter(train_loader))
        image, label = batch["image_tensor"], batch["class_id"]
        print(image.size())
        print(label)
        while True:
            logits = self.model(image)
            loss = self.criterion(logits, label)
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            print(loss.item())
