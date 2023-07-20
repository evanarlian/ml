import torch
from accelerate import Accelerator
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm.auto import tqdm


class DetectorTrainer:
    def __init__(
        self,
        accelerator: Accelerator,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        val_loss_metric: Metric,
    ):
        self.accelerator = accelerator
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        # metrics related
        self.val_loss_metric = val_loss_metric
        self.global_step = 0
        self.best_loss = float("inf")

    def train(self, train_loader: DataLoader):
        self.model.train()
        for batch in (pbar := tqdm(train_loader, desc="train", leave=False)):
            self.global_step += 1
            preds = self.model(batch["image_tensor"])
            loss = self.criterion(
                preds,
                batch["bbox_label"],
                batch["objectness_label"],
                batch["class_label"],
            )
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
            curr_lr = self.scheduler.get_last_lr()[0]  # only 1 optimizer param group
            if self.accelerator.step_scheduler_with_optimizer:
                self.scheduler.step()
            # metrics reporting
            loss = loss.item()
            pbar.set_postfix({"loss": loss})
            if self.global_step % 50 == 0:
                self.accelerator.log({"train/loss": loss}, step=self.global_step)
                self.accelerator.log({"train/lr": curr_lr}, step=self.global_step)
        if not self.accelerator.step_scheduler_with_optimizer:
            self.scheduler.step()

    @torch.no_grad()
    def val(self, val_loader: DataLoader) -> float:
        self.model.eval()
        for batch in (pbar := tqdm(val_loader, desc="val", leave=False)):
            preds = self.model(batch["image_tensor"])
            loss = self.criterion(
                preds,
                batch["bbox_label"],
                batch["objectness_label"],
                batch["class_label"],
            )
            # metrics reporting
            loss = loss.item()
            self.val_loss_metric(loss)
            pbar.set_postfix({"loss": loss})
        # metrics agg
        avg_loss = self.val_loss_metric.compute()
        self.accelerator.log({"val/loss": avg_loss}, step=self.global_step)
        self.val_loss_metric.reset()
        return avg_loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
    ):
        for epoch in range(1, n_epochs + 1):
            print(f"epoch {epoch}/{n_epochs}")
            self.train(train_loader)
            avg_val_loss = self.val(val_loader)
            if avg_val_loss < self.best_loss:
                print(f"Found new best loss: {avg_val_loss}")
                self.best_loss = avg_val_loss
                self.accelerator.save_state()
            self.accelerator.log({"train/epoch": epoch}, step=self.global_step)

    def overfit_one_batch(self, train_loader: DataLoader):
        print("🔥 overfitting for one batch 🔥")
        self.model.train()
        batch = next(iter(train_loader))
        (
            image_tensor,
            bbox_label,
            objectness_label,
            class_label,
        ) = (
            batch["image_tensor"],
            batch["bbox_label"],
            batch["objectness_label"],
            batch["class_label"],
        )
        print("image_tensor", image_tensor.size())
        print("bbox_label", bbox_label.size())
        print("objectness_label", objectness_label.size())
        print("class_label", class_label.size())
        for i in range(999999):
            preds = self.model(image_tensor)
            loss = self.criterion(
                preds,
                bbox_label,
                objectness_label,
                class_label,
            )
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
            print(i, loss.item())
