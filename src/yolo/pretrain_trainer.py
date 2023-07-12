import torch
from accelerate import Accelerator
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm.auto import tqdm


class BackboneTrainer:
    def __init__(
        self,
        accelerator: Accelerator,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        train_metrics: Metric,
        val_metrics: Metric,
        val_loss_metric: Metric,
    ):
        self.accelerator = accelerator
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        # metrics related
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.val_loss_metric = val_loss_metric
        self.global_step = 0
        self.best_loss = float("inf")

    def train(self, train_loader: DataLoader):
        self.model.train()
        for batch in (pbar := tqdm(train_loader, desc="train", leave=False)):
            self.global_step += 1
            image, label = batch["image_tensor"], batch["class_id"]
            logits = self.model(image)
            loss = self.criterion(logits, label)
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
            curr_lr = self.scheduler.get_last_lr()[0]  # only 1 optimizer param group
            if self.accelerator.step_scheduler_with_optimizer:
                self.scheduler.step()
            # metrics reporting
            loss = loss.item()
            self.train_metrics(logits, label)
            pbar.set_postfix({"loss": loss})
            self.train
            if self.global_step % 50 == 0:
                self.accelerator.log({"train/loss": loss}, step=self.global_step)
                self.accelerator.log({"train/lr": curr_lr}, step=self.global_step)
        if not self.accelerator.step_scheduler_with_optimizer:
            self.scheduler.step()
        # metrics agg
        self.accelerator.log(self.train_metrics.compute(), step=self.global_step)
        self.train_metrics.reset()

    @torch.no_grad()
    def val(self, val_loader: DataLoader) -> float:
        self.model.eval()
        for batch in (pbar := tqdm(val_loader, desc="val", leave=False)):
            image, label = batch["image_tensor"], batch["class_id"]
            logits = self.model(image)
            loss = self.criterion(logits, label).item()
            # metrics reporting
            self.val_loss_metric(loss)
            self.val_metrics(logits, label)
            pbar.set_postfix({"loss": loss})
        # metrics agg
        avg_loss = self.val_loss_metric.compute()
        self.accelerator.log({"val/loss": avg_loss}, step=self.global_step)
        self.accelerator.log(self.val_metrics.compute(), step=self.global_step)
        self.val_loss_metric.reset()
        self.val_metrics.reset()
        return avg_loss.item()

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
        image, label = batch["image_tensor"], batch["class_id"]
        print(image.size())
        print(label)
        for i in range(999999):
            logits = self.model(image)
            loss = self.criterion(logits, label)
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
            print(i, loss.item())

    # TODO lr finder make hehe
