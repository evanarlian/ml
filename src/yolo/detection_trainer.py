import torch
from accelerate import Accelerator
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm.auto import tqdm
from utils import split_yolo_tensor


class DetectorTrainer:
    def __init__(
        self,
        accelerator: Accelerator,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        # metadata
        class2id: dict,
        id2class: dict,
        hparams: dict,
        # metrics
        train_map_metric: Metric,
        val_map_metric: Metric,
        val_loss_metric: Metric,
    ):
        self.accelerator = accelerator
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        # metadata
        self.class2id = class2id
        self.id2class = id2class
        self.hparams = hparams
        self.S = hparams["s"]
        self.B = hparams["b"]
        self.C = hparams["c"]
        # metrics related
        self.train_map_metric = train_map_metric
        self.val_map_metric = val_map_metric
        self.val_loss_metric = val_loss_metric
        self.global_step = 0
        self.best_loss = float("inf")

    # def calculate_map(self, batch: dict, preds: torch.Tensor) -> dict:
    #     # separate tensors
    #     xywhc_pred, class_pred = preds.split([self.B * 5, self.C], dim=-1)
    #     xywhc_pred = xywhc_pred.view(-1, self.S, self.S, self.B, 5)
    #     bbox_pred, objectness_pred = xywhc_pred.split([4, 1], dim=-1)
    #     objectness_pred = objectness_pred.squeeze(-1)

    #     # calculate map from torchmetrics
    #     preds_ = [
    #         {
    #             # The boxes keyword should contain an [N,4] tensor,
    #             # where N is the number of detected boxes with boxes of the format
    #             # [xmin, ymin, xmax, ymax] in absolute image coordinates
    #             "boxes": torch.tensor([[0.3, 0.3, 0.7, 0.7]]) * 100,
    #             # The scores keyword should contain an [N,] tensor where
    #             # each element is confidence score between 0 and 1
    #             "scores": torch.tensor([0.7]),
    #             # The labels keyword should contain an [N,] tensor
    #             # with integers of the predicted classes
    #             "labels": torch.tensor([0]),
    #             # The masks keyword should contain an [N,H,W] tensor,
    #             # where H and W are the image height and width, respectively,
    #             # with boolean masks. This is only required when iou_type is `segm`.
    #         }
    #     ]  # images per batch

    #     labels = [
    #         {
    #             "boxes": torch.tensor(
    #                 [
    #                     [0.3, 0.3, 0.7, 0.7],
    #                     [0.2, 0.2, 0.8, 0.8],
    #                 ]
    #             )
    #             * 100,
    #             "labels": torch.tensor([0, 0]),
    #         }
    #     ]

    #     metric(preds, labels)
    #     pass

    def single_step(self, batch: dict) -> torch.Tensor:
        # single step shared by train and val
        preds = self.model(batch["image_tensor"])
        bbox_pred, objectness_pred, class_pred = split_yolo_tensor(
            preds, self.S, self.B, self.C
        )
        loss = self.criterion(
            batch["bbox_label"],
            batch["objectness_label"],
            batch["class_label"],
            bbox_pred,
            objectness_pred,
            class_pred,
        )
        return loss

    def train(self, train_loader: DataLoader):
        self.model.train()
        for batch in (pbar := tqdm(train_loader, desc="train", leave=False)):
            self.global_step += 1
            loss = self.single_step(batch)
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
                # TODO do whatever neccesary in train or val, calculating map or log images
                self.accelerator.log({"train/loss": loss}, step=self.global_step)
                self.accelerator.log({"train/lr": curr_lr}, step=self.global_step)
        if not self.accelerator.step_scheduler_with_optimizer:
            self.scheduler.step()

    @torch.no_grad()
    def val(self, val_loader: DataLoader) -> float:
        self.model.eval()
        for batch in (pbar := tqdm(val_loader, desc="val", leave=False)):
            loss = self.single_step(batch)
            # metrics reporting
            loss = loss.item()
            self.val_loss_metric(loss)
            pbar.set_postfix({"loss": loss})
        # metrics agg
        avg_loss = self.val_loss_metric.compute()
        self.accelerator.log({"val/loss": avg_loss}, step=self.global_step)
        self.val_loss_metric.reset()
        return avg_loss

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, n_epochs: int):
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
        print("image_tensor", batch["image_tensor"].size())
        print("bbox_label", batch["bbox_label"].size())
        print("objectness_label", batch["objectness_label"].size())
        print("class_label", batch["class_label"].size())
        for i in range(999999):
            loss = self.single_step(batch)
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
            print(i, loss.item())
