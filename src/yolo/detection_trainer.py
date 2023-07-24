import torch
from accelerate import Accelerator
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm.auto import tqdm
from utils import get_best_iou, split_yolo_tensor


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

    def calculate_map(self, batch: dict, preds: Tensor, metric: Metric) -> dict:
        # NOTE map per class is broken in torchmetrics since it does not accept
        # n_classes in the beginning and only looking at unique labels so far
        # also i don't care about map on small, medium, large detections

        # separate label tensor
        (
            bbox_label,
            objectness_label,
            class_label,
        ) = (
            batch["bbox_label"].cpu(),
            batch["objectness_label"].cpu().bool(),
            batch["class_label"].cpu(),
        )

        # separate pred tensor
        bbox_pred, objectness_pred, class_pred = split_yolo_tensor(
            preds.cpu(), self.S, self.B, self.C
        )

        # get best iou mask
        best_iou_mask = get_best_iou(bbox_label, bbox_pred)
        resp_mask = best_iou_mask * objectness_label

        # manually calculate per sample in batch
        pred_list = []
        target_list = []
        for i in range(bbox_label.size(0)):
            # mask preds
            bbox_pred_masked = bbox_pred[i][resp_mask[i]]
            objectness_pred_masked = objectness_pred[i][resp_mask[i]]
            class_pred_masked = class_pred[i][objectness_label[i].squeeze(-1)].argmax(-1)  # fmt: skip
            # masks labels
            bbox_label_masked = bbox_label[i][objectness_label[i]]
            class_label_masked = class_label[i][objectness_label[i].squeeze(-1)].argmax(-1)  # fmt: skip
            pred_list.append(
                {
                    "boxes": bbox_pred_masked,
                    "scores": objectness_pred_masked,
                    "labels": class_pred_masked,
                }
            )
            target_list.append(
                {
                    "boxes": bbox_label_masked,
                    "labels": class_label_masked,
                }
            )
        # raise RuntimeError("ASd")
        try:
            retval = metric(pred_list, target_list)
        except Exception as e:
            torch.save(
                [
                    bbox_label,
                    objectness_label,
                    class_label,
                    bbox_pred,
                    objectness_pred,
                    class_pred,
                    best_iou_mask,
                    resp_mask,
                    pred_list,
                    target_list,
                ],
                "asd.pickle",
            )
            raise e
        return retval

    def single_step(self, batch: dict) -> tuple[Tensor, Tensor]:
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
        return loss, preds

    def train(self, train_loader: DataLoader):
        self.model.train()
        for batch in (pbar := tqdm(train_loader, desc="train", leave=False)):
            self.global_step += 1
            loss, preds = self.single_step(batch)
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
                self.accelerator.log(
                    self.calculate_map(batch, preds, self.train_map_metric),
                    step=self.global_step,
                )
                # TODO log images?
        if not self.accelerator.step_scheduler_with_optimizer:
            self.scheduler.step()
        self.train_map_metric.reset()

    @torch.no_grad()
    def val(self, val_loader: DataLoader) -> float:
        self.model.eval()
        for batch in (pbar := tqdm(val_loader, desc="val", leave=False)):
            loss, preds = self.single_step(batch)
            # metrics reporting
            loss = loss.item()
            self.val_loss_metric(loss)
            self.calculate_map(batch, preds, self.val_map_metric)
            pbar.set_postfix({"loss": loss})
        # metrics agg
        avg_loss = self.val_loss_metric.compute()
        self.accelerator.log({"val/loss": avg_loss}, step=self.global_step)
        self.val_loss_metric.reset()
        self.accelerator.log(self.val_map_metric.compute(), step=self.global_step)
        self.val_map_metric.reset()
        # TODO log images
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
            if i == 100:
                print("UNFREEZE!!!!!!!!!!!")
                self.model.backbone.requires_grad_(True)
            loss, preds = self.single_step(batch)
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
            print(i, loss.item())
