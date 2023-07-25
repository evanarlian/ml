import torch
from torch import Tensor, nn
from utils import get_best_iou


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction="none")  # we'll sum manually during forward

    def forward(
        self,
        bbox_label: Tensor,
        objectness_label: Tensor,
        class_label: Tensor,
        bbox_pred: Tensor,
        objectness_pred: Tensor,
        class_pred: Tensor,
    ):
        # at this point:
        # * both bbox: (bs, S, S, B|1, 4)
        # * both objectness: (bs, S, S, B|1)
        # * both class: (bs, S, S, C)

        # max_ious: (bs, S, S, B)
        max_ious = get_best_iou(bbox_label, bbox_pred)

        # losses
        # from paper, 1_obj is similar to "who is responsible"
        # * in bbox-level data, responsible means there is object and max iou
        # * in grid-level data, responsible means there is object
        batch_size = bbox_pred.size(0)
        xy_loss = (
            (
                self.lambda_coord
                * max_ious
                * objectness_label
                * (
                    self.mse(bbox_label[..., 0], bbox_pred[..., 0])
                    + self.mse(bbox_label[..., 1], bbox_pred[..., 1])
                )
            )
            .view(batch_size, -1)
            .sum(-1)
            .mean()
        )
        wh_loss = (
            (
                self.lambda_coord
                * max_ious
                * objectness_label
                * (
                    self.mse(
                        bbox_label[..., 2].sqrt(),
                        bbox_pred[..., 2].sign() * bbox_pred[..., 2].abs().sqrt(),
                    )
                    + self.mse(
                        bbox_label[..., 3].sqrt(),
                        bbox_pred[..., 3].sign() * bbox_pred[..., 3].abs().sqrt(),
                    )
                )
            )
            .view(batch_size, -1)
            .sum(-1)
            .mean()
        )
        obj_confidence_loss = (
            (max_ious * objectness_label * self.mse(objectness_label, objectness_pred))
            .view(batch_size, -1)
            .sum(-1)
            .mean()
        )
        noobj_confidence_loss = (
            (
                self.lambda_noobj
                * (1 - (max_ious * objectness_label))
                * self.mse(objectness_label, objectness_pred)
            )
            .view(batch_size, -1)
            .sum(-1)
            .mean()
        )
        class_loss = (
            (objectness_label * self.mse(class_label, class_pred))
            .view(batch_size, -1)
            .sum(-1)
            .mean()
        )
        yolo_loss = (
            xy_loss + wh_loss + obj_confidence_loss + noobj_confidence_loss + class_loss
        )
        return yolo_loss


def main():
    # test yolo loss shapes
    from utils import split_yolo_tensor

    # fake labels
    bs, S, B, C = 5, 7, 2, 20
    yolo_loss = YoloLoss(S, B, C)
    bbox_label = torch.rand(bs, S, S, 1, 4)
    objectness_label = torch.rand(bs, S, S, 1)
    class_label = torch.rand(bs, S, S, C)

    # fake preds
    preds = torch.randn(bs, S, S, (B * 5 + C))
    bbox_pred, objectness_pred, class_pred = split_yolo_tensor(preds, S, B, C)

    # calc loss
    loss = yolo_loss(
        bbox_label,
        objectness_label,
        class_label,
        bbox_pred,
        objectness_pred,
        class_pred,
    )
    print(loss)


if __name__ == "__main__":
    main()
