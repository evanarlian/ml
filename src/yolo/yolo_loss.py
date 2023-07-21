from torch import Tensor, nn
from utils import convert_yolo_to_pascalvoc, iou


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
        preds: Tensor,
        bbox_label: Tensor,
        objectness_label: Tensor,
        class_label: Tensor,
    ):
        # adjust bbox and objectness label tensors so that they have B = 1
        bbox_label = bbox_label.unsqueeze(-2)
        objectness_label = objectness_label.unsqueeze(-1)
        # print(bbox_label.size())
        # print(objectness_label.size())
        # print(class_label.size())
        # print()

        # "match" raw preds dimensions to those 3 labels
        xywhc_pred, class_pred = preds.split([self.B * 5, self.C], dim=-1)
        xywhc_pred = xywhc_pred.view(-1, self.S, self.S, self.B, 5)
        bbox_pred, objectness_pred = xywhc_pred.split([4, 1], dim=-1)
        objectness_pred = objectness_pred.squeeze(-1)
        # print(bbox_pred.size())
        # print(objectness_pred.size())
        # print(class_pred.size())
        # print()

        # at this point:
        # * both bbox: (bs, S, S, B|1, 4)
        # * both objectness: (bs, S, S, B|1)
        # * both class: (bs, S, S, C)

        # calculate iou from bbox pred vs bbox label
        # iou must be in pascalvoc format
        # we also need max_ious boolean mask based on the max iou
        # ious: (bs, S, S, B)
        # max_ious: (bs, S, S, B)
        ious = iou(
            convert_yolo_to_pascalvoc(bbox_label),
            convert_yolo_to_pascalvoc(bbox_pred),
        )
        max_ious = ious == ious.max(-1).values.unsqueeze(-1)
        # print(ious.size())
        # print(max_ious.size())
        # print()

        # losses
        # from paper, 1_obj is similar to "who is responsible"
        # * in bbox-level data, responsible means there is object and max iou
        # * in grid-level data, responsible means there is object
        batch_size = preds.size(0)
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
