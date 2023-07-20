import torch
from torch import Tensor, nn


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


def convert_yolo_to_pascalvoc(bboxes: Tensor) -> Tensor:
    """
    Converts yolo format to unnormalized pascalvoc format.
    Yolo format is (x_center, y_center, w_box, h_box).
    Pascalvoc format is (x_min, y_min, x_max, y_max).

    Args:
        bboxes (Tensor): Tensor with shape [..., 4]

    Returns:
        Tensor: Tensor with same shape as input
    """
    bbox_x, bbox_y, bbox_w, bbox_h = bboxes.unbind(-1)
    bbox_xmin = bbox_x - bbox_w / 2
    bbox_ymin = bbox_y - bbox_h / 2
    bbox_xmax = bbox_x + bbox_w / 2
    bbox_ymax = bbox_y + bbox_h / 2
    return torch.stack((bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax), dim=-1)


def iou(labels: Tensor, preds: Tensor) -> Tensor:
    """
    Calculate iou between labels bboxes and preds bboxes.
    Supports multidimensional inputs as long as they are broadcastable
    and the last dim length is 4.

    Args:
        labels (Tensor): Bbox labels with shape [..., 4]
        preds (Tensor):  Bbox predictions with shape [..., 4]

    Returns:
        Tensor: iou with broadcasted shape of labels and preds, and reduced on last dim
    """
    assert labels.size(-1) == preds.size(-1) == 4
    # labels: (bs, S, S, 1, 4)
    # preds: (bs, S, S, B, 4)
    # each unbound label shape: (bs, S, S, 1)
    # each unbound pred shape: (bs, S, S, B)
    label_xmin, label_ymin, label_xmax, label_ymax = labels.unbind(-1)
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = preds.unbind(-1)

    # calculate the intersection coordinates
    # below shapes: (bs, S, S, B)
    isect_xmin = torch.maximum(label_xmin, pred_xmin)
    isect_ymin = torch.maximum(label_ymin, pred_ymin)
    isect_xmax = torch.minimum(label_xmax, pred_xmax)
    isect_ymax = torch.minimum(label_ymax, pred_ymax)

    # now we have 4 different areas: label, pred, intersection, union
    # intersection cannot be negative to we clip
    label_area = (label_ymax - label_ymin) * (label_xmax - label_xmin)
    pred_area = (pred_ymax - pred_ymin) * (pred_xmax - pred_xmin)
    isect_area = (isect_ymax - isect_ymin).clamp(0) * (isect_xmax - isect_xmin).clamp(0)
    union_area = label_area + pred_area - isect_area

    # iou: (bs, S, S, B)
    return isect_area / union_area


def main():
    # manual testing
    t = torch.tensor
    assert torch.allclose(iou(t([0, 0, 10, 10]), t([0, 0, 10, 10])), t([1.0]))
    assert torch.allclose(iou(t([0, 0, 10, 10]), t([1, 1, 9, 9])), t([0.64]))
    assert torch.allclose(iou(t([0, 0, 10, 10]), t([11, 11, 14, 14])), t([0.0]))
    assert torch.allclose(iou(t([0, 0, 10, 10]), t([-5, 5, 5, 15])), t([1 / 7]))
    print("Passed iou manual testing")

    # fuzz testing vs torchvision implementation
    from torchvision.ops import box_iou

    for i in range(5000):
        point = torch.randn(1, 2)
        label = torch.cat([point, point + torch.rand(2)], dim=-1)
        pred = torch.cat([point + torch.rand(2) / 2, point + torch.rand(2)], dim=-1)
        my_iou = iou(pred, label)
        torchvision_iou = box_iou(pred, label)
        assert torch.allclose(my_iou, torchvision_iou)
    print("Passed iou fuzz testing")


if __name__ == "__main__":
    main()
