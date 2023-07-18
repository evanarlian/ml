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
        self.mse = nn.MSELoss(reduction="sum")  # summation symbol from paper

    def forward(
        self,
        preds: Tensor,
        bbox_label: Tensor,
        objectness_label: Tensor,
        class_label: Tensor,
    ):
        # split preds tensor to bboxes and clases
        # preds: (bs, S, S, (B*5+C)) will become:
        # * bbox_preds: (bs, S, S, (B*5))
        # * class_preds: (bs, S, S, C)
        bbox_preds, class_preds = preds.split([self.B * 5, self.C], dim=-1)

        # (B*5) part in bbox_preds makes doing tensor operations painful
        # just make B to a standalone dimension
        # bbox_preds: (bs, S, S, (B*5)) -> (bs, S, S, B, 5)
        bbox_preds = bbox_preds.view(-1, self.S, self.S, self.B, 5)

        # preds recap:
        # * bbox_preds: (bs, S, S, B, 5)
        # * class_preds: (bs, S, S, C)
        # label recap:
        # * bbox_label: (bs, S, S, 4)
        # * objectness_label: (bs, S, S)
        # * class_label: (bs, S, S, C)

        # calculate iou from bbox pred vs bbox label
        #
        # TODO TODO unsqueeze shuld be done manually here, not in the iou
        # xy_loss = objectness_label *


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
    isect_xmin = torch.max(label_xmin, pred_xmin)
    isect_ymin = torch.max(label_ymin, pred_ymin)
    isect_xmax = torch.min(label_xmax, pred_xmax)
    isect_ymax = torch.min(label_ymax, pred_ymax)

    # now we have 4 different areas: label, pred, intersection, union
    # intersection cannot be negative to we clip
    label_area = (label_ymax - label_ymin) * (label_xmax - label_xmin)
    pred_area = (pred_ymax - pred_ymin) * (pred_xmax - pred_xmin)
    isect_area = (isect_ymax - isect_ymin).clamp(0) * (isect_xmax - isect_xmin).clamp(0)
    union_area = label_area + pred_area - isect_area

    # iou: (bs, S, S, B)
    return isect_area / union_area


def main():
    # sanity checks
    # TODO this is better separeted to pytest
    t = torch.tensor
    ac = torch.allclose
    assert ac(iou(t([0, 0, 10, 10]), t([0, 0, 10, 10])), t([1.0]))


if __name__ == "__main__":
    main()
