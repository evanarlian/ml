import torch
from torch import Tensor


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
    The bboc format is pascalvoc (xyxy)

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


def nms():
    pass


def average_precision():
    pass


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
