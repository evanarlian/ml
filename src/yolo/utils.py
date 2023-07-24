import random
from collections import OrderedDict

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
    The bbox format is pascalvoc (xyxy).

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


def nms(bboxes: Tensor, confidences: Tensor, iou_thresh: float) -> Tensor:
    """
    Perform non max supression and will return the selected indices.
    The bboxes must belong to the same class label (if object detection).
    The bbox format is pascalvoc (xyxy).

    Args:
        bboxes (Tensor): Bbox with shape [N, 4]
        confidences (Tensor): Bbox with shape [N,]
        iou_thresh (float): Remove bbox with iou >= iou_thresh

    Returns:
        Tensor: The kept indices after nms
    """
    assert bboxes.size(0) == confidences.size(0)
    assert bboxes.size(1) == 4
    ious = iou(bboxes, bboxes.unsqueeze(1)).tolist()
    sorted_indices = confidences.argsort(descending=True).tolist()
    kept_indices = OrderedDict.fromkeys(sorted_indices)  # we use this for ordered set
    for i in sorted_indices:
        if i not in kept_indices:
            continue
        for j in sorted_indices:
            if j not in kept_indices or i == j:
                continue
            if ious[i][j] >= iou_thresh:
                kept_indices.pop(j)
    return torch.tensor(list(kept_indices.keys()))


def split_yolo_tensor(
    preds: Tensor, S: int, B: int, C: int
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Splits yolo prediction tensor (bs, S, S, (B*5+C)) to 3 tensors:
    * Bbox tensor (bs, S, S, B, 4)
    * Objectness tensor (bs, S, S, B)
    * Class tensor (bs, S, S, C)

    Args:
        preds (Tensor): Raw yolo predictions
        S (int): S by S grid
        B (int): Bbox per grid
        C (int): Num classes

    Returns:
        tuple[Tensor, Tensor, Tensor]: Bbox, objectness, class tensors
    """
    xywhc_pred, class_pred = preds.split([B * 5, C], dim=-1)
    xywhc_pred = xywhc_pred.view(-1, S, S, B, 5)
    bbox_pred, objectness_pred = xywhc_pred.split([4, 1], dim=-1)
    objectness_pred = objectness_pred.squeeze(-1)
    return bbox_pred, objectness_pred, class_pred


def get_best_iou(bbox_label: Tensor, bbox_pred: Tensor) -> Tensor:
    """
    Selects the bbox with max iou for every grid. This is because the network
    produces many bboxes per grid, but at the end the best must be seleted.
    The input must be in yolo format (cxcywh), because the iou calculation
    will use pascalvoc format automatically.

    Args:
        bbox_label (Tensor): Bbox tensor with shape [..., 1, 4]
        bbox_pred (Tensor): Bbox tensor with shape [..., B, 4]

    Returns:
        Tensor: Boolean mask tensor of the max iou with shape [..., B]
    """
    # ious: (bs, S, S, B)
    # best_iou_mask: (bs, S, S, B)
    B = bbox_pred.size(-2)
    ious = iou(
        convert_yolo_to_pascalvoc(bbox_label),
        convert_yolo_to_pascalvoc(bbox_pred),
    )
    # we must use argmax and eye do tiebreaking (commonly found on all B ious are 0)
    best_iou_mask = ious == ious.max(-1).values.unsqueeze(-1)
    best_iou_mask = best_iou_mask.int().argmax(-1)
    best_iou_mask = torch.eye(B, device=best_iou_mask.device)[best_iou_mask].bool()
    return best_iou_mask


def main():
    from torchvision.ops import box_iou
    from torchvision.ops import nms as torch_nms

    def generate_fake_bboxes(n: int) -> tuple[Tensor, Tensor, Tensor]:
        center = torch.randn(n, 2)
        labels = torch.cat(
            [center - torch.rand(n, 2), center + torch.rand(n, 2)], dim=-1
        )
        preds = torch.cat(
            [center - torch.rand(n, 2), center + torch.rand(n, 2)], dim=-1
        )
        confidences = torch.rand(n)
        return labels, preds, confidences

    t = torch.tensor

    # iou manual testing
    assert torch.allclose(iou(t([0, 0, 10, 10]), t([0, 0, 10, 10])), t([1.0]))
    assert torch.allclose(iou(t([0, 0, 10, 10]), t([1, 1, 9, 9])), t([0.64]))
    assert torch.allclose(iou(t([0, 0, 10, 10]), t([11, 11, 14, 14])), t([0.0]))
    assert torch.allclose(iou(t([0, 0, 10, 10]), t([-5, 5, 5, 15])), t([1 / 7]))
    print("Passed iou manual testing")

    # iou fuzz testing vs torchvision implementation
    for i in range(100):
        labels, preds, _ = generate_fake_bboxes(1000)
        my_iou = iou(preds.unsqueeze(1), labels)  # because torchvision is pairwise
        torchvision_iou = box_iou(preds, labels)
        assert torch.allclose(my_iou, torchvision_iou)
    print("Passed iou fuzz testing")

    # nms manual testing
    assert torch.all(
        nms(
            t([[0, 0, 10, 10], [0, 0, 9.5, 9.5], [0, 0, 4, 4]]),
            t([0.7, 0.9, 0.3]),
            iou_thresh=0.5,
        )
        == t([1, 2])
    )
    print("Passed nms manual testing")

    # nms fuzz testing vs torchvision implementation
    for i in range(100):
        _, preds, confidences = generate_fake_bboxes(n=1000)
        # prevent birthday attack that can mess with argsort
        if confidences.unique().size(0) != 1000:
            continue
        thresh = random.random()
        my_kept = nms(preds, confidences, iou_thresh=thresh)
        torchvision_kept = torch_nms(preds, confidences, iou_threshold=thresh)
        assert torch.all(my_kept == torchvision_kept)
    print("Passed nms fuzz testing")

    # yolo tensor split test
    bs, S, B, C = 5, 7, 2, 20
    preds = torch.randn(bs, S, S, B * 5 + C)
    bbox, objectness, classes = split_yolo_tensor(preds, S, B, C)
    assert bbox.size() == torch.Size([bs, S, S, B, 4])
    assert objectness.size() == torch.Size([bs, S, S, B])
    assert classes.size() == torch.Size([bs, S, S, C])
    print("Passed yolo tensor split test")

    # bbox responsibility test (test in cxcywh format)
    bbox_label = t([[0.0, 0.0, 1.0, 1.0]])
    bbox_pred = t([[0.0, 0.0, 1.1, 1.1], [0.0, 0.0, 1.0, 1.0]])
    best_iou_mask = get_best_iou(bbox_label, bbox_pred)
    assert torch.all(best_iou_mask == t([False, True]))
    print("Passed max iou mask test")


if __name__ == "__main__":
    main()
