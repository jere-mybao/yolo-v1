import torch
from yolo_v1.utils.utils import iou


def test_iou_basic_overlap():
    # Two overlapping boxes
    box1 = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    box2 = torch.tensor([[0.6, 0.6, 0.4, 0.4]])
    expected_iou = 0.39130434
    assert torch.isclose(iou(box1, box2), torch.tensor(expected_iou), atol=1e-7)


def test_iou_no_overlap():
    # Two non-overlapping boxes
    box1 = torch.tensor([[0.2, 0.2, 0.1, 0.1]])
    box2 = torch.tensor([[0.8, 0.8, 0.1, 0.1]])
    expected_iou = 0.0
    assert torch.isclose(iou(box1, box2), torch.tensor(expected_iou), atol=1e-7)


def test_iou_complete_overlap():
    # Two identical boxes
    box1 = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    box2 = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    expected_iou = 1.0
    assert torch.isclose(iou(box1, box2), torch.tensor(expected_iou), atol=1e-7)


def test_iou_one_box_inside_another():
    # One box completely inside another
    box1 = torch.tensor([[0.5, 0.5, 0.8, 0.8]])
    box2 = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    expected_iou = 0.25
    assert torch.isclose(iou(box1, box2), torch.tensor(expected_iou), atol=1e-7)


def test_iou_touching_boxes():
    # Boxes touching at an edge
    box1 = torch.tensor([[0.25, 0.5, 0.5, 0.5]])
    box2 = torch.tensor([[0.75, 0.5, 0.5, 0.5]])
    expected_iou = 0.0
    assert torch.isclose(iou(box1, box2), torch.tensor(expected_iou), atol=1e-7)


def test_iou_batch_input():
    # Batch input with multiple pairs
    boxes1 = torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.1, 0.1, 0.1, 0.1]])
    boxes2 = torch.tensor([[0.6, 0.6, 0.4, 0.4], [0.8, 0.8, 0.1, 0.1]])
    expected_iou = torch.tensor([[0.39130434], [0.0]])
    assert torch.allclose(iou(boxes1, boxes2), expected_iou, atol=1e-7)
