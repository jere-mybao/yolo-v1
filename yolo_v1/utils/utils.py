import torch

def iou(bboxes_preds, bboxes_gts):    
    """
    Calculates intersection of unions.
    Input: Bounding box predictions (N , 4)
            Bounding box ground truths (N, 4).
    Output: Intersection over union.
    """
    box1_x1 = bboxes_preds[...,0:1] - bboxes_preds[...,2:3] / 2
    box1_y1 = bboxes_preds[...,1:2] - bboxes_preds[...,3:4] / 2
    box1_x2 = bboxes_preds[...,0:1] + bboxes_preds[...,2:3] / 2
    box1_y2 = bboxes_preds[...,1:2] + bboxes_preds[...,3:4] / 2
    box2_x1 = bboxes_gts[...,0:1] - bboxes_gts[...,2:3] / 2
    box2_y1 = bboxes_gts[...,1:2] - bboxes_gts[...,3:4] / 2
    box2_x2 = bboxes_gts[...,0:1] +  bboxes_gts[...,2:3] / 2
    box2_y2 = bboxes_gts[...,1:2] +  bboxes_gts[...,3:4] / 2
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    intersec = torch.clip((x2 - x1), min = 0) * torch.clip((y2 - y1), min = 0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union = box1_area + box2_area - intersec + 1e-6
    iou = intersec / union
    return iou