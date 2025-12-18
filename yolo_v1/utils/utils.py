import torch
import numpy as np
import cv2 as cv
from PIL import Image

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

def transform_image(image, factor=20):
    """
    Input: image.
    Output: transformed image.
    """
    img_array = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    original_height, original_width = img_array.shape[:2]

    scale_min_x = original_width * (1 - factor / 100)
    scale_max_x = original_width
    scaled_width = np.random.randint(low=scale_min_x, high=scale_max_x)

    scale_min_y = original_height * (1 - factor / 100)
    scale_max_y = original_height
    scaled_height = np.random.randint(low=scale_min_y, high=scale_max_y)

    scale_ratio_x = scaled_width / original_width
    scale_ratio_y = scaled_height / original_height

    translate_limit_x = original_width * factor / 100
    translate_limit_y = original_height * factor / 100

    translate_x = np.random.uniform(low=-translate_limit_x, high=translate_limit_x)
    translate_y = np.random.uniform(low=-translate_limit_y, high=translate_limit_y)

    translation_matrix = np.float32([[1, 0, translate_x], [0, 1, translate_y]])

    scaled_image_content = cv.resize(img_array, (scaled_width, scaled_height), interpolation=cv.INTER_CUBIC)

    padded_image = np.zeros(shape=[original_height, original_width, 3], dtype=np.uint8)
    offset_y = round((original_height - scaled_height) / 2)
    offset_x = round((original_width - scaled_width) / 2)
    padded_image[offset_y:offset_y + scaled_height, offset_x:offset_x + scaled_width] = scaled_image_content

    transformed_image_cv = cv.warpAffine(padded_image, translation_matrix, (original_width, original_height))
    transformed_image_pil = Image.fromarray(cv.cvtColor(transformed_image_cv, cv.COLOR_BGR2RGB))

    transform_params = np.array([[original_height, original_width],
                                 [translate_x, translate_y],
                                 [offset_x, offset_y],
                                 [scale_ratio_x, scale_ratio_y]])

    return transformed_image_pil, transform_params

def transform_bboxes(bboxes, transform_params):
    """
    Input: bboxes (x, y ,w, h).
    Output: transformed bboxes (x, y ,w, h).
    """
    transformed_bboxes = bboxes.copy()
    original_height, original_width = transform_params[0]
    translate_x, translate_y = transform_params[1]
    offset_x, offset_y = transform_params[2]
    scale_ratio_x, scale_ratio_y = transform_params[3]

    bbox_coords = transformed_bboxes[:, 1:]

    bbox_coords[:, 0] = np.clip(((bbox_coords[:, 0] * scale_ratio_x) + (offset_x / original_width) + (translate_x / original_width)), 0, 0.999)
    bbox_coords[:, 1] = np.clip(((bbox_coords[:, 1] * scale_ratio_y) + (offset_y / original_height) + (translate_y / original_height)), 0, 0.999)
    bbox_coords[:, 2] = np.clip((bbox_coords[:, 2] * scale_ratio_x), 0, 0.999)
    bbox_coords[:, 3] = np.clip((bbox_coords[:, 3] * scale_ratio_y), 0, 0.999)

    transformed_bboxes[:, 1:] = bbox_coords
        
    return transformed_bboxes