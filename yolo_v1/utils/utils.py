import torch
import numpy as np
import cv2 as cv
from PIL import Image
from collections import Counter
from typing import Tuple, Sequence

def iou(bboxes_pred: torch.Tensor, bboxes_gt: torch.Tensor) -> torch.Tensor:
    """
    Input: bboxes_pred, bboxes_gt.
    Output: iou.
    """
    pred_x1 = bboxes_pred[..., 0:1] - bboxes_pred[..., 2:3] / 2
    pred_y1 = bboxes_pred[..., 1:2] - bboxes_pred[..., 3:4] / 2
    pred_x2 = bboxes_pred[..., 0:1] + bboxes_pred[..., 2:3] / 2
    pred_y2 = bboxes_pred[..., 1:2] + bboxes_pred[..., 3:4] / 2

    gt_x1 = bboxes_gt[..., 0:1] - bboxes_gt[..., 2:3] / 2
    gt_y1 = bboxes_gt[..., 1:2] - bboxes_gt[..., 3:4] / 2
    gt_x2 = bboxes_gt[..., 0:1] + bboxes_gt[..., 2:3] / 2
    gt_y2 = bboxes_gt[..., 1:2] + bboxes_gt[..., 3:4] / 2

    inter_x1 = torch.max(pred_x1, gt_x1)
    inter_y1 = torch.max(pred_y1, gt_y1)
    inter_x2 = torch.min(pred_x2, gt_x2)
    inter_y2 = torch.min(pred_y2, gt_y2)

    inter_width = (inter_x2 - inter_x1).clamp(min=0)
    inter_height = (inter_y2 - inter_y1).clamp(min=0)
    intersection = inter_width * inter_height

    pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
    gt_area = (gt_x2 - gt_x1).clamp(min=0) * (gt_y2 - gt_y1).clamp(min=0)

    union = pred_area + gt_area - intersection + 1e-6

    return intersection / union

def transform_image(image: Image.Image, factor: float = 20.0) -> Tuple[Image.Image, np.ndarray]:
    """
    Input: image.
    Output: transformed image.
    """
    img_array = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    original_height, original_width = img_array.shape[:2]

    scale_min_x = int(original_width * (1 - factor / 100))
    scale_max_x = int(original_width)
    scaled_width = np.random.randint(low=scale_min_x, high=scale_max_x)

    scale_min_y = int(original_height * (1 - factor / 100))
    scale_max_y = int(original_height)
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

def transform_bboxes(bboxes: Sequence[Sequence[float]], transform_params: np.ndarray) -> np.ndarray:
    image_height, image_width = transform_params[0]
    translate_x, translate_y = transform_params[1]
    offset_x, offset_y = transform_params[2]
    scale_x, scale_y = transform_params[3]

    bboxes_array = np.asarray(bboxes, dtype=np.float32)
    transformed = bboxes_array.copy()

    coords = bboxes_array[:, 1:5]
    if coords.shape[1] != 4:
        raise ValueError(f"Expected 4 bbox values (x, y, w, h). Got {coords.shape[1]} bbox values.")

    x = (coords[:, 0] * scale_x) + (offset_x + translate_x) / image_width
    y = (coords[:, 1] * scale_y) + (offset_y + translate_y) / image_height
    w = coords[:, 2] * scale_x
    h = coords[:, 3] * scale_y

    transformed[:, 1] = np.clip(x, 0.0, 0.999)
    transformed[:, 2] = np.clip(y, 0.0, 0.999)
    transformed[:, 3] = np.clip(w, 0.0, 0.999)
    transformed[:, 4] = np.clip(h, 0.0, 0.999)

    return transformed

def mAP(predictions: Sequence[Sequence[float]], targets: Sequence[Sequence[float]], iou_threshold: float = 0.5, num_classes: int = 20) -> torch.Tensor:
    """
    Input: predictions, targets, iou_threshold, num_classes.
    Output: mAP.
    """
    average_precisions = []
    for c in range(num_classes):
        detections = [detection for detection in predictions if detection[1] == c]
        ground_truths = [true_bbox for true_bbox in targets if true_bbox[1] == c]

        amount_of_gts = Counter([gt[0] for gt in ground_truths])
        for key, val in amount_of_gts.items():
            amount_of_gts[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)

        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        for det_idx, detection in enumerate(detections):
            gt_for_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            best_iou = 0
            best_gt_idx = -1

            for idx, gt in enumerate(gt_for_img):
                current_iou = iou(
                    bboxes_pred=torch.unsqueeze(torch.tensor(detection[3:]), 0),
                    bboxes_gt=torch.unsqueeze(torch.tensor(gt[3:]), 0)
                ).item()

                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_of_gts[detection[0]][best_gt_idx] == 0:
                    TP[det_idx] = 1
                    amount_of_gts[detection[0]][best_gt_idx] = 1
                else:
                    FP[det_idx] = 1
            else:
                FP[det_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = torch.div(TP_cumsum, (total_true_bboxes + 1e-6))
        precisions = torch.div(TP_cumsum, (TP_cumsum + FP_cumsum + 1e-6))

        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def get_bboxes(loader: torch.utils.data.DataLoader, model: torch.nn.Module, iou_threshold: float, threshold: float, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Tuple[list[list[float]], list[list[float]]]:
    """
    Input: loader, model, iou_threshold, threshold, device.
    Output: all_predicted_boxes, all_true_boxes.
    """
    all_predicted_boxes = []
    all_true_boxes = []

    model.eval()
    image_idx_counter = 0 

    for batch_idx, (image_batch, label_batch) in enumerate(loader):
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)

        with torch.no_grad():
            model_predictions = model(image_batch)

        batch_size = image_batch.shape[0]

        true_bboxes_batch = cells_to_boxes(label_batch) 
        predicted_bboxes_batch = cells_to_boxes(model_predictions) 

        for i in range(batch_size):
            nms_processed_boxes = nms(predicted_bboxes_batch[i], iou_threshold=iou_threshold, threshold=threshold)

            for nms_box in nms_processed_boxes:
                all_predicted_boxes.append([image_idx_counter] + nms_box)

            for true_box in true_bboxes_batch[i]:
                if true_box[1] > threshold:
                    all_true_boxes.append([image_idx_counter] + true_box)

            image_idx_counter += 1
 
    return all_predicted_boxes, all_true_boxes

def nms(bboxes: Sequence[Sequence[float]], iou_threshold: float, threshold: float) -> Sequence[Sequence[float]]:
    """
    Input: bboxes, iou_threshold, threshold.
    Output: bboxes_after_nms.
    """
    if not isinstance(bboxes, list):
        raise TypeError("Input 'bboxes' must be a list.")

    filtered_bboxes = [box for box in bboxes if box[1] > threshold]
    filtered_bboxes.sort(key=lambda x: x[1], reverse=True)

    bboxes_after_nms = []

    while filtered_bboxes:
        chosen_box = filtered_bboxes.pop(0)
        bboxes_after_nms.append(chosen_box)

        filtered_bboxes = [
            box
            for box in filtered_bboxes
            if box[0] != chosen_box[0]  # Different class
            or iou(torch.tensor(chosen_box[2:]), torch.tensor(box[2:])).item() < iou_threshold
        ]

    return bboxes_after_nms

def convert_cells(predictions: torch.Tensor, grid_size: int = 7) -> torch.Tensor:
    """
    Input: predictions, grid_size.
    Output: final_predictions.
    """
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, grid_size, grid_size, -1)

    bbox1_coords = predictions[..., 21:25] 
    bbox2_coords = predictions[..., 26:30] 

    bbox1_confidence = predictions[..., 20].unsqueeze(-1) 
    bbox2_confidence = predictions[..., 25].unsqueeze(-1)

    stacked_confidences = torch.cat((bbox1_confidence, bbox2_confidence), dim=-1)
    best_bbox_selector = stacked_confidences.argmax(dim=-1, keepdim=True)

    best_bboxes_coords = bbox1_coords * (1 - best_bbox_selector) + bbox2_coords * best_bbox_selector
    best_bboxes_confidence = torch.max(bbox1_confidence, bbox2_confidence) 

    cell_indices_x = torch.arange(grid_size, device=predictions.device).repeat(batch_size, grid_size, 1).unsqueeze(-1)
    cell_indices_y = torch.arange(grid_size, device=predictions.device).repeat(batch_size, grid_size, 1).unsqueeze(-1).permute(0, 2, 1, 3)

    x_center = (1 / grid_size) * (best_bboxes_coords[..., :1] + cell_indices_x)
    y_center = (1 / grid_size) * (best_bboxes_coords[..., 1:2] + cell_indices_y)
    width_height = (1 / grid_size) * best_bboxes_coords[..., 2:4]

    converted_bboxes = torch.cat((x_center, y_center, width_height), dim=-1)

    predicted_class_labels = predictions[..., :20].argmax(dim=-1, keepdim=True)

    final_predictions = torch.cat(
        (predicted_class_labels, best_bboxes_confidence, converted_bboxes), dim=-1
    )

    return final_predictions


def cells_to_boxes(model_output: torch.Tensor, grid_size: int = 7) -> list[list[list[float]]]:
    """
    Input: model_output, grid_size.
    Output: all_image_bboxes.
    """
    converted_predictions = convert_cells(model_output, grid_size).reshape(model_output.shape[0], grid_size * grid_size, -1)
    converted_predictions[..., 0] = converted_predictions[..., 0].long()

    all_image_bboxes = []

    for img_idx in range(model_output.shape[0]):
        image_specific_bboxes = []
        for bbox_idx in range(grid_size * grid_size):
            image_specific_bboxes.append([val.item() for val in converted_predictions[img_idx, bbox_idx, :]])
        all_image_bboxes.append(image_specific_bboxes)

    return all_image_bboxes
