import torch
import torch.nn as nn
from yolo_v1.utils.utils import iou


class YoloV1Loss(nn.Module):
    def __init__(self, grid_size=7, num_bboxes=2, num_classes=20):
        super(YoloV1Loss, self).__init__()
        self.grid_size = grid_size
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes
        self.lambda_no_obj = 0.5
        self.lambda_obj = 5
        self.class_prob_offset = 0
        self.bbox1_conf_offset = self.num_classes
        self.bbox1_coords_offset = self.num_classes + 1
        self.bbox2_conf_offset = self.num_classes + 5
        self.bbox2_coords_offset = self.num_classes + 6
        self.gt_class_prob_offset = 0
        self.gt_obj_conf_offset = self.num_classes
        self.gt_bbox_coords_offset = self.num_classes + 1

    def forward(self, predictions, ground_truth):
        mse_loss = nn.MSELoss(reduction="sum")
        predictions = predictions.reshape(
            -1, self.grid_size, self.grid_size, self.num_classes + self.num_bboxes * 5
        )

        iou_bbox1 = iou(
            predictions[..., self.bbox1_coords_offset : self.bbox1_coords_offset + 4],
            ground_truth[
                ..., self.gt_bbox_coords_offset : self.gt_bbox_coords_offset + 4
            ],
        )
        iou_bbox2 = iou(
            predictions[..., self.bbox2_coords_offset : self.bbox2_coords_offset + 4],
            ground_truth[
                ..., self.gt_bbox_coords_offset : self.gt_bbox_coords_offset + 4
            ],
        )
        ious = torch.cat([iou_bbox1.unsqueeze(0), iou_bbox2.unsqueeze(0)], dim=0)
        _, bestbox = torch.max(ious, dim=0)

        identity_obj_i = ground_truth[..., self.gt_obj_conf_offset].unsqueeze(3)

        box_predictions = identity_obj_i * (
            bestbox
            * predictions[..., self.bbox2_coords_offset : self.bbox2_coords_offset + 4]
            + (1 - bestbox)
            * predictions[..., self.bbox1_coords_offset : self.bbox1_coords_offset + 4]
        )
        box_targets = (
            identity_obj_i
            * ground_truth[
                ..., self.gt_bbox_coords_offset : self.gt_bbox_coords_offset + 4
            ]
        )

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = mse_loss(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        pred_box_conf = (
            bestbox
            * predictions[..., self.bbox2_conf_offset : self.bbox2_conf_offset + 1]
            + (1 - bestbox)
            * predictions[..., self.bbox1_conf_offset : self.bbox1_conf_offset + 1]
        )
        object_loss = mse_loss(
            torch.flatten(identity_obj_i * pred_box_conf),
            torch.flatten(
                identity_obj_i
                * ground_truth[
                    ..., self.gt_obj_conf_offset : self.gt_obj_conf_offset + 1
                ]
            ),
        )

        no_object_loss = mse_loss(
            torch.flatten(
                (1 - identity_obj_i)
                * predictions[..., self.bbox1_conf_offset : self.bbox1_conf_offset + 1],
                start_dim=1,
            ),
            torch.flatten(
                (1 - identity_obj_i)
                * ground_truth[
                    ..., self.gt_obj_conf_offset : self.gt_obj_conf_offset + 1
                ],
                start_dim=1,
            ),
        )
        no_object_loss += mse_loss(
            torch.flatten(
                (1 - identity_obj_i)
                * predictions[..., self.bbox2_conf_offset : self.bbox2_conf_offset + 1],
                start_dim=1,
            ),
            torch.flatten(
                (1 - identity_obj_i)
                * ground_truth[
                    ..., self.gt_obj_conf_offset : self.gt_obj_conf_offset + 1
                ],
                start_dim=1,
            ),
        )

        class_loss = mse_loss(
            torch.flatten(
                identity_obj_i * predictions[..., : self.num_classes], end_dim=-2
            ),
            torch.flatten(
                identity_obj_i * ground_truth[..., : self.num_classes], end_dim=-2
            ),
        )

        total_loss = (
            self.lambda_obj * box_loss
            + object_loss
            + self.lambda_no_obj * no_object_loss
            + class_loss
        )
        return total_loss
