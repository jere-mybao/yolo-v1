from typing import Any
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from yolo_v1.dataset.dataset import Dataset
from yolo_v1.loss.loss import YoloV1Loss
from yolo_v1.models.model import YoloV1_Resnet50
from yolo_v1.utils.utils import mAP, get_bboxes
import wandb
import yaml

with open("yolo_v1/config/yolo_trainer.yaml", "r") as f:
    config = yaml.safe_load(f)

if config["wandb"]["enabled"]:
    wandb.init(project=config["wandb"]["project"], name=config["wandb"]["name"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_count = torch.cuda.device_count()
print(f"Using {gpu_count} GPUs")
if gpu_count > 1:
    batch_size = config["training"]["batch_size"] * gpu_count
    nworkers = config["training"]["nworkers"] * gpu_count
else:
    batch_size = config["training"]["batch_size"]
    nworkers = config["training"]["nworkers"]

weight_decay = config["optimizer"]["weight_decay"]
epochs = config["training"]["epochs"]
lr_sched_original = config["optimizer"]["lr_scheduler"]["original"]
lr_sched_conservative = config["optimizer"]["lr_scheduler"]["conservative"]


def train(model, train_loader, optimizer, loss_fn):
    accumulation_steps = config["training"]["accumulation_steps"]
    total_loss = 0
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        with torch.set_grad_enabled(True):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            total_loss += loss.item()
            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(
                train_loader
            ) - 1:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    return float(total_loss / len(train_loader))


def test(model, test_loader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            total_loss += loss.item()
    return float(total_loss / len(test_loader))


def main():
    train_dataset = Dataset(
        csv_file=config["data"]["csv_file_train"],
        img_dir=config["data"]["img_dir"],
        label_dir=config["data"]["label_dir"],
        grid_size=config["model"]["grid_size"],
        num_bboxes=config["model"]["num_bboxes"],
        num_classes=config["model"]["num_classes"],
        additional_transform=config["data"]["additional_transform_train"],
    )
    test_dataset = Dataset(
        csv_file=config["data"]["csv_file_test"],
        img_dir=config["data"]["img_dir"],
        label_dir=config["data"]["label_dir"],
        grid_size=config["model"]["grid_size"],
        num_bboxes=config["model"]["num_bboxes"],
        num_classes=config["model"]["num_classes"],
        additional_transform=config["data"]["additional_transform_test"],
    )

    train_loader = DataLoader[Any](
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=nworkers
    )
    test_loader = DataLoader[Any](
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=nworkers
    )

    model = YoloV1_Resnet50(
        grid_size=config["model"]["grid_size"],
        num_bboxes=config["model"]["num_bboxes"],
        num_classes=config["model"]["num_classes"],
    ).to(device)
    if gpu_count > 1:
        model = torch.nn.DataParallel(model)

    optimizer = Adam(
        model.parameters(),
        lr=config["optimizer"]["learning_rate"],
        weight_decay=weight_decay,
    )
    loss_fn = YoloV1Loss(
        grid_size=config["model"]["grid_size"],
        num_bboxes=config["model"]["num_bboxes"],
        num_classes=config["model"]["num_classes"],
    )

    train_loss = []
    train_mAP = []
    test_loss = []
    test_mAP = []

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        if lr_sched_original:
            for g in optimizer.param_groups:
                if epoch > 0 and epoch <= 5:
                    g["lr"] = 0.001 + 0.0018 * epoch
                if epoch <= 80 and epoch > 5:
                    g["lr"] = 0.01
                if epoch <= 110 and epoch > 80:
                    g["lr"] = 0.001
                if epoch > 110:
                    g["lr"] = 0.00001
        if lr_sched_conservative:
            for g in optimizer.param_groups:
                if epoch > 0 and epoch <= 5:
                    g["lr"] = 0.00001 + (0.00009 / 5) * (epoch)
                if epoch <= 80 and epoch > 5:
                    g["lr"] = 0.0001
                if epoch <= 110 and epoch > 80:
                    g["lr"] = 0.00001
                if epoch > 110:
                    g["lr"] = 0.000001

        train_loss_value = train(model, train_loader, optimizer, loss_fn)
        train_loss.append(train_loss_value)

        test_loss_value = test(model, test_loader, loss_fn)
        test_loss.append(test_loss_value)

        pred_bbox, target_bbox = get_bboxes(
            train_loader,
            model,
            iou_threshold=config["training"]["iou_threshold"],
            threshold=config["training"]["threshold"],
        )
        test_pred_bbox, test_target_bbox = get_bboxes(
            test_loader,
            model,
            iou_threshold=config["training"]["iou_threshold"],
            threshold=config["training"]["threshold"],
        )

        train_mAP_val = mAP(
            pred_bbox, target_bbox, iou_threshold=config["training"]["iou_threshold"]
        )
        test_mAP_val = mAP(
            test_pred_bbox,
            test_target_bbox,
            iou_threshold=config["training"]["iou_threshold"],
        )

        train_mAP.append(train_mAP_val.item())
        test_mAP.append(test_mAP_val.item())

        print(
            {
                "train_loss": train_loss_value,
                "test_loss": test_loss_value,
                "train_mAP": train_mAP_val,
                "test_mAP": test_mAP_val,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        wandb.log(
            {
                "train_loss": train_loss_value,
                "test_loss": test_loss_value,
                "train_mAP": train_mAP_val,
                "test_mAP": test_mAP_val,
                "lr": optimizer.param_groups[0]["lr"],
            },
            step=epoch + 1,
        )


if __name__ == "__main__":
    main()
