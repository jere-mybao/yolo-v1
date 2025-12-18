from __future__ import annotations

from typing import Any
import os
import time

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import yaml
import wandb

from yolo_v1.dataset.dataset import Dataset
from yolo_v1.loss.loss import YoloV1Loss
from yolo_v1.models.model import YoloV1_Resnet50
from yolo_v1.utils.utils import mAP, get_bboxes


def get_model_for_eval(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    accumulation_steps: int,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)
        loss = loss_fn(out, y) / accumulation_steps
        loss.backward()

        total_loss += float(loss.item() * accumulation_steps)

        do_step = ((batch_idx + 1) % accumulation_steps == 0) or (
            batch_idx == len(train_loader) - 1
        )
        if do_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return total_loss / max(1, len(train_loader))


@torch.no_grad()
def eval_loss(
    model: torch.nn.Module,
    test_loader: DataLoader[Any],
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        loss = loss_fn(out, y)
        total_loss += float(loss.item())
    return total_loss / max(1, len(test_loader))


def apply_lr_schedule(
    config: dict, optimizer: torch.optim.Optimizer, epoch: int
) -> None:
    lr_sched_original = bool(config["optimizer"]["lr_scheduler"]["original"])
    lr_sched_conservative = bool(config["optimizer"]["lr_scheduler"]["conservative"])

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


def main():
    with open("yolo_v1/config/yolo_trainer.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count()
    print(f"Using {gpu_count} GPUs")
    if gpu_count > 1:
        batch_size = config["training"]["batch_size"] * gpu_count
        nworkers = config["training"]["nworkers"] * gpu_count
    else:
        batch_size = config["training"]["batch_size"]
        nworkers = config["training"]["nworkers"]

    wandb.init(
        project=config["wandb"]["project"],
        name=config["wandb"]["name"],
        config=config,
    )

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
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nworkers,
        pin_memory=True,
        persistent_workers=(nworkers > 0),
        prefetch_factor=int(config["training"].get("prefetch_factor", 4)),
        drop_last=True,
    )
    test_loader = DataLoader[Any](
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nworkers,
        pin_memory=True,
        persistent_workers=(nworkers > 0),
        prefetch_factor=int(config["training"].get("prefetch_factor", 4)),
        drop_last=False,
    )

    model = YoloV1_Resnet50(
        grid_size=config["model"]["grid_size"],
        num_bboxes=config["model"]["num_bboxes"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    if bool(config["training"].get("compile", True)) and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = Adam(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    loss_fn = YoloV1Loss(
        grid_size=config["model"]["grid_size"],
        num_bboxes=config["model"]["num_bboxes"],
        num_classes=config["model"]["num_classes"],
    )

    epochs = int(config["training"]["epochs"])
    accumulation_steps = int(config["training"]["accumulation_steps"])

    map_every = int(config["training"].get("map_every", 5))
    eval_subset_batches = int(
        config["training"].get("eval_subset_batches", 0)
    )  # 0 = full
    save_every = int(config["training"].get("save_every", 2))
    save_model = bool(config["training"].get("save_model", True))
    ckpt_path = str(config["training"]["path_cpt_file"])

    for epoch in range(epochs):
        epoch_start = time.time()

        apply_lr_schedule(config, optimizer, epoch)

        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            accumulation_steps=accumulation_steps,
        )
        t1 = time.time()

        t2 = time.time()
        test_loss = eval_loss(
            model=model,
            test_loader=test_loader,
            loss_fn=loss_fn,
            device=device,
        )
        t3 = time.time()

        do_map = ((epoch + 1) % map_every == 0) or (epoch == epochs - 1)
        map_time = 0.0
        test_map_val = None

        if do_map:
            eval_model = get_model_for_eval(model)

            if eval_subset_batches > 0:
                from itertools import islice

                def limited_iter(loader, k):
                    for batch in islice(iter(loader), k):
                        yield batch

                class LimitedLoader:
                    def __init__(self, loader, k):
                        self.loader = loader
                        self.k = k

                    def __iter__(self):
                        return limited_iter(self.loader, self.k)

                    def __len__(self):
                        return min(len(self.loader), self.k)

                eval_loader = LimitedLoader(test_loader, eval_subset_batches)
            else:
                eval_loader = test_loader

            mt0 = time.time()
            pred_bbox, target_bbox = get_bboxes(
                eval_loader,
                eval_model,
                iou_threshold=float(config["training"]["iou_threshold"]),
                threshold=float(config["training"]["threshold"]),
            )
            test_map_val = mAP(
                pred_bbox,
                target_bbox,
                iou_threshold=float(config["training"]["iou_threshold"]),
            )
            mt1 = time.time()
            map_time = mt1 - mt0

        epoch_time = time.time() - epoch_start

        log_dict = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_time_s": t1 - t0,
            "test_time_s": t3 - t2,
            "map_time_s": map_time,
            "epoch_time_s": epoch_time,
        }
        if test_map_val is not None:
            log_dict["test_mAP"] = float(test_map_val.item())

        print(log_dict)
        if config["wandb"]["enabled"]:
            wandb.log(log_dict, step=epoch + 1)

    if save_model and (((epoch + 1) % save_every == 0) or (epoch == epochs - 1)):
        to_save = {
            "epoch": epoch,
            "model_state_dict": get_model_for_eval(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        }
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(to_save, ckpt_path)


if __name__ == "__main__":
    main()
