from typing import Any
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from yolo_v1.dataset.dataset import Dataset
from yolo_v1.loss.loss import YoloV1Loss
from yolo_v1.models.model import YoloV1_Resnet50
from yolo_v1.utils.utils import mAP, get_bboxes
import wandb

wandb.init(project="yolo-v1", name="yolo-v1-pascal-voc")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_count = torch.cuda.device_count()
print(f"Using {gpu_count} GPUs")
if gpu_count > 1:
    batch_size = 4 * gpu_count
    nworkers = 4 * gpu_count
else:
    batch_size = 4
    nworkers = 2

weight_decay = 0.0005
epochs = 200
lr_sched_original = False
lr_sched_conservative = True


def train(model, train_loader, optimizer, loss_fn):
    accumulation_steps = 16 
    total_loss = 0
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        with torch.set_grad_enabled(True):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            total_loss += loss.item()
            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
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
    train_dataset = Dataset(csv_file="data/train.csv", img_dir="data/images", label_dir="data/labels", grid_size=7, num_bboxes=2, num_classes=20, additional_transform=True)
    test_dataset = Dataset(csv_file="data/test.csv", img_dir="data/images", label_dir="data/labels", grid_size=7, num_bboxes=2, num_classes=20, additional_transform=False)
    
    train_loader = DataLoader[Any](train_dataset, batch_size=batch_size, shuffle=True, num_workers=nworkers)
    test_loader = DataLoader[Any](test_dataset, batch_size=batch_size, shuffle=False, num_workers=nworkers)

    model = YoloV1_Resnet50(grid_size=7, num_bboxes=2, num_classes=20).to(device)
    if gpu_count > 1:
        model = torch.nn.DataParallel(model)

    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=weight_decay)
    loss_fn = YoloV1Loss(grid_size=7, num_bboxes=2, num_classes=20)

    train_loss  = []
    train_mAP = []
    test_loss = []
    test_mAP = []

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        if lr_sched_original == True:
            for g in optimizer.param_groups:
                if epoch > 0 and epoch <= 5:
                    g['lr'] = 0.001 + 0.0018 * epoch
                if epoch <=80 and epoch > 5:
                    g['lr'] = 0.01
                if epoch <= 110 and epoch > 80:
                    g['lr'] = 0.001
                if epoch > 110:
                    g['lr'] = 0.00001
        if lr_sched_conservative == True:
             for g in optimizer.param_groups:
                if epoch > 0 and epoch <= 5:
                    g['lr'] = 0.00001 +(0.00009/5) * (epoch)
                if epoch <=80 and epoch > 5:
                    g['lr'] = 0.0001
                if epoch <= 110 and epoch > 80:
                    g['lr'] = 0.00001
                if epoch > 110:
                    g['lr'] = 0.000001
                           
        train_loss_value = train(model, train_loader, optimizer, loss_fn)
        train_loss.append(train_loss_value)

        test_loss_value = test(model, test_loader, loss_fn)
        test_loss.append(test_loss_value)

        pred_bbox, target_bbox = get_bboxes(train_loader, model, iou_threshold = 0.5, threshold = 0.4)
        test_pred_bbox, test_target_bbox = get_bboxes(test_loader, model, iou_threshold = 0.5, threshold = 0.4)   

        train_mAP_val = mAP(pred_bbox, target_bbox, iou_threshold = 0.5)
        test_mAP_val = mAP(test_pred_bbox, test_target_bbox, iou_threshold = 0.5)

        train_mAP.append(train_mAP_val.item())
        test_mAP.append(test_mAP_val.item())

        print({"train_loss": train_loss_value,"test_loss": test_loss_value,"train_mAP": train_mAP_val,"test_mAP": test_mAP_val,"lr": optimizer.param_groups[0]["lr"]})
        wandb.log({"train_loss": train_loss_value,"test_loss": test_loss_value,"train_mAP": train_mAP_val,"test_mAP": test_mAP_val,"lr": optimizer.param_groups[0]["lr"]}, step=epoch + 1)

if __name__ == "__main__":
    main()