import torch
import pandas as pd
from PIL import Image
import os
from utils.utils import transform_image, transform_bboxes


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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, grid_size=7, num_bboxes=2, num_classes=20, transform=None, additional_transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.grid_size = grid_size
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes
        self.transform = transform
        self.additional_transform = additional_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_filepath = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes_data = []
        with open(label_filepath) as label_file:
            for label_line in label_file.readlines():
                class_label, x, y, width, height = [
                    float(val) if float(val) != int(float(val)) else int(val)
                    for val in label_line.replace("\n", "").split()]
                boxes_data.append([class_label, x, y, width, height])

        image_filepath = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(image_filepath)

        if self.additional_transform:
            image, transform_parameters = transform_image(image)
            boxes_data = transform_bboxes(boxes_data, transform_parameters)
        
        boxes_tensor = torch.tensor(boxes_data)
        if self.transform:
            image, boxes_tensor = self.transform(image, boxes_tensor)

        target_matrix = torch.zeros((self.grid_size, self.grid_size, self.num_classes + 5 * self.num_bboxes))

        for box_item in boxes_tensor:
            class_label, x, y, width, height = box_item.tolist()
            class_label = int(class_label)

            cell_y, cell_x = int(self.grid_size * y), int(self.grid_size * x)
            x_in_cell, y_in_cell = self.grid_size * x - cell_x, self.grid_size * y - cell_y

            if target_matrix[cell_y, cell_x, 20] == 0:
                target_matrix[cell_y, cell_x, 20] = 1
                bbox_coordinates = torch.tensor([x_in_cell, y_in_cell, width, height])
                target_matrix[cell_y, cell_x, 21:25] = bbox_coordinates
                target_matrix[cell_y, cell_x, class_label] = 1

        return image, target_matrix


def train(model, train_loader, optimizer, loss_fn):
    accumulation_steps = 16 
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        with torch.set_grad_enabled(True):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    return float(loss.item())


def test(model, test_loader, loss_fn):
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
    return float(loss.item())

def main():
    pass  