import torch
import pandas as pd
from PIL import Image
import os
import torchvision.transforms as T
from yolo_v1.utils.utils import transform_image, transform_bboxes


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes


train_transform = Compose(
    [
        T.Resize((448, 448)),
        T.ColorJitter(brightness=[0, 1.5], saturation=[0, 1.5]),
        T.ToTensor(),
    ]
)

test_transform = Compose([T.Resize((448, 448)), T.ToTensor()])


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        grid_size=7,
        num_bboxes=2,
        num_classes=20,
        transform=None,
        additional_transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.grid_size = grid_size
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes
        self.additional_transform = additional_transform
        if self.additional_transform:
            self.transform = train_transform
        else:
            self.transform = test_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_filepath = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes_data = []
        with open(label_filepath) as label_file:
            for label_line in label_file.readlines():
                class_label, x, y, width, height = [
                    float(val) if float(val) != int(float(val)) else int(val)
                    for val in label_line.replace("\n", "").split()
                ]
                boxes_data.append([class_label, x, y, width, height])

        image_filepath = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(image_filepath)

        if self.additional_transform:
            image, transform_parameters = transform_image(image)
            boxes_data = transform_bboxes(boxes_data, transform_parameters)

        boxes_tensor = torch.tensor(boxes_data)
        if self.transform:
            image, boxes_tensor = self.transform(image, boxes_tensor)

        target_matrix = torch.zeros(
            (self.grid_size, self.grid_size, self.num_classes + 5 * self.num_bboxes)
        )

        for box_item in boxes_tensor:
            class_label, x, y, width, height = box_item.tolist()
            class_label = int(class_label)

            cell_y, cell_x = int(self.grid_size * y), int(self.grid_size * x)
            x_in_cell, y_in_cell = (
                self.grid_size * x - cell_x,
                self.grid_size * y - cell_y,
            )

            width, height = (
                width * self.grid_size,
                height * self.grid_size,
            )

            if target_matrix[cell_y, cell_x, 20] == 0:
                target_matrix[cell_y, cell_x, 20] = 1
                bbox_coordinates = torch.tensor([x_in_cell, y_in_cell, width, height])
                target_matrix[cell_y, cell_x, 21:25] = bbox_coordinates
                target_matrix[cell_y, cell_x, class_label] = 1

        return image, target_matrix
