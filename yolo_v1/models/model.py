import torch
import torch.nn as nn
import torchvision.models as models

class YoloV1_Resnet50(nn.Module):
    def __init__(self, grid_size = 7, num_bboxes = 2, num_classes = 20):
        # original implementation used darknet as backbone (no pretrained weights in pytorch)
        super(YoloV1_Resnet50, self).__init__()
        self.backbone = nn.Sequential(*list(models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2').children())[:-2])
        self.detection_head_conv = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 448, 448)
            dummy_output = self.detection_head_conv(self.backbone(dummy_input))
            in_features = dummy_output.shape[1]

        self.detection_head_fc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=4096, out_features=grid_size * grid_size * (num_classes + num_bboxes * 5)),
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        # xavier initialization was not used in the original implementation
        for layer in self.detection_head_conv:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        for layer in self.detection_head_fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, input_tensor):
        input_tensor = self.backbone(input_tensor)
        input_tensor = self.detection_head_conv(input_tensor)
        input_tensor = self.detection_head_fc(input_tensor)
        return input_tensor