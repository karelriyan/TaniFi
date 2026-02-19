# src/simulation/model.py
import torch.nn as nn

class YOLOv11ClassificationModel(nn.Module):
    """
    YOLOv11 Small classification model wrapper for federated learning.
    Uses pretrained ImageNet backbone with custom classification head.
    Exposes .features and .classifier for LoRA adapter insertion.
    """
    def __init__(self, num_classes=3):
        super().__init__()
        from ultralytics import YOLO

        yolo = YOLO('yolo11s-cls.pt')
        yolo_model = yolo.model.model  # nn.Sequential of YOLO layers

        # Backbone: layers 0‑9 (Conv, C3k2, C2PSA blocks)
        backbone = nn.Sequential(*list(yolo_model.children())[:-1])

        # Classify layer has: conv(256→1280) + pool + drop + linear(1280→1000)
        classify_layer = yolo_model[-1]

        # Features = backbone + classify.conv + classify.pool → [batch, 1280, 1, 1]
        self.features = nn.Sequential(
            backbone,
            classify_layer.conv,
            classify_layer.pool,
        )

        # Feature dimension after pooling
        self.feature_dim = 1280

        # New classification head for our num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

__all__ = ["YOLOv11ClassificationModel"]
