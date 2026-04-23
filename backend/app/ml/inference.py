import os
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from app.config import settings, CLASS_NAMES, AVAILABLE_MODELS


_model_cache: dict[str, nn.Module] = {}
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(name: str, num_classes: int) -> nn.Module:
    if name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Modelo desconhecido: {name}")
    return model


def load_model(name: str) -> nn.Module:
    if name in _model_cache:
        return _model_cache[name]

    model = _build_model(name, settings.num_classes)

    weight_path = Path(settings.weights_dir) / f"{name}.pth"
    if weight_path.exists():
        state = torch.load(weight_path, map_location=_device)
        model.load_state_dict(state)

    model.to(_device)
    model.eval()
    _model_cache[name] = model
    return model


def predict(model: nn.Module, tensor: torch.Tensor) -> dict:
    tensor = tensor.to(_device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    scores = {cls: float(probs[i]) for i, cls in enumerate(CLASS_NAMES)}
    predicted = max(scores, key=scores.get)
    confidence = scores[predicted]

    return {
        "predicted_class": predicted,
        "confidence": confidence,
        "scores": scores,
        "demo_mode": settings.demo_mode,
        "device": str(_device),
    }


def is_model_calibrated(name: str) -> bool:
    weight_path = Path(settings.weights_dir) / f"{name}.pth"
    return weight_path.exists()


def get_models_status() -> list[dict]:
    return [
        {
            "id": name,
            "calibrated": is_model_calibrated(name),
        }
        for name in AVAILABLE_MODELS
    ]
