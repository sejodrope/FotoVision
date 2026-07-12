import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from app.config import settings, CLASS_NAMES, AVAILABLE_MODELS

logger = logging.getLogger("fitovision.inference")

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
        state = torch.load(weight_path, map_location=_device, weights_only=True)
        model.load_state_dict(state)
        logger.info("Pesos carregados: %s", weight_path)
    else:
        logger.warning(
            "Pesos não encontrados para '%s' (%s). A usar pesos ImageNet — modo demo activado.",
            name,
            weight_path,
        )

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


_BINARY_MODEL_NAME = "efficientnet_b0"
_BINARY_WEIGHT_FILENAME = f"{_BINARY_MODEL_NAME}_binary.pth"
_BINARY_CALIB_FILENAME = f"{_BINARY_MODEL_NAME}_binary_calibration.json"
_BINARY_CLASSES = ["healthy", "anomalous"]  # índice 0 = healthy, 1 = anomalous

# Calibração carregada de disco (produzida por calibrate.py).
# Defaults neutros: T=1 (sem correcção) e limiar do config.
_calibration: dict | None = None


def _load_calibration() -> dict:
    """Carrega temperatura e limiar de abstenção do ficheiro gerado por calibrate.py."""
    global _calibration
    if _calibration is not None:
        return _calibration

    path = Path(settings.weights_dir) / _BINARY_CALIB_FILENAME
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            _calibration = {
                "temperature": float(data.get("temperature", 1.0)),
                "threshold": float(data.get("threshold", settings.confidence_threshold)),
            }
            logger.info(
                "Calibração carregada: T=%.4f | limiar=%.2f",
                _calibration["temperature"], _calibration["threshold"],
            )
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("Calibração ilegível (%s) — a usar defaults sem correcção.", exc)
            _calibration = {"temperature": 1.0, "threshold": settings.confidence_threshold}
    else:
        logger.warning(
            "Sem ficheiro de calibração (%s). As probabilidades NÃO estão calibradas "
            "e tendem a ser excessivamente confiantes. Execute: python calibrate.py",
            path,
        )
        _calibration = {"temperature": 1.0, "threshold": settings.confidence_threshold}

    return _calibration


def load_binary_model() -> nn.Module:
    key = "efficientnet_b0_binary"
    if key in _model_cache:
        return _model_cache[key]

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    weight_path = Path(settings.weights_dir) / _BINARY_WEIGHT_FILENAME
    if not weight_path.exists():
        raise FileNotFoundError(f"Pesos binários não encontrados: {weight_path}")

    state = torch.load(weight_path, map_location=_device, weights_only=True)
    model.load_state_dict(state)
    model.to(_device)
    model.eval()
    _model_cache[key] = model
    logger.info("Modelo binário carregado de %s (device=%s)", weight_path, _device)
    return model


def vegetation_fraction(image: Image.Image) -> float:
    """
    Fracção de píxeis que parecem vegetação, pelo índice Excess Green (ExG).

        ExG = 2g - r - b   (sobre canais RGB normalizados; Woebbecke et al., 1995)

    Serve de guarda de domínio: o classificador foi treinado só com folhas, mas o
    softmax devolve uma resposta confiante para QUALQUER imagem — incluindo um gato,
    uma parede ou um print de ecrã. Sem esta verificação, o sistema diagnostica
    'anomalous' com 97% de confiança numa foto que não tem planta nenhuma, que é
    precisamente o comportamento que invalida o teste com fotos aleatórias.

    Devolve um valor em [0, 1]. Folhas típicas ficam bem acima de 0,25.
    """
    small = image.convert("RGB").resize((128, 128))
    arr = np.asarray(small, dtype=np.float32) / 255.0

    total = arr.sum(axis=2, keepdims=True)
    total[total == 0] = 1.0                 # evita divisão por zero em píxeis pretos
    r, g, b = (arr / total)[..., 0], (arr / total)[..., 1], (arr / total)[..., 2]

    exg = 2.0 * g - r - b
    return float((exg > 0.05).mean())


def predict_binary(tensor: torch.Tensor, image: Image.Image | None = None) -> dict:
    """
    Classifica uma folha em healthy/anomalous, com calibração e abstenção.

    Três resultados possíveis:
      • 'healthy' / 'anomalous'  — diagnóstico com confiança acima do limiar
      • 'inconclusive'           — o modelo não tem confiança suficiente
      • 'not_a_leaf'             — a imagem não parece conter vegetação

    As probabilidades devolvidas são CALIBRADAS (temperature scaling); ao contrário
    do softmax cru, um valor de 0,80 significa de facto ~80% de acerto.
    """
    calib = _load_calibration()
    temperature = calib["temperature"]
    threshold = calib["threshold"]

    # ── Guarda de domínio: isto é sequer uma folha? ───────────────────────────
    veg = None
    if image is not None:
        veg = vegetation_fraction(image)
        if veg < settings.min_vegetation_fraction:
            logger.info("Imagem rejeitada: vegetação=%.3f < %.3f",
                        veg, settings.min_vegetation_fraction)
            return {
                "label": "not_a_leaf",
                "confidence": 0.0,
                "healthy_prob": 0.0,
                "anomalous_prob": 0.0,
                "vegetation_fraction": veg,
                "calibrated": temperature != 1.0,
                "message": (
                    "A imagem não parece conter uma folha. Envie uma foto aproximada "
                    "da folha, bem iluminada e enquadrada."
                ),
            }

    model = load_binary_model()
    tensor = tensor.to(_device)
    with torch.no_grad():
        logits = model(tensor)
        # Temperature scaling: achata os logits para que a confiança seja honesta.
        probs = torch.softmax(logits / temperature, dim=1)[0]

    healthy_prob = float(probs[0])
    anomalous_prob = float(probs[1])
    confidence = max(healthy_prob, anomalous_prob)
    label = "healthy" if healthy_prob >= anomalous_prob else "anomalous"

    # ── Abstenção: confiança insuficiente ⇒ não arrisca um diagnóstico ────────
    if confidence < threshold:
        logger.info("Predição inconclusiva: confiança=%.4f < limiar=%.2f", confidence, threshold)
        return {
            "label": "inconclusive",
            "confidence": confidence,
            "healthy_prob": healthy_prob,
            "anomalous_prob": anomalous_prob,
            "vegetation_fraction": veg,
            "calibrated": temperature != 1.0,
            "message": (
                f"Confiança insuficiente ({confidence:.0%}) para um diagnóstico fiável. "
                "Tente uma foto mais nítida, com a folha preenchendo o enquadramento."
            ),
        }

    return {
        "label": label,
        "confidence": confidence,
        "healthy_prob": healthy_prob,
        "anomalous_prob": anomalous_prob,
        "vegetation_fraction": veg,
        "calibrated": temperature != 1.0,
        "message": None,
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
