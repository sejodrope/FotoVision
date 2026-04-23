from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    demo_mode: bool = True
    database_url: str = "sqlite+aiosqlite:///./fitovision.db"
    weights_dir: str = "./weights"
    num_classes: int = 6
    cors_origins: str = "http://localhost:5173"

    model_config = {"env_file": ".env"}


settings = Settings()

CLASS_NAMES = [
    "saudavel",
    "mildio",
    "oidio",
    "clorose_nitrogenio",
    "danos_pragas",
    "estresse_hidrico",
]

CLASS_LABELS = {
    "saudavel": "Saudável",
    "mildio": "Míldio",
    "oidio": "Oídio",
    "clorose_nitrogenio": "Clorose por Nitrogênio",
    "danos_pragas": "Danos por Pragas",
    "estresse_hidrico": "Estresse Hídrico",
}

CLASS_COLORS = {
    "saudavel": "#22c55e",
    "mildio": "#ef4444",
    "oidio": "#f97316",
    "clorose_nitrogenio": "#eab308",
    "danos_pragas": "#8b5cf6",
    "estresse_hidrico": "#3b82f6",
}

AVAILABLE_MODELS = ["mobilenet_v2", "resnet50", "efficientnet_b0", "vit_b_16"]

MODEL_LABELS = {
    "mobilenet_v2": "MobileNetV2",
    "resnet50": "ResNet-50",
    "efficientnet_b0": "EfficientNet-B0",
    "vit_b_16": "ViT-B/16",
}
