from datetime import datetime
from pydantic import BaseModel


class DiagnosisRequest(BaseModel):
    model_name: str = "mobilenet_v2"
    generate_gradcam: bool = True


class DiagnosisResult(BaseModel):
    id: int
    created_at: datetime
    model_name: str
    predicted_class: str
    predicted_label: str
    confidence: float
    scores: dict[str, float]
    gradcam_image: str | None
    demo_mode: bool
    calibrated: bool
    original_filename: str | None


class DiagnosisListItem(BaseModel):
    id: int
    created_at: datetime
    model_name: str
    predicted_class: str
    predicted_label: str
    confidence: float
    demo_mode: bool
    original_filename: str | None


class ModelInfo(BaseModel):
    id: str
    label: str
    calibrated: bool
    description: str


class ModelsStatusResponse(BaseModel):
    models: list[ModelInfo]
    demo_mode: bool
