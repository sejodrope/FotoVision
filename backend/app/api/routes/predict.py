import logging

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.ml.preprocessing import preprocess_image
from app.ml.inference import predict_binary
from app.config import ALLOWED_CONTENT_TYPES, settings

logger = logging.getLogger("fitovision.predict")
router = APIRouter(prefix="/predict", tags=["predict"])


class PredictResult(BaseModel):
    label: str           # "healthy" | "anomalous"
    confidence: float
    healthy_prob: float
    anomalous_prob: float


@router.post("/", response_model=PredictResult)
async def predict_binary_endpoint(file: UploadFile = File(...)):
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo de ficheiro não suportado: {file.content_type}. Use JPEG, PNG, WebP ou BMP.",
        )

    image_bytes = await file.read()

    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Ficheiro demasiado grande. Máximo: {settings.max_upload_bytes // (1024 * 1024)} MB.",
        )

    tensor, _ = preprocess_image(image_bytes)

    try:
        result = predict_binary(tensor)
    except FileNotFoundError as exc:
        logger.error("Pesos binários ausentes: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))

    logger.info(
        "Predição binária | label=%s | confiança=%.4f | healthy=%.4f | anomalous=%.4f",
        result["label"],
        result["confidence"],
        result["healthy_prob"],
        result["anomalous_prob"],
    )

    return PredictResult(**result)