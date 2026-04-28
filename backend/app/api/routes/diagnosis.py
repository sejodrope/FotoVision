import json
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.models import Diagnosis
from app.ml.preprocessing import preprocess_image
from app.ml.inference import load_model, predict, is_model_calibrated
from app.ml.gradcam import generate_gradcam
from app.config import (
    CLASS_NAMES,
    CLASS_LABELS,
    AVAILABLE_MODELS,
    ALLOWED_CONTENT_TYPES,
    settings,
)
from app.schemas.diagnosis import DiagnosisResult

logger = logging.getLogger("fitovision.diagnosis")
router = APIRouter(prefix="/diagnosis", tags=["diagnosis"])


@router.post("/", response_model=DiagnosisResult)
async def run_diagnosis(
    file: UploadFile = File(...),
    model_name: str = Form("mobilenet_v2"),
    generate_gradcam_flag: bool = Form(True),
    db: AsyncSession = Depends(get_db),
):
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Modelo inválido: {model_name}")

    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo de ficheiro não suportado: {file.content_type}. Use JPEG, PNG, WebP ou BMP.",
        )

    image_bytes = await file.read()

    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Ficheiro demasiado grande. Máximo permitido: {settings.max_upload_bytes // (1024*1024)} MB.",
        )

    tensor, pil_image = preprocess_image(image_bytes)

    model = load_model(model_name)
    result = predict(model, tensor)

    predicted_class = result["predicted_class"]

    if settings.confidence_threshold > 0 and result["confidence"] < settings.confidence_threshold:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Confiança insuficiente ({result['confidence']:.1%}). "
                "A imagem pode não conter uma planta reconhecível."
            ),
        )

    class_idx = CLASS_NAMES.index(predicted_class)

    gradcam_b64 = None
    if generate_gradcam_flag:
        try:
            gradcam_b64 = generate_gradcam(model, model_name, tensor, pil_image, class_idx)
        except Exception as exc:
            logger.warning("GradCAM falhou para modelo '%s': %s", model_name, exc)

    record = Diagnosis(
        model_name=model_name,
        predicted_class=predicted_class,
        confidence=result["confidence"],
        scores_json=json.dumps(result["scores"]),
        gradcam_image=gradcam_b64,
        original_filename=file.filename,
        demo_mode=result["demo_mode"],
    )
    try:
        db.add(record)
        await db.commit()
        await db.refresh(record)
    except Exception as exc:
        await db.rollback()
        logger.error("Erro ao guardar diagnóstico na DB: %s", exc)
        raise HTTPException(status_code=500, detail="Erro interno ao guardar diagnóstico.")

    logger.info(
        "Diagnóstico id=%d | modelo=%s | classe=%s | confiança=%.2f",
        record.id,
        model_name,
        predicted_class,
        result["confidence"],
    )

    return DiagnosisResult(
        id=record.id,
        created_at=record.created_at,
        model_name=model_name,
        predicted_class=predicted_class,
        predicted_label=CLASS_LABELS[predicted_class],
        confidence=result["confidence"],
        scores=result["scores"],
        gradcam_image=gradcam_b64,
        demo_mode=result["demo_mode"],
        calibrated=is_model_calibrated(model_name),
        original_filename=file.filename,
    )
