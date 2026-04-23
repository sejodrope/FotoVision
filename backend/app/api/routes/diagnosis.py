import json
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db
from app.db.models import Diagnosis
from app.ml.preprocessing import preprocess_image
from app.ml.inference import load_model, predict, is_model_calibrated
from app.ml.gradcam import generate_gradcam
from app.config import CLASS_NAMES, CLASS_LABELS, AVAILABLE_MODELS
from app.schemas.diagnosis import DiagnosisResult

router = APIRouter(prefix="/diagnosis", tags=["diagnosis"])

_MODEL_DESCRIPTIONS = {
    "mobilenet_v2": "Leve e eficiente. Ideal para dispositivos móveis e edge AI.",
    "resnet50": "Arquitetura clássica com skip connections. Alta robustez.",
    "efficientnet_b0": "Escalonamento compound. Melhor precisão com menos parâmetros.",
    "vit_b_16": "Vision Transformer. Captura dependências globais na imagem.",
}


@router.post("/", response_model=DiagnosisResult)
async def run_diagnosis(
    file: UploadFile = File(...),
    model_name: str = Form("mobilenet_v2"),
    generate_gradcam_flag: bool = Form(True),
    db: AsyncSession = Depends(get_db),
):
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Modelo inválido: {model_name}")

    image_bytes = await file.read()
    tensor, pil_image = preprocess_image(image_bytes)

    model = load_model(model_name)
    result = predict(model, tensor)

    predicted_class = result["predicted_class"]
    class_idx = CLASS_NAMES.index(predicted_class)

    gradcam_b64 = None
    if generate_gradcam_flag:
        try:
            gradcam_b64 = generate_gradcam(model, model_name, tensor, pil_image, class_idx)
        except Exception:
            gradcam_b64 = None

    record = Diagnosis(
        model_name=model_name,
        predicted_class=predicted_class,
        confidence=result["confidence"],
        scores_json=json.dumps(result["scores"]),
        gradcam_image=gradcam_b64,
        original_filename=file.filename,
        demo_mode=result["demo_mode"],
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

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
