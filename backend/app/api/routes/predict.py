import logging

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from pydantic import BaseModel

from app.ml.preprocessing import preprocess_image
from app.ml.inference import predict_binary, load_binary_model_by_name, predict_binary_with_model
from app.ml.gradcam import generate_gradcam
from app.config import ALLOWED_CONTENT_TYPES, settings

logger = logging.getLogger("fitovision.predict")
router = APIRouter(prefix="/predict", tags=["predict"])

_BINARY_MODEL_NAMES = {"efficientnet_b0", "mobilenet_v2", "resnet50"}


class PredictResult(BaseModel):
    # "healthy" | "anomalous" | "inconclusive" | "not_a_leaf"
    #
    # 'inconclusive' e 'not_a_leaf' são estados novos e deliberados: o sistema
    # anterior devolvia sempre healthy/anomalous, mesmo para fotos que não eram
    # folhas, sempre com confiança alta. Abster-se é a resposta correcta quando o
    # modelo não sabe.
    label: str
    confidence: float
    healthy_prob: float
    anomalous_prob: float
    calibrated: bool = False
    vegetation_fraction: float | None = None
    message: str | None = None


class GradcamResult(BaseModel):
    label: str
    confidence: float
    healthy_prob: float
    anomalous_prob: float
    gradcam_image: str  # data:image/png;base64,...


def _validate_upload(file: UploadFile, image_bytes: bytes):
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo de ficheiro não suportado: {file.content_type}. Use JPEG, PNG, WebP ou BMP.",
        )
    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Ficheiro demasiado grande. Máximo: {settings.max_upload_bytes // (1024 * 1024)} MB.",
        )


@router.post("/", response_model=PredictResult)
async def predict_binary_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    _validate_upload(file, image_bytes)
    tensor, pil_image = preprocess_image(image_bytes)
    try:
        # A imagem PIL vai junto para a guarda de vegetação (ExG) poder correr
        # sobre os píxeis originais, antes da normalização.
        result = predict_binary(tensor, image=pil_image)
    except FileNotFoundError as exc:
        logger.error("Pesos binários ausentes: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    logger.info(
        "Predição | label=%s | confiança=%.4f | healthy=%.4f | anomalous=%.4f | veg=%s",
        result["label"],
        result["confidence"],
        result["healthy_prob"],
        result["anomalous_prob"],
        f"{result['vegetation_fraction']:.3f}" if result.get("vegetation_fraction") is not None else "n/d",
    )
    return PredictResult(**result)


@router.post("/gradcam", response_model=GradcamResult)
async def predict_gradcam_endpoint(
    file: UploadFile = File(...),
    model_name: str = Query(default="efficientnet_b0"),
):
    if model_name not in _BINARY_MODEL_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Modelo não suportado: '{model_name}'. Use: efficientnet_b0, mobilenet_v2, resnet50.",
        )
    image_bytes = await file.read()
    _validate_upload(file, image_bytes)
    tensor, original_image = preprocess_image(image_bytes)
    try:
        model = load_binary_model_by_name(model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    result = predict_binary_with_model(model, tensor)
    pred_idx = 0 if result["label"] == "healthy" else 1
    gradcam_b64 = generate_gradcam(model, model_name, tensor, original_image, pred_idx)
    logger.info(
        "Grad-CAM | model=%s | label=%s | conf=%.4f",
        model_name, result["label"], result["confidence"],
    )
    return GradcamResult(**result, gradcam_image=gradcam_b64)
