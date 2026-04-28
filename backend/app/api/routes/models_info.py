from fastapi import APIRouter

from app.ml.inference import get_models_status
from app.config import MODEL_LABELS, MODEL_DESCRIPTIONS, settings
from app.schemas.diagnosis import ModelInfo, ModelsStatusResponse

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/", response_model=ModelsStatusResponse)
async def list_models():
    statuses = get_models_status()
    return ModelsStatusResponse(
        models=[
            ModelInfo(
                id=m["id"],
                label=MODEL_LABELS[m["id"]],
                calibrated=m["calibrated"],
                description=MODEL_DESCRIPTIONS[m["id"]],
            )
            for m in statuses
        ],
        demo_mode=settings.demo_mode,
    )
