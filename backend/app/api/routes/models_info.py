from fastapi import APIRouter
from app.ml.inference import get_models_status
from app.config import MODEL_LABELS, settings
from app.schemas.diagnosis import ModelInfo, ModelsStatusResponse

router = APIRouter(prefix="/models", tags=["models"])

_DESCRIPTIONS = {
    "mobilenet_v2": "Leve e eficiente, projetado para dispositivos móveis e edge AI. Usa depthwise separable convolutions.",
    "resnet50": "Arquitetura clássica com skip connections (conexões residuais) que previnem o vanishing gradient em redes profundas.",
    "efficientnet_b0": "Compound scaling que equilibra profundidade, largura e resolução. Melhor acurácia por parâmetro.",
    "vit_b_16": "Vision Transformer que divide a imagem em patches de 16×16 e aplica atenção global. Estado da arte.",
}


@router.get("/", response_model=ModelsStatusResponse)
async def list_models():
    statuses = get_models_status()
    return ModelsStatusResponse(
        models=[
            ModelInfo(
                id=m["id"],
                label=MODEL_LABELS[m["id"]],
                calibrated=m["calibrated"],
                description=_DESCRIPTIONS[m["id"]],
            )
            for m in statuses
        ],
        demo_mode=settings.demo_mode,
    )
