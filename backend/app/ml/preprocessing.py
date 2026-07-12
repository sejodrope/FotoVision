import io
import torch
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ATENÇÃO: tem de ser IDÊNTICO ao VAL_TRANSFORMS do dataset.py.
#
# Antes era Resize((224, 224)) — um resize que ESMAGA a proporção da imagem. Como
# o treino usava RandomResizedCrop (que preserva a proporção), o modelo era servido
# em produção com uma geometria que nunca viu no treino. Numa foto de telemóvel
# (4:3 ou 16:9) a distorção é grande, e é uma das razões pelas quais o desempenho
# em fotos reais não correspondia ao do conjunto de teste.
#
# Resize(256) + CenterCrop(224) é a convenção do ImageNet e mantém a proporção.
_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_image(image_bytes: bytes) -> tuple[torch.Tensor, Image.Image]:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="Ficheiro não é uma imagem válida.")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Erro ao processar imagem: {exc}")
    tensor = _transform(image).unsqueeze(0)
    return tensor, image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    inv = transforms.Normalize(
        mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1 / s for s in IMAGENET_STD],
    )
    img = inv(tensor.squeeze(0)).clamp(0, 1)
    return transforms.ToPILImage()(img)
