from PIL import Image
from torchvision import transforms
import torch
import io


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_image(image_bytes: bytes) -> tuple[torch.Tensor, Image.Image]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(image).unsqueeze(0)
    return tensor, image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    inv = transforms.Normalize(
        mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1 / s for s in IMAGENET_STD],
    )
    img = inv(tensor.squeeze(0)).clamp(0, 1)
    return transforms.ToPILImage()(img)
