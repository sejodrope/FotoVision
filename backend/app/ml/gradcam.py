import base64
import io
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def _get_target_layer(model: nn.Module, model_name: str):
    if model_name == "resnet50":
        return [model.layer4[-1]]
    elif model_name == "mobilenet_v2":
        return [model.features[-1]]
    elif model_name == "efficientnet_b0":
        return [model.features[-1]]
    elif model_name == "vit_b_16":
        return [model.encoder.layers[-1].ln_1]
    raise ValueError(f"Modelo não suportado para Grad-CAM: {model_name}")


def _vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    h = w = int((tensor.shape[1] - 1) ** 0.5)
    result = tensor[:, 1:, :].reshape(tensor.size(0), h, w, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)


def generate_gradcam(
    model: nn.Module,
    model_name: str,
    tensor: torch.Tensor,
    original_image: Image.Image,
    target_class_idx: int,
) -> str:
    reshape = _vit_reshape_transform if model_name == "vit_b_16" else None
    target_layers = _get_target_layer(model, model_name)

    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape)

    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    targets = [ClassifierOutputTarget(target_class_idx)]

    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]

    rgb = np.array(original_image.resize((224, 224))).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

    pil_overlay = Image.fromarray(overlay)
    buffer = io.BytesIO()
    pil_overlay.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
