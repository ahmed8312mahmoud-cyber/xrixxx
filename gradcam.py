"""
gradcam.py
----------
تنفيذ Grad-CAM لعرض الـ Heatmap على صورة الأشعة، لتوضيح أي منطقة في الصورة
"نظر" الموديل إليها عشان يقرر وجود مرض معيّن.

الـ Hook موضوع على آخر طبقة في الـ backbone: features.norm5
(آخر BatchNorm2d قبل الـ ReLU + LSEPooling في SingleInputDenseNet.forward)
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Grad-CAM عام لأي موديل فيه self.features (Sequential من Conv layers)
    و forward بيرجع logits مباشرة (multi-label / multi-class).
    """

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        # الطبقة الافتراضية: آخر BatchNorm في backbone الـ DenseNet (norm5)
        self.target_layer = target_layer or model.features.norm5

        self._activations: Optional[torch.Tensor] = None

        # ملاحظة: نستخدم forward hook + retain_grad() على الـ tensor نفسه
        # بدل register_full_backward_hook، لأن forward بتاع الموديل بيعمل
        # F.relu(..., inplace=True) مباشرة بعد norm5، وde بيتعارض مع
        # full_backward_hook (PyTorch بيرفض inplace op على view راجع من hook).
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input_, output):
        # نحتفظ بنسخة "live" (بدون detach) عشان نقدر نعمل عليها retain_grad()
        # ونقرأ .grad منها بعد الـ backward، مع الحفاظ على ربطها بالـ autograd graph.
        output.retain_grad()
        self._activations = output

    def remove_hooks(self) -> None:
        self._fwd_handle.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        يحسب خريطة Grad-CAM (قيم 0..1) بحجم نفس الـ feature map،
        لاستخدامها بعد ذلك لعمل overlay على الصورة الأصلية.

        input_tensor: shape (1, 3, H, W), يجب أن يكون requires_grad جاهز داخلياً.
        class_idx: انديكس الكلاس (المرض) المطلوب توليد الـ heatmap بناءً عليه.
        """
        self.model.eval()
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        logits = self.model(input_tensor)          # (1, n_classes)
        score = logits[0, class_idx]

        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        activations = self._activations[0].detach()          # (C, H, W)
        gradients = self._activations.grad[0].detach()       # (C, H, W)

        # أهمية كل channel = المتوسط المكاني لتدرجاته (Grad-CAM الأصلي)
        weights = gradients.mean(dim=(1, 2))         # (C,)

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for c, w in enumerate(weights):
            cam += w * activations[c]

        cam = F.relu(cam)  # نهتم فقط بالمناطق التي تدعم الكلاس (تأثير إيجابي)

        cam_min, cam_max = cam.min(), cam.max()
        if (cam_max - cam_min) > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy()


def _apply_colormap(gray: np.ndarray) -> np.ndarray:
    """
    تحويل مصفوفة 2D (0..1) إلى صورة RGB بألوان Jet-like
    بدون الاعتماد على matplotlib/opencv (تجنباً لإضافة dependency ثقيلة).
    """
    # تقريب بسيط لـ colormap "jet": أزرق → سماوي → أخضر → أصفر → أحمر
    r = np.clip(1.5 - np.abs(4 * gray - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * gray - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * gray - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def overlay_heatmap(
    original_image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
    output_size: Optional[tuple] = None,
) -> Image.Image:
    """
    يدمج خريطة Grad-CAM (cam: numpy array 0..1, حجم صغير HxW) مع الصورة الأصلية.

    original_image: صورة PIL الأصلية (RGB)
    cam: مصفوفة 2D من generate()
    alpha: شفافية الـ heatmap فوق الصورة (0 = شفاف كامل، 1 = heatmap فقط)
    output_size: (W, H) الحجم النهائي للصورة المُخرجة؛ افتراضياً نفس حجم original_image
    """
    base = original_image.convert("RGB")
    target_size = output_size or base.size  # (W, H)

    # تكبير الـ CAM لحجم الصورة الأصلية
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(target_size, Image.BICUBIC)
    cam_resized = np.asarray(cam_img).astype(np.float32) / 255.0

    heatmap_rgb = _apply_colormap(cam_resized)  # (H, W, 3) uint8

    base_resized = base.resize(target_size)
    base_np = np.asarray(base_resized).astype(np.float32)

    blended = (1 - alpha) * base_np + alpha * heatmap_rgb.astype(np.float32)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def generate_gradcam_overlay(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image: Image.Image,
    class_idx: int,
    alpha: float = 0.45,
) -> Image.Image:
    """دالة مختصرة: تشغّل Grad-CAM كامل وترجع صورة overlay جاهزة للعرض."""
    cam_engine = GradCAM(model)
    try:
        cam = cam_engine.generate(input_tensor, class_idx)
        overlay = overlay_heatmap(original_image, cam, alpha=alpha)
    finally:
        cam_engine.remove_hooks()
    return overlay
