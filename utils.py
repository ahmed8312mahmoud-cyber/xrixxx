"""
utils.py
--------
دوال الـ Preprocessing والـ Inference.
الـ Transforms مستخرجة بشكل مطابق من val_transforms في الـ notebook.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Tuple, List, Dict


# ======================================================================
# ثوابت المشروع
# ======================================================================

CLASS_NAMES: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

# ======================================================================
# Inference Transforms — مطابقة 100% لـ val_transforms في الـ notebook
# لا يوجد Resize هنا لأن الصور في التدريب كانت بالفعل 380x380
# نحن نضيف Resize هنا فقط لاستيعاب أي صورة يرفعها المستخدم
# ======================================================================

INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((380, 380)),          # تحويل أي صورة للحجم المُدرَّب عليه
    transforms.ToTensor(),                  # [0,255] → [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],         # نفس القيم من notebook
        std=[0.229, 0.224, 0.225],
    ),
])


def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """
    تحويل صورة PIL إلى tensor جاهز للموديل.
    - تحويل لـ RGB (لضمان 3 channels حتى لو الصورة grayscale)
    - تطبيق INFERENCE_TRANSFORMS
    - إضافة batch dimension
    """
    rgb_image = pil_image.convert("RGB")
    tensor = INFERENCE_TRANSFORMS(rgb_image)
    return tensor.unsqueeze(0)  # shape: (1, 3, 380, 380)


def run_inference(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """
    تمرير الـ tensor عبر الموديل وإرجاع:
    - probabilities: dict {class_name: probability}
    - predictions: dict {class_name: True/False} بناءً على threshold
    """
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)               # raw output, shape: (1, 14)
        probs = torch.sigmoid(logits)        # تحويل logits → probabilities [0, 1]

    probs_np = probs.squeeze(0).cpu().numpy()  # shape: (14,)

    probabilities = {
        name: float(prob) for name, prob in zip(CLASS_NAMES, probs_np)
    }
    predictions = {
        name: float(prob) >= threshold for name, prob in probabilities.items()
    }

    return probabilities, predictions


def get_top_findings(
    probabilities: Dict[str, float],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """إرجاع أعلى top_k نتائج مرتبة تنازلياً."""
    sorted_findings = sorted(
        probabilities.items(), key=lambda x: x[1], reverse=True
    )
    return sorted_findings[:top_k]


def get_class_index(class_name: str) -> int:
    """إرجاع انديكس الكلاس بالاسم (مفيد لتوليد Grad-CAM لمرض معيّن)."""
    return CLASS_NAMES.index(class_name)
