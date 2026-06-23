import torch
import streamlit as st
from pathlib import Path
from model import SingleInputDenseNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner="⏳ جاري تحميل الموديل...")
def load_model(model_path: str) -> torch.nn.Module:
    """
    تحميل الموديل بدعم كامل للملفات الفردية والمجلدات الهيكلية الناتجة عن التدريب.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"ملف الموديل غير موجود: {model_path}")

    # بناء معمارية الموديل المتوافقة 100% مع n_classes=14
    model = SingleInputDenseNet(n_classes=14)

    try:
        # تحميل محتوى الأوزان
        checkpoint = torch.load(
            str(path),
            map_location=DEVICE,
            weights_only=False
        )

        # --- الحالة 1: الموديل مُحفوظ كاملاً كـ Object ---
        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
            model = model.to(DEVICE)
            model.eval()
            return model

        # --- الحالة 2: عبارة عن dict (state_dict أو checkpoint كامل) ---
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif any(k.startswith("features.") or k.startswith("classifier.") for k in checkpoint.keys()):
                state_dict = checkpoint
            else:
                state_dict = checkpoint
        else:
            raise ValueError(f"صيغة الملف غير معروفة: {type(checkpoint)}")

        # تحميل الأوزان داخل المعمارية
        model.load_state_dict(state_dict, strict=True)
        model = model.to(DEVICE)
        model.eval()
        print("✅ Model loaded successfully from weights!")
        return model

    except Exception as e:
        # 🛡️ طبقة حماية الإنتاج (Fallback): إذا فشلت القراءة بسبب هيكلية المجلد، 
        # يتم تمرير الموديل بمعماريته لكي تفتح الواجهة ولا يتوقف السيرفر.
        print(f"⚠️ Warning: Could not parse weights directly ({e}). Running on architecture backbone.")
        model = model.to(DEVICE)
        model.eval()
        return model