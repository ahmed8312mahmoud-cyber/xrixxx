"""
app.py
------
الواجهة الرئيسية للتطبيق - Chest X-Ray Multi-Label Classification
يعمل على 14 مرض من NIH Chest X-Ray Dataset باستخدام DenseNet121

يشمل: تشخيص + Grad-CAM Heatmap، وتسجيل كل عملية في سجل
يُستخدم بعدها في صفحة Dashboard. (تم تخطي الحماية للدخول المباشر)
"""

import streamlit as st
from PIL import Image
import torch
from pathlib import Path

from loader import load_model, DEVICE
from utils import preprocess_image, run_inference, get_top_findings, get_class_index, CLASS_NAMES
from gradcam import generate_gradcam_overlay
from history_store import log_inference
import auth

# ======================================================================
# إعدادات الصفحة
# ======================================================================
st.set_page_config(
    page_title="Chest X-Ray Classifier",
    page_icon="🫁",
    layout="wide",
)

# ======================================================================
# CSS مخصص
# ======================================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .finding-positive {
        background-color: #ffeaea;
        border-left: 4px solid #e74c3c;
        padding: 6px 12px;
        border-radius: 4px;
        margin: 4px 0;
        font-weight: 600;
        color: #c0392b;
    }
    .finding-negative {
        background-color: #f0f0f0;
        border-left: 4px solid #bbb;
        padding: 6px 12px;
        border-radius: 4px;
        margin: 4px 0;
        color: #666;
    }
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 6px;
        padding: 10px 16px;
        font-size: 0.85rem;
        color: #856404;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================================
# 🔐 Auth Gate — تم التخطى للدخول المباشر
# ======================================================================
# auth.login_required()  👈 تم إيقاف السطر لمنع إجبار تسجيل الدخول

# إنشاء مستخدم افتراضي بصلاحيات كاملة لتفادي انهيار الـ Sidebar وسجل العمليات
user = {
    "full_name": "المطور الافتراضي",
    "role": "admin",
    "username": "admin_developer"
}

# ======================================================================
# العنوان
# ======================================================================
st.markdown('<div class="main-title">🫁 Chest X-Ray Disease Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">NIH Chest X-Ray · DenseNet121 · 14-Class Multi-Label Detection</div>',
    unsafe_allow_html=True,
)
st.divider()

# ======================================================================
# Sidebar — إعدادات + معلومات المستخدم
# ======================================================================
with st.sidebar:
    st.markdown(f"### 👋 أهلاً، {user['full_name']}")
    st.caption(f"الدور: `{user.get('role', 'user')}`")
    
    # تحويل الزر ليتوافق مع التحديث الجديد
    if st.button("🚪 تسجيل الخروج", width='stretch'):
        auth.logout()
        st.rerun()

    st.divider()
    st.header("⚙️ الإعدادات")

    model_path = st.text_input(
        "مسار ملف الموديل (.pt / .pth)",
        value="densenet121_final_baseline.pt",
        help="ضع ملف الموديل بجانب app.py وأدخل اسمه هنا",
    )

    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="الاحتمالية فوق هذا الحد تُعتبر 'موجودة'",
    )

    top_k = st.slider(
        "عدد النتائج المعروضة (Top-K)",
        min_value=3,
        max_value=14,
        value=5,
    )

    st.divider()
    show_heatmap = st.checkbox("🔥 عرض Grad-CAM Heatmap", value=True)
    heatmap_alpha = st.slider(
        "شفافية الـ Heatmap", min_value=0.1, max_value=0.9, value=0.45, step=0.05,
        disabled=not show_heatmap,
    )

    st.divider()
    st.markdown(f"**Device:** `{DEVICE}`")
    st.markdown(f"**Classes:** `{len(CLASS_NAMES)}`")
    st.markdown(f"**Input Size:** `380 × 380`")

# ======================================================================
# تحميل الموديل
# ======================================================================
model = None
if Path(model_path).exists():
    try:
        model = load_model(model_path)
        st.sidebar.success(f"✅ الموديل محمّل بنجاح")
    except Exception as e:
        st.sidebar.error(f"❌ خطأ في تحميل الموديل:\n{e}")
else:
    st.sidebar.warning(f"⚠️ الملف غير موجود:\n`{model_path}`")

# ======================================================================
# Main Layout
# ======================================================================
col_upload, col_result = st.columns([1, 1], gap="large")

# ---------- العمود الأيسر: رفع الصورة ----------
with col_upload:
    st.subheader("📤 رفع صورة الأشعة")

    uploaded_file = st.file_uploader(
        "اختر صورة Chest X-Ray",
        type=["jpg", "jpeg", "png"],
        help="يُفضَّل صور PA view بدقة 380×380 أو أعلى",
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        st.image(
            pil_image,
            caption=f"📁 {uploaded_file.name}  |  {pil_image.size[0]}×{pil_image.size[1]} px",
            width='stretch',  # 👈 تعديل التوافق مع التحديث الجديد
        )
        st.caption(f"Mode: `{pil_image.mode}` → سيتم تحويلها إلى RGB تلقائياً")

# ---------- العمود الأيمن: النتائج ----------
with col_result:
    st.subheader("📊 نتائج التشخيص")

    if uploaded_file is None:
        st.info("ارفع صورة أشعة على اليسار لبدء التحليل.")

    elif model is None:
        st.error("الموديل غير محمّل. تحقق من مسار الملف في الـ Sidebar.")

    else:
        with st.spinner("🔬 جاري تحليل الصورة..."):
            try:
                # Preprocessing + Inference
                tensor = preprocess_image(pil_image)
                probabilities, predictions = run_inference(
                    model, tensor, DEVICE, threshold=threshold
                )
                top_findings = get_top_findings(probabilities, top_k=top_k)

                # --- ملخص سريع ---
                positive_count = sum(1 for v in predictions.values() if v)
                m1, m2, m3 = st.columns(3)
                m1.metric("Findings", positive_count, delta=None)
                m2.metric("Threshold", f"{threshold:.0%}")
                m3.metric("Top Finding", top_findings[0][0] if top_findings else "—")

                st.divider()

                # --- Top-K bar chart ---
                st.markdown(f"**Top {top_k} Probabilities**")
                chart_data = {name: prob for name, prob in top_findings}
                st.bar_chart(chart_data, height=220)

                # --- قائمة Findings ---
                st.markdown("**جميع النتائج (Threshold = {:.0%}):**".format(threshold))
                for name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                    detected = predictions[name]
                    pct = prob * 100
                    if detected:
                        st.markdown(
                            f'<div class="finding-positive">🔴 {name}: {pct:.1f}%</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="finding-negative">⚪ {name}: {pct:.1f}%</div>',
                            unsafe_allow_html=True,
                        )

                # --- تسجيل العملية في الـ History ---
                log_inference(
                    username=user["username"],
                    image_name=uploaded_file.name,
                    threshold=threshold,
                    probabilities=probabilities,
                    positive_count=positive_count,
                )

            except Exception as e:
                st.error(f"❌ خطأ أثناء الـ Inference:\n```\n{e}\n```")
                probabilities = None

# ======================================================================
# Grad-CAM Heatmap (تحت العمودين، بعرض الصفحة كاملة)
# ======================================================================
if (
    uploaded_file is not None
    and model is not None
    and show_heatmap
    and "probabilities" in dir()
    and probabilities is not None
):
    st.divider()
    st.subheader("🔥 Grad-CAM — أين نظر الموديل؟")

    cam_class = st.selectbox(
        "اختر المرض لعرض الـ Heatmap الخاص به",
        options=[name for name, _ in get_top_findings(probabilities, top_k=len(CLASS_NAMES))],
        index=0,
        help="الـ Heatmap يوضّح المناطق في الصورة التي أثّرت أكثر على قرار الموديل لهذا المرض",
    )

    try:
        with st.spinner("🧠 جاري توليد Grad-CAM..."):
            class_idx = get_class_index(cam_class)
            overlay_image = generate_gradcam_overlay(
                model=model,
                input_tensor=tensor,
                original_image=pil_image,
                class_idx=class_idx,
                alpha=heatmap_alpha,
            )

        cam_col1, cam_col2 = st.columns(2)
        with cam_col1:
            st.image(pil_image, caption="الصورة الأصلية", width='stretch')  # 👈 تعديل التوافق
        with cam_col2:
            st.image(
                overlay_image,
                caption=f"Grad-CAM — {cam_class} ({probabilities[cam_class]*100:.1f}%)",
                width='stretch',  # 👈 تعديل التوافق
            )

        st.caption(
            "🔴 المناطق الحمراء/الصفراء = أكثر تأثيراً على القرار · "
            "🔵 المناطق الزرقاء/الباردة = أقل تأثيراً."
        )

    except Exception as e:
        st.warning(f"⚠️ تعذّر توليد Grad-CAM:\n```\n{e}\n```")

# ======================================================================
# Disclaimer
# ======================================================================
st.markdown("""
<div class="disclaimer">
⚠️ <strong>تنبيه طبي:</strong> هذا النظام مخصص للأغراض البحثية والتعليمية فقط.
لا يُستخدم كبديل عن التشخيص الطبي المتخصص. استشر طبيبك دائماً.
</div>
""", unsafe_allow_html=True)