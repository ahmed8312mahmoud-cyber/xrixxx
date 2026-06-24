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
    page_title="xraix — Chest AI",
    page_icon="🫁",
    layout="wide",
)

# ======================================================================
# CSS مخصص — تصميم xraix الاحترافي
# ======================================================================
st.markdown("""
<style>
    /* ── الخطوط والألوان الأساسية ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── إخفاء عناصر Streamlit الافتراضية ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 1rem !important;
        max-width: 1200px;
    }

    /* ── Sidebar احترافي ── */
    [data-testid="stSidebar"] {
        background: #0B1C2E !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }
    [data-testid="stSidebar"] * {
        color: rgba(255,255,255,0.75) !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: rgba(255,255,255,0.6) !important;
        font-size: 0.82rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: rgba(255,255,255,0.55) !important;
        font-size: 0.82rem;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08) !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: rgba(255,255,255,0.6) !important;
        border-radius: 8px !important;
        font-size: 0.82rem !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.12) !important;
        color: #fff !important;
    }

    /* ── شعار xraix في الـ Sidebar ── */
    .xraix-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 0 16px;
        border-bottom: 1px solid rgba(255,255,255,0.07);
        margin-bottom: 16px;
    }
    .xraix-brand-icon {
        width: 36px;
        height: 36px;
        border-radius: 9px;
        background: linear-gradient(135deg, #1e6fa8, #00c6a0);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
    }
    .xraix-brand-text .name {
        font-size: 1.15rem;
        font-weight: 600;
        color: #ffffff !important;
        letter-spacing: 0.5px;
        line-height: 1.1;
    }
    .xraix-brand-text .tag {
        font-size: 0.68rem;
        color: rgba(255,255,255,0.35) !important;
        letter-spacing: 1.2px;
        text-transform: uppercase;
    }

    /* ── Header الرئيسي ── */
    .xraix-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 20px;
        background: #ffffff;
        border: 1px solid #e8eaed;
        border-radius: 12px;
        margin-bottom: 1.2rem;
    }
    .xraix-header-left {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .xraix-header-icon {
        width: 38px;
        height: 38px;
        border-radius: 9px;
        background: linear-gradient(135deg, #1e6fa8, #00c6a0);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
    }
    .xraix-header-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #0B1C2E;
    }
    .xraix-header-sub {
        font-size: 0.78rem;
        color: #6b7280;
        margin-top: 1px;
    }
    .xraix-badge {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1D4ED8;
        font-size: 0.72rem;
        padding: 4px 10px;
        border-radius: 99px;
        font-weight: 500;
    }

    /* ── بطاقات Section ── */
    .xraix-card {
        background: #ffffff;
        border: 1px solid #e8eaed;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 1rem;
    }
    .xraix-card-title {
        font-size: 0.88rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 7px;
        border-bottom: 1px solid #f3f4f6;
        padding-bottom: 10px;
    }

    /* ── Findings ── */
    .finding-positive {
        background: #FEF2F2;
        border-left: 3px solid #EF4444;
        padding: 7px 12px;
        border-radius: 0 8px 8px 0;
        margin: 4px 0;
        font-weight: 500;
        color: #991B1B;
        font-size: 0.84rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .finding-negative {
        background: #F9FAFB;
        border-left: 3px solid #D1D5DB;
        padding: 7px 12px;
        border-radius: 0 8px 8px 0;
        margin: 4px 0;
        color: #6B7280;
        font-size: 0.84rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* ── Disclaimer ── */
    .disclaimer {
        background: #FFFBEB;
        border: 1px solid #FCD34D;
        border-radius: 10px;
        padding: 10px 16px;
        font-size: 0.8rem;
        color: #92400E;
        margin-top: 1.2rem;
        display: flex;
        align-items: flex-start;
        gap: 8px;
    }

    /* ── Section Headers (بدل subheader الافتراضي) ── */
    .section-header {
        font-size: 0.88rem;
        font-weight: 600;
        color: #1e3a5f;
        margin-bottom: 10px;
        padding: 8px 12px;
        background: #F0F7FF;
        border-radius: 8px;
        border-left: 3px solid #1e6fa8;
    }

    /* ── Metric cards تحسين ── */
    [data-testid="metric-container"] {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 10px !important;
    }
    [data-testid="metric-container"] label {
        font-size: 0.75rem !important;
        color: #64748B !important;
        font-weight: 500 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        color: #0B1C2E !important;
        font-weight: 600 !important;
    }

    /* ── Divider ── */
    hr {
        border-color: #F1F5F9 !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #1e6fa8 !important;
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
# العنوان الرئيسي — xraix Brand Header
# ======================================================================
st.markdown("""
<div class="xraix-header">
    <div class="xraix-header-left">
        <div class="xraix-header-icon">🫁</div>
        <div>
            <div class="xraix-header-title">xraix</div>
            <div class="xraix-header-sub">Chest X-Ray AI &nbsp;·&nbsp; DenseNet121 &nbsp;·&nbsp; 14-Class Detection</div>
        </div>
    </div>
    <div style="display:flex; gap:8px; align-items:center;">
        <span class="xraix-badge">NIH Dataset</span>
        <span class="xraix-badge">Multi-Label</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ======================================================================
# Sidebar — إعدادات + معلومات المستخدم
# ======================================================================
with st.sidebar:
    st.markdown(f"""
    <div class="xraix-brand">
        <div class="xraix-brand-icon">🫁</div>
        <div class="xraix-brand-text">
            <div class="name">xraix</div>
            <div class="tag">Chest AI Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"**👤 {user['full_name']}**")
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
    st.markdown('<div class="section-header">📤 رفع صورة الأشعة</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="section-header">📊 نتائج التشخيص</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="section-header">🔥 Grad-CAM — أين نظر الموديل؟</div>', unsafe_allow_html=True)

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
    ⚠️ <div><strong>تنبيه طبي:</strong> هذا النظام مخصص للأغراض البحثية والتعليمية فقط.
    لا يُستخدم كبديل عن التشخيص الطبي المتخصص. استشر طبيبك دائماً.</div>
</div>
""", unsafe_allow_html=True)