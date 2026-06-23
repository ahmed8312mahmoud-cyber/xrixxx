"""
pages/1_📊_Dashboard.py
------------------------
لوحة تحكم (Dashboard) لعرض إحصائيات استخدام النظام:
- عدد الفحوصات الكلي ومتوسط الثقة
- توزيع الأمراض الأكثر اكتشافاً
- نشاط المستخدمين عبر الوقت
- آخر العمليات المسجّلة (Log)

البيانات مصدرها سجل history.csv الذي يُحدَّث تلقائياً من app.py
في كل مرة يتم فيها تشخيص صورة.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# للسماح باستيراد auth/history_store من جذر المشروع عند التشغيل كصفحة فرعية
sys.path.append(str(Path(__file__).parent.parent))

import auth
from history_store import load_history, clear_history

st.set_page_config(page_title="Dashboard · Chest X-Ray", page_icon="📊", layout="wide")

# ---------------- Auth Gate ----------------
auth.login_required()
user = auth.current_user()

st.markdown(
    '<div style="font-size:2rem; font-weight:700; color:#1f77b4; text-align:center;">'
    '📊 لوحة تحكم الاستخدام</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="text-align:center; color:#777; margin-bottom:1.5rem;">'
    'إحصائيات الفحوصات والنتائج المسجَّلة على هذا النظام</div>',
    unsafe_allow_html=True,
)
st.divider()

with st.sidebar:
    st.markdown(f"### 👋 {user['full_name']}")
    if st.button("🚪 تسجيل الخروج", use_container_width=True):
        auth.logout()
        st.rerun()
    st.divider()
    only_mine = st.checkbox("👤 عرض فحوصاتي فقط", value=False)
    if user.get("role") == "admin":
        st.divider()
        st.markdown("**🛠️ أدوات إدارية**")
        if st.button("🗑️ حذف كل السجل", type="secondary", use_container_width=True):
            clear_history()
            st.success("تم حذف السجل بالكامل.")
            st.rerun()

# ---------------- تحميل البيانات ----------------
df = load_history()

if only_mine and not df.empty:
    df = df[df["username"] == user["username"]]

if df.empty:
    st.info("لا يوجد أي فحوصات مسجّلة حتى الآن. ارفع صورة من الصفحة الرئيسية لتبدأ تظهر البيانات هنا.")
    st.stop()

# ---------------- KPIs ----------------
total_scans = len(df)
unique_users = df["username"].nunique()
avg_confidence = df["top_finding_prob"].mean() if "top_finding_prob" in df else 0.0
avg_positive = df["positive_count"].mean() if "positive_count" in df else 0.0
most_common_finding = (
    df["top_finding"].mode().iloc[0] if not df["top_finding"].empty else "—"
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("🔬 إجمالي الفحوصات", total_scans)
k2.metric("👥 عدد المستخدمين", unique_users)
k3.metric("📈 متوسط ثقة أعلى نتيجة", f"{avg_confidence*100:.1f}%")
k4.metric("🩺 متوسط عدد النتائج الموجبة", f"{avg_positive:.1f}")
k5.metric("🏆 المرض الأكثر تكراراً", most_common_finding)

st.divider()

chart_col1, chart_col2 = st.columns(2, gap="large")

# ---------------- توزيع الأمراض الأكثر اكتشافاً ----------------
with chart_col1:
    st.subheader("🩻 توزيع أكثر النتائج (Top Finding)")
    finding_counts = df["top_finding"].value_counts().sort_values(ascending=True)
    st.bar_chart(finding_counts, height=320)

# ---------------- متوسط احتمالية كل مرض عبر كل الفحوصات ----------------
with chart_col2:
    st.subheader("📈 متوسط احتمالية كل مرض (كل الفحوصات)")
    try:
        parsed = df["all_probabilities"].dropna().apply(json.loads)
        probs_df = pd.DataFrame(list(parsed))
        avg_probs = probs_df.mean().sort_values(ascending=True)
        st.bar_chart(avg_probs, height=320)
    except Exception:
        st.caption("تعذّر تحليل بيانات الاحتمالات التفصيلية لبعض السجلات القديمة.")

st.divider()

# ---------------- النشاط عبر الوقت ----------------
st.subheader("🗓️ عدد الفحوصات عبر الوقت")
if "timestamp" in df.columns and df["timestamp"].notna().any():
    daily_counts = (
        df.dropna(subset=["timestamp"])
        .set_index("timestamp")
        .resample("D")
        .size()
    )
    st.line_chart(daily_counts, height=260)
else:
    st.caption("لا توجد بيانات زمنية كافية لعرض الرسم.")

st.divider()

# ---------------- نشاط حسب المستخدم ----------------
st.subheader("👥 عدد الفحوصات حسب المستخدم")
user_counts = df["username"].value_counts()
st.bar_chart(user_counts, height=260)

st.divider()

# ---------------- جدول السجل التفصيلي ----------------
st.subheader("📋 آخر العمليات المسجّلة")
display_cols = ["timestamp", "username", "image_name", "threshold", "top_finding", "top_finding_prob", "positive_count"]
display_cols = [c for c in display_cols if c in df.columns]
st.dataframe(
    df[display_cols].sort_values("timestamp", ascending=False),
    use_container_width=True,
    height=350,
)

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ تحميل السجل الكامل (CSV)",
    data=csv_bytes,
    file_name="diagnosis_history.csv",
    mime="text/csv",
)
