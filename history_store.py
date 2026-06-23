"""
history_store.py
-----------------
تسجيل كل عملية تشخيص (Inference) في ملف CSV محلي، لاستخدامها بعد ذلك
في صفحة Dashboard لعرض إحصائيات الاستخدام (عدد الفحوصات، توزيع الأمراض،
متوسط الثقة، نشاط كل مستخدم، إلخ).

⚠️ ملاحظة: هذا تخزين محلي بسيط (CSV) مناسب للتطبيق البحثي/التعليمي على نطاق
صغير. لاستخدام إنتاجي حقيقي مع عدد مستخدمين كبير، يُفضَّل استخدام قاعدة بيانات
حقيقية (SQLite/PostgreSQL) بدلاً من ملف CSV.
"""

import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

HISTORY_FILE = Path(__file__).parent / "data" / "history.csv"
HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

FIELDNAMES = [
    "timestamp",
    "username",
    "image_name",
    "threshold",
    "top_finding",
    "top_finding_prob",
    "positive_count",
    "all_probabilities",  # JSON string: {class: prob}
]


def log_inference(
    username: str,
    image_name: str,
    threshold: float,
    probabilities: Dict[str, float],
    positive_count: int,
) -> None:
    """يضيف سطر جديد في سجل التشخيصات."""
    import json

    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    top_finding, top_prob = sorted_probs[0] if sorted_probs else ("—", 0.0)

    file_exists = HISTORY_FILE.exists()
    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "username": username,
            "image_name": image_name,
            "threshold": threshold,
            "top_finding": top_finding,
            "top_finding_prob": round(top_prob, 4),
            "positive_count": positive_count,
            "all_probabilities": json.dumps(probabilities, ensure_ascii=False),
        })


def load_history() -> pd.DataFrame:
    """يحمّل كل السجل كـ DataFrame. يرجع DataFrame فاضي بالأعمدة الصحيحة لو الملف غير موجود."""
    if not HISTORY_FILE.exists():
        return pd.DataFrame(columns=FIELDNAMES)

    try:
        df = pd.read_csv(HISTORY_FILE)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame(columns=FIELDNAMES)


def load_history_for_user(username: str) -> pd.DataFrame:
    """سجل خاص بمستخدم واحد فقط (لو احتجنا فرز صلاحيات لاحقاً)."""
    df = load_history()
    if df.empty:
        return df
    return df[df["username"] == username]


def clear_history() -> None:
    """حذف كل السجل (لاستخدام إداري فقط)."""
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()
