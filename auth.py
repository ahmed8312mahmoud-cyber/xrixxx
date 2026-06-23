"""
auth.py
-------
نظام تسجيل دخول / إنشاء حساب محلي بسيط.

⚠️ ملاحظة مهمة:
هذا نظام Auth "محلي" مناسب لتطبيق بحثي/تعليمي يعمل على جهاز واحد أو سيرفر داخلي.
كلمات المرور يتم تخزينها مُجزَّأة (hash) عبر SHA-256 + Salt عشوائي لكل مستخدم،
لكنه ليس بديلاً عن نظام Auth حقيقي (مثل OAuth / Auth0 / قاعدة بيانات مع HTTPS
و rate-limiting) إذا كان التطبيق سيُنشر للعامة أو سيتعامل مع بيانات مرضى حقيقية.
"""

import json
import hashlib
import hmac
import secrets
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import streamlit as st

USERS_FILE = Path(__file__).parent / "data" / "users.json"
USERS_FILE.parent.mkdir(parents=True, exist_ok=True)

USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,20}$")


# ======================================================================
# تخزين المستخدمين (JSON بسيط بدل قاعدة بيانات)
# ======================================================================

def _load_users() -> dict:
    if not USERS_FILE.exists():
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_users(users: dict) -> None:
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def _hash_password(password: str, salt: str) -> str:
    """PBKDF2-HMAC-SHA256، أقوى بكثير من SHA-256 العادي لتخزين كلمات السر."""
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 100_000)
    return dk.hex()


def _verify_password(password: str, salt: str, stored_hash: str) -> bool:
    candidate = _hash_password(password, salt)
    return hmac.compare_digest(candidate, stored_hash)


# ======================================================================
# عمليات المستخدمين
# ======================================================================

def register_user(username: str, password: str, full_name: str = "") -> Tuple[bool, str]:
    """تسجيل مستخدم جديد. يرجع (success, message)."""
    username = username.strip()

    if not USERNAME_RE.match(username):
        return False, "اسم المستخدم يجب أن يكون 3-20 حرف (حروف/أرقام/underscore فقط)."

    if len(password) < 6:
        return False, "كلمة المرور يجب أن تكون 6 أحرف على الأقل."

    users = _load_users()
    if username.lower() in users:
        return False, "اسم المستخدم موجود بالفعل، اختر اسماً آخر."

    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)

    users[username.lower()] = {
        "username": username,
        "full_name": full_name.strip() or username,
        "salt": salt,
        "password_hash": password_hash,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "role": "user",
    }
    _save_users(users)
    return True, "تم إنشاء الحساب بنجاح! يمكنك تسجيل الدخول الآن."


def authenticate_user(username: str, password: str) -> Tuple[bool, str, Optional[dict]]:
    """التحقق من بيانات الدخول. يرجع (success, message, user_dict_or_None)."""
    username = username.strip().lower()
    users = _load_users()

    user = users.get(username)
    if user is None:
        return False, "اسم المستخدم غير موجود.", None

    if not _verify_password(password, user["salt"], user["password_hash"]):
        return False, "كلمة المرور غير صحيحة.", None

    return True, "تم تسجيل الدخول بنجاح!", user


def ensure_default_admin() -> None:
    """ينشئ حساب admin افتراضي (admin / admin123) أول مرة فقط، لتسهيل أول دخول."""
    users = _load_users()
    if not users:
        salt = secrets.token_hex(16)
        users["admin"] = {
            "username": "admin",
            "full_name": "Administrator",
            "salt": salt,
            "password_hash": _hash_password("admin123", salt),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "role": "admin",
        }
        _save_users(users)


# ======================================================================
# دوال Session State + الواجهة
# ======================================================================

def is_logged_in() -> bool:
    return bool(st.session_state.get("auth_user"))


def current_user() -> Optional[dict]:
    return st.session_state.get("auth_user")


def logout() -> None:
    for key in ("auth_user",):
        st.session_state.pop(key, None)


def login_required(stop_app: bool = True) -> bool:
    """
    تستخدم في بداية أي صفحة (app.py أو ملفات pages/) للتأكد من تسجيل الدخول.
    لو المستخدم مش مسجل دخول، تعرض صفحة تسجيل الدخول/التسجيل وتوقف باقي الصفحة.
    """
    ensure_default_admin()

    if is_logged_in():
        return True

    render_login_page()
    if stop_app:
        st.stop()
    return False


def render_login_page() -> None:
    """صفحة تسجيل الدخول / إنشاء حساب جديد بتصميم Tabs."""
    st.markdown(
        """
        <div style="text-align:center; margin-top: 2rem;">
            <div style="font-size:2.4rem;">🫁</div>
            <div style="font-size:1.6rem; font-weight:700; color:#1f77b4;">
                Chest X-Ray Classifier
            </div>
            <div style="color:#777; margin-bottom:1.5rem;">
                سجّل الدخول للوصول إلى أداة التشخيص المساعد بالأشعة
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, center_col, _ = st.columns([1, 1.2, 1])
    with center_col:
        tab_login, tab_register = st.tabs(["🔑 تسجيل الدخول", "🆕 حساب جديد"])

        # ---------------- تسجيل الدخول ----------------
        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("اسم المستخدم", key="login_username")
                password = st.text_input("كلمة المرور", type="password", key="login_password")
                submitted = st.form_submit_button("تسجيل الدخول", use_container_width=True)

            if submitted:
                if not username or not password:
                    st.error("من فضلك أدخل اسم المستخدم وكلمة المرور.")
                else:
                    success, message, user = authenticate_user(username, password)
                    if success:
                        st.session_state["auth_user"] = user
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

            st.caption("👤 أول مرة؟ تقدر تستخدم: `admin` / `admin123` أو تسجّل حساب جديد من التاب التاني.")

        # ---------------- حساب جديد ----------------
        with tab_register:
            with st.form("register_form", clear_on_submit=True):
                new_full_name = st.text_input("الاسم بالكامل (اختياري)", key="reg_full_name")
                new_username = st.text_input("اسم المستخدم (3-20 حرف، إنجليزي)", key="reg_username")
                new_password = st.text_input("كلمة المرور (6 أحرف على الأقل)", type="password", key="reg_password")
                new_password_confirm = st.text_input("تأكيد كلمة المرور", type="password", key="reg_password_confirm")
                reg_submitted = st.form_submit_button("إنشاء الحساب", use_container_width=True)

            if reg_submitted:
                if new_password != new_password_confirm:
                    st.error("كلمتا المرور غير متطابقتين.")
                else:
                    success, message = register_user(new_username, new_password, new_full_name)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
