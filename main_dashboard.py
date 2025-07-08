import os
import sys
import types
import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore
import requests
import time
import json
# ─── 1) Page Config & THEME ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Growify AI Toolkit",
    page_icon="🔒",
    layout="wide",
)

# Optional hook for your theme.py
try:
    from theme import apply_dark_theme
    apply_dark_theme()
except ImportError:
    pass

# ─── 2) Custom CSS for Login Card ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: white;
    }
    .login-card {
      background-color: transparent;
      padding: 2rem;
      border-radius: 1rem;
      box-shadow: 0 4px 10px rgba(0,0,0,0.1);
      max-width: 400px;
      margin: 5% auto;
      text-align: center;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
    }
    .login-card h2 {
      margin-top: 1rem;
      font-size: 1.5rem;
      font-weight: bold;
      color: #fff;
    }
    .login-card img {
      width: 150px;
      margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

# ─── 3) Firebase Initialization ──────────────────────────────────────────────
firebase_config = {
    "apiKey":            os.getenv("FIREBASE_API_KEY"),
    "authDomain":        os.getenv("FIREBASE_AUTH_DOMAIN"),
    "databaseURL":       os.getenv("FIREBASE_DATABASE_URL"),
    "projectId":         os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket":     os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId":             os.getenv("FIREBASE_APP_ID"),
}

try:
    pb = pyrebase.initialize_app(firebase_config)
    firebase_auth = pb.auth()
except Exception as e:
    st.error(f"❌ Firebase init error: {e}")
    st.stop()

if not firebase_admin._apps:
    service_account_json = os.getenv("FIREBASE_ADMIN_JSON")
    if not service_account_json:
        st.error("❌ FIREBASE_ADMIN_JSON environment variable is not set!")
        st.stop()

    try:
        # Load JSON
        parsed = json.loads(service_account_json)

        # FIX: Replace literal \\n with real \n in private_key
        parsed["private_key"] = parsed["private_key"].replace("\\n", "\n")

        cred = credentials.Certificate(parsed)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"❌ Failed to initialize Firebase Admin SDK: {e}")
        st.stop()


db = firestore.client()

# ─── 4) Firebase Session Logging Helpers ─────────────────────────────────────
def log_user_login(user_id, email):
    user_ref = db.collection('user_activity').document(user_id)
    user_ref.set({'email': email}, merge=True)

    session_ref = user_ref.collection('sessions').document()
    session_ref.set({
        'login_time': firestore.SERVER_TIMESTAMP,
        'logout_time': None,
        'session_duration': None,
    })

    st.session_state.session_doc_id = session_ref.id
    st.session_state.login_time = time.time()

def log_user_logout(user_id):
    logout_time = time.time()
    session_doc_id = st.session_state.get('session_doc_id')
    if not session_doc_id:
        st.warning("⚠️ Session ID not found. Unable to log logout time.")
        return

    session_ref = db.collection('user_activity').document(user_id).collection('sessions').document(session_doc_id)
    session_duration = logout_time - st.session_state.login_time

    session_ref.update({
        'logout_time': firestore.SERVER_TIMESTAMP,
        'session_duration': session_duration
    })

# ─── 5) Login UI ─────────────────────────────────────────────────────────────
def do_login():
    st.markdown("<h2>Company Login Portal</h2>", unsafe_allow_html=True)

    with st.form("login_form"):
        email = st.text_input("Email")
        pwd   = st.text_input("Password", type="password")
        submit = st.form_submit_button("Log In")
        if submit:
            try:
                user = firebase_auth.sign_in_with_email_and_password(email.strip(), pwd)
                if user:
                    st.session_state.user = user
                    st.session_state.logged_in = True
                    user_id = user['localId']
                    log_user_login(user_id, email)
                    st.rerun()
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Network error: {e}")
            except firebase_admin.exceptions.FirebaseError as e:
                st.error(f"❌ Firebase error: {e}")
            except Exception as e:
                st.error(f"❌ Login failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# ─── 6) Logout UI ─────────────────────────────────────────────────────────────
def do_logout():
    with st.sidebar:
        if st.button("Log out"):
            user_id = st.session_state.user['localId']
            log_user_logout(user_id)
            st.session_state.clear()
            st.rerun()

# ─── 7) Main App Flow ────────────────────────────────────────────────────────
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    do_login()
    st.stop()
else:
    do_logout()

    # Torch Patch
    class DummyModule(types.ModuleType):
        __path__ = []
    sys.modules["torch.classes"] = DummyModule("torch.classes")

    # Add Project Paths
    base_dir = os.path.dirname(__file__)
    projects = {
        "blog": os.path.join(base_dir, "GF_BLOG_IDEA"),
        "img": os.path.join(base_dir, "GF_IMG_RE"),
        "seo": os.path.join(base_dir, "GF_SEO"),
        "scraper": os.path.join(base_dir, "GF_SCRAPERS"),
        "pixelmatch": os.path.join(base_dir, "GF_IMG_FIND"),
    }
    for p in projects.values():
        if p not in sys.path:
            sys.path.insert(0, p)

    # Import Tool Modules
    try:
        from GF_BLOG_IDEA import blog_ideas as blog_module
    except ImportError:
        blog_module = None

    try:
        from GF_IMG_RE import app as img_module
    except ImportError:
        img_module = None

    try:
        from GF_SEO import app as seo_module
    except ImportError:
        seo_module = None

    try:
        from GF_SCRAPERS import app as scraper_module
    except ImportError:
        scraper_module = None

    try:
        from GF_IMG_FIND import app as pixelmatch_module
    except ImportError:
        pixelmatch_module = None

    # Session State Tool Tracker
    if "tool" not in st.session_state:
        st.session_state.tool = "home"

    if st.session_state.tool != "home":
        with st.sidebar:
            if st.button("Back to Home"):
                st.session_state.tool = "home"
                st.rerun()

    # ─── Home Screen ──────────────────────────────────────────────────────────
    def render_home():
        st.title("🧰 Growify Master AI Toolkit")
        st.markdown("#### 👇 Select a tool to get started")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("Promptly", use_container_width=True, on_click=lambda: st.session_state.update(tool="blog"))
        with col2:
            st.button("Snipster", use_container_width=True, on_click=lambda: st.session_state.update(tool="img"))
        with col3:
            st.button("MetaScan", use_container_width=True, on_click=lambda: st.session_state.update(tool="seo"))

        col4, col5 = st.columns(2)
        with col4:
            st.button("PixelMatch", use_container_width=True, on_click=lambda: st.session_state.update(tool="pixelmatch"))
        with col5:
            st.button("PitchKit", use_container_width=True, on_click=lambda: st.session_state.update(tool="scraper"))

    # ─── Routing ─────────────────────────────────────────────────────────────
    t = st.session_state.tool
    if t == "home":
        render_home()
    elif t == "blog":
        if blog_module: blog_module.main()
        else: st.error("⚠️ Blog Idea module not found.")
    elif t == "img":
        if img_module: img_module.main()
        else: st.error("⚠️ Image Resizer module not found.")
    elif t == "seo":
        if seo_module: seo_module.main()
        else: st.error("⚠️ SEO module not found.")
    elif t == "scraper":
        if scraper_module: scraper_module.main()
        else: st.error("⚠️ Scraper module not found.")
    elif t == "pixelmatch":
        if pixelmatch_module: pixelmatch_module.main()
        else: st.error("⚠️ PixelMatch module not found.")