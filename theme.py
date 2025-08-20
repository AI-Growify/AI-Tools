import streamlit as st

def apply_dark_theme():
    st.markdown("""
    <style>
    :root{
        --app-bg: #0e1117;
        --panel-bg: #0b0f14;
        --text-color: #e6eef8;
        --muted: #9aa9c7;
        --btn-shadow: rgba(2,12,41,0.30);
        --overlay-opacity: 0.14;
        --sheen-opacity: 0.18;
    }


    /* Background with dark overlay */
    html, body, .stApp {
    background: linear-gradient(135deg, #0e1117, #0a0d12, #101520);
    background-size: 400% 400%;
    animation: bgShift 18s ease infinite;
    color: var(--text-color) !important;
}

    @keyframes bgShift {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }



    /* ---------- Primary button styling ---------- */
    .stButton > button {
        position: relative;
        overflow: hidden;
        border: none !important;
        padding: 0.56rem 1.08rem !important;
        font-weight: 800 !important;
        font-size: 0.98rem !important;
        border-radius: 12px !important;
        color: white !important;
        cursor: pointer !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        letter-spacing: 0.2px;
        background-image: linear-gradient(90deg,#00D1FF 0%,#0091FF 35%,#5A00FF 75%) !important;
        background-size: 200% 100% !important;
        background-position: 0% 0% !important;
        box-shadow: 0 10px 28px var(--btn-shadow) !important;
        transition: transform 180ms ease, box-shadow 180ms ease, background-position 420ms cubic-bezier(.2,.8,.2,1) !important;
    }
    .stButton > button::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(rgba(0,0,0,var(--overlay-opacity)), rgba(0,0,0,calc(var(--overlay-opacity) - 0.04)));
        z-index: 0;
        pointer-events: none;
    }
    .stButton > button::after {
        content: "";
        position: absolute;
        left: -40%;
        top: -40%;
        width: 180%;
        height: 120%;
        background: linear-gradient(120deg,rgba(255,255,255,var(--sheen-opacity)) 0%,rgba(255,255,255,0.02) 50%,rgba(255,255,255,0.0) 60%);
        transform: rotate(-18deg) translateX(0);
        transition: transform 650ms cubic-bezier(.2,.8,.2,1), opacity 240ms ease;
        opacity: var(--sheen-opacity);
        z-index: 1;
        pointer-events: none;
        mix-blend-mode: overlay;
    }
    .stButton > button span {
        position: relative;
        z-index: 2;
        color: #ffffff !important;
        text-shadow: 0 1px 1px rgba(0,0,0,0.68);
        font-weight: 800 !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        background-position: 100% 0% !important;
        box-shadow: 0 18px 42px rgba(2,12,41,0.60) !important;
    }
    .stButton > button:hover::after{
        transform: rotate(-18deg) translateX(18%);
        opacity: calc(var(--sheen-opacity) + 0.04);
    }
    .stButton > button:active {
        transform: translateY(0) scale(0.998) !important;
        box-shadow: 0 6px 18px rgba(2,12,41,0.60) !important;
    }
    .stButton > button[disabled] {
        opacity: 0.5 !important;
        background-image: linear-gradient(90deg, #23262b 0%, #181b20 100%) !important;
    }

    /* ---------- Sidebar styling ---------- */
    [data-testid="stSidebar"] {
        min-width: 240px !important;
        max-width: 340px !important;
        background: linear-gradient(180deg, #0b0f14 0%, #0d1116 58%, #0f1419 100%) !important;
        box-shadow: 10px 0 30px rgba(2,8,18,0.68), inset -1px 0 0 rgba(255,255,255,0.02) !important;
        border-right: 1px solid rgba(255,255,255,0.03) !important;
        padding: 1rem 1.25rem !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background-image: linear-gradient(90deg, #00D1FF 0%, #0091FF 42%, #5A00FF 100%) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        padding: 0.48rem 0.92rem !important;
        width: 94% !important;
    }

    /* ---------- Responsive layout fix when sidebar is collapsed ---------- */
    [data-testid="stSidebar"][aria-expanded="false"] ~ div [data-testid="stAppViewContainer"] {
        margin-left: 0 !important;
        width: 100% !important;
    }
    [data-testid="stAppViewContainer"] {
        transition: margin-left 0.3s ease-in-out, width 0.3s ease-in-out;
    }
    </style>
    """, unsafe_allow_html=True)
