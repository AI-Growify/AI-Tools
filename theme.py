import streamlit as st

def apply_dark_theme():
    st.markdown("""
    <style>
        html, body, .stApp {
            background-color: #0e1117 !important;
            color: white !important;
        }

        .stButton > button {
            background-color: #1F6FEB !important;
            color: white !important;
}
        .stButton > button:hover {
            background-color: #58A6FF !important;
        color: white !important;
}
    </style>
    """, unsafe_allow_html=True)
