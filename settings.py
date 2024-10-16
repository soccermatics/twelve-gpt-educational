import os
import streamlit as st

ENGINE_ADA = st.secrets.get("ENGINE_ADA")
GPT_DEFAULT = "3.5"
GPT3_BASE = st.secrets.get("GPT_BASE")
GPT3_VERSION = st.secrets.get("GPT_VERSION")
GPT3_KEY = st.secrets.get("GPT_KEY")
GPT3_ENGINE = st.secrets.get("GPT_ENGINE")
GPT4_BASE = st.secrets.get('GPT4o_BASE')
GPT4_VERSION = st.secrets.get('GPT4o_VERSION')
GPT4_KEY = st.secrets.get('GPT4o_KEY')
GPT4_ENGINE = st.secrets.get("GPT4o_ENGINE")

# Gemini secrets
USE_GEMINI = st.secrets.get("USE_GEMINI", False)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = st.secrets.get("GEMINI_CHAT_MODEL", "")
GEMINI_EMBEDDING_MODEL = st.secrets.get("GEMINI_EMBEDDING_MODEL", "")

if GPT_DEFAULT == "4":
    GPT_BASE = GPT4_BASE
    GPT_VERSION = GPT4_VERSION
    GPT_KEY = GPT4_KEY
    GPT_ENGINE = GPT4_ENGINE
elif GPT_DEFAULT == "3.5":
    GPT_BASE = GPT3_BASE
    GPT_VERSION = GPT3_VERSION
    GPT_KEY = GPT3_KEY
    GPT_ENGINE = GPT3_ENGINE
else:
    raise ValueError("GPT_DEFAULT must be '3.5' or '4'")
