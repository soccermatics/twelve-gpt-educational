import os
import streamlit as st
try:
    API_URL = st.secrets.get('API_URL')
    NEW_API_URL = st.secrets.get('NEW_API_URL')
    TOKEN = st.secrets.get('TOKEN')

    ENGINE_ADA = st.secrets.get("ENGINE_ADA")
    GPT_DEFAULT = "4"
    GPT3_BASE = st.secrets.get("GPT_BASE")
    GPT3_VERSION = st.secrets.get("GPT_VERSION")
    GPT3_KEY = st.secrets.get("GPT_KEY")
    GPT3_ENGINE = st.secrets.get("ENGINE_GPT")
    CUSTOMERIO_KEY = st.secrets.get('CUSTOMERIO_KEY')
    GPT4_BASE = st.secrets.get('GPT4_BASE')
    GPT4_VERSION = st.secrets.get('GPT4_VERSION')
    GPT4_KEY = st.secrets.get('GPT4_KEY')
    GPT4_ENGINE = st.secrets.get("GPT4_ENGINE")
    AUTH0_CLIENT_ID = st.secrets.get("AUTH0_CLIENT_ID")
    AUTH0_DOMAIN = st.secrets.get("AUTH0_DOMAIN")

    # BLOBS for pdf
    AZURE_BLOB_CONTAINER = os.environ.get('AZURE_BLOB_CONTAINER')
    AZURE_BLOB_CONNECTION = f"AZURE_BLOB_CONNECTION=DefaultEndpointsProtocol=https;AccountName={os.environ.get('AccountName')};AccountKey={os.environ.get('AccountKey')};EndpointSuffix=core.windows.net"
    BLOB_URL = os.environ.get('BLOB_URL')

    WYSCOUT_USER = st.secrets.get("WYSCOUT_USER") #TODO:remove @agust and use fastapi
    WYSCOUT_PASS = st.secrets.get("WYSCOUT_PASS")

except FileNotFoundError as err:

    #if GPT3_BASE is None or GPT3_VERSION is None or GPT3_KEY is None or GPT3_ENGINE is None:

    # Load from azure not streamlit
    from dotenv import load_dotenv

    load_dotenv()  # Load variables from .env file
    #TODO move elsewhere
    GPT_DEFAULT = "3.5"
    GPT3_BASE = os.environ.get("GPT3_BASE")
    GPT3_VERSION = os.environ.get("GPT3_VERSION")
    GPT3_KEY = os.environ.get("GPT3_KEY")
    GPT3_ENGINE = os.environ.get("GPT3_ENGINE")
    CUSTOMERIO_KEY = os.environ.get('CUSTOMERIO_KEY')
    TOKEN = os.environ.get('TOKEN')
    API_URL = os.environ.get('API_URL')
    NEW_API_URL = os.environ.get('NEW_API_URL')

    ENGINE_ADA = os.environ.get("ENGINE_ADA")

    GPT4_BASE = os.environ.get('GPT4_BASE')
    GPT4_VERSION = os.environ.get('GPT4_VERSION')
    GPT4_KEY = os.environ.get('GPT4_KEY')
    GPT4_ENGINE = os.environ.get("GPT4_ENGINE")
    AUTH0_CLIENT_ID = os.environ.get("AUTH0_CLIENT_ID")
    AUTH0_DOMAIN = os.environ.get("AUTH0_DOMAIN")

    # BLOBS for pdf
    AZURE_BLOB_CONTAINER = os.environ.get('AZURE_BLOB_CONTAINER')
    AZURE_BLOB_CONNECTION = f"AZURE_BLOB_CONNECTION=DefaultEndpointsProtocol=https;AccountName={os.environ.get('AccountName')};AccountKey={os.environ.get('AccountKey')};EndpointSuffix=core.windows.net"
    BLOB_URL = os.environ.get('BLOB_URL')

    WYSCOUT_USER = ""
    WYSCOUT_PASS = ""

#WYSCOUT_USER = st.secrets["WYSCOUT_USER"]
#WYSCOUT_PASS = st.secrets["WYSCOUT_PASS"]

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