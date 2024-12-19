"""
Page components for app.py and pages/*.py
"""

# Stdlib imports
import base64
from pathlib import Path

from typing import Optional
import traceback
import copy
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import requests

from classes.chat import Chat
from classes.data_source import PlayerStats
from classes.data_point import Player

from utils.sentences import pronouns


def insert_local_css():
    """
    Injects the local CSS file into the app.
    Replaces the logo and font URL placeholders in the CSS file with base64 encoded versions.
    """
    with open("data/style.css", "r") as f:
        css = f.read()

    logo_url = (
        "url(data:image/png;base64,"
        + base64.b64encode(
            Path('data/ressources/img/twelve_logo_light.png')
            .read_bytes()
        ).decode()
        + ")"
    )
    font_url_medium = (
        "url(data:font/otf;base64,"
        + base64.b64encode(
            Path('data/ressources/fonts/Gilroy-Medium.otf')
            .read_bytes()
        ).decode()
        + ")"
    )
    font_url_light = (
        "url(data:font/otf;base64,"
        + base64.b64encode(
            Path('data/ressources/fonts/Gilroy-Light.otf')
            .read_bytes()
        ).decode()
        + ")"
    )
    
    css = css.replace("replace_logo_url", logo_url)
    css = css.replace("replace_font_url_medium", font_url_medium)
    css = css.replace("replace_font_url_light", font_url_light)

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)



def set_page_config():
    """
    Sets the page configuration for the app.
    """
    st.set_page_config(
        layout="centered",
        page_title="TwelveGPT Scout",
        page_icon="data/ressources/img/TwelveEdu.png",
        initial_sidebar_state="expanded",
        menu_items={
            "Report a bug": "mailto:matthias@twelve.football?subject=Bug report"
        },
    )


def add_page_selector():
    st.image("data/ressources/img/TwelveEdu.png")
    #st.page_link("app.py", label="About")
    #st.page_link("pages/football_scout.py", label="Football Scout")
    #st.page_link("pages/embedder.py", label="Embdedding Tool")
    #st.page_link("pages/trolley.py", label="Trolley Problem")
    #st.page_link("pages/own_page.py", label="Your own page")
    st.page_link("pages/interactive_education.py", label="CP teaching agent")
    

def add_common_page_elements():
    """
    Sets page config, injects local CSS, adds page selector and login button.
    Returns a container that MUST be used instead of st.sidebar in the rest of the app.
    
    Returns:
        sidebar_container: A container in the sidebar to hold all other sidebar elements.
    """
    # Set page config must be the first st. function called
    set_page_config()
    # Insert local CSS as fast as possible for better display
    insert_local_css()
    # Create a page selector
    page_selector_container = st.sidebar.container()
    sidebar_container = st.sidebar.container()

    page_selector_container = st.sidebar.container()
    sidebar_container = st.sidebar.container()

    with page_selector_container:
        add_page_selector()

    sidebar_container.divider()

    return sidebar_container    


def select_player(container,players,gender,position):

    # Make a copy of Players object
    player=copy.deepcopy(players)

    # Filter players by position and select a player with sidebar selectors
    with container:

        # Filter for player name
        player.select_and_filter(
            column_name="player_name",
            label="Player",
        )

        # Return data point

        player=player.to_data_point(gender,position)
        
    return player

def create_chat(to_hash, chat_class, *args, **kwargs):
    chat_hash_state = hash(to_hash)
    chat = chat_class(chat_hash_state, *args, **kwargs)
    return chat
