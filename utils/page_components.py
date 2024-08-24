"""
Page components for app.py and pages/*.py
"""

# Stdlib imports
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
from utils.frontend import (
    insert_local_css,
    set_page_config,
)
from utils.sentences import pronouns


def add_page_selector():
    st.image("data/ressources/img/twelve_new_logo.svg")
    st.write("OPEN SOURCE")
    st.page_link("app.py", label="Football Scout")
    st.page_link("pages/embedder.py", label="Embdedding Tool")
    

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

import copy
import streamlit as st

def select_two_players(container, players, gender, position):
    # Make a copy of the Players object for each player
    player1 = copy.deepcopy(players)
    player2 = copy.deepcopy(players)

    with container:
        # Filter and select the first player
        player1.select_and_filter(
            column_name="player_name",
            label="Select the first player",
        )

        # Manually handle the Streamlit key for the second player selection
        player2_selectbox = st.selectbox(
            "Select the second player",
            player2.df["player_name"].unique(),
            key="player2_select"
        )
        player2.df = player2.df[player2.df["player_name"] == player2_selectbox]

    # Convert the selections into data points
    player1 = player1.to_data_point(gender, position)
    player2 = player2.to_data_point(gender, position)
    
    return player1, player2


def create_chat(to_hash, chat_class, *args, **kwargs):
    chat_hash_state = hash(to_hash)
    chat = chat_class(chat_hash_state, *args, **kwargs)
    return chat

def add_page_selector():
    st.image("data/ressources/img/twelve_new_logo.svg")
    st.write("OPEN SOURCE")
    st.page_link("app.py", label="Football Scout")
    st.page_link("pages/embedder.py", label="Embdedding Tool")
    st.page_link("pages/player_comparison.py", label="Player Comparison (Ana)")