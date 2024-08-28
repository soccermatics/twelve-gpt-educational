#Init
# Library imports
import streamlit as st
import pandas as pd
import argparse
import tiktoken
import os
from utils.utils import normalize_text

from classes.data_source import PlayerStats
from classes.data_point import Player

from utils.page_components import (
    add_common_page_elements
)


def file_walk(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            if not name.endswith('.DS_Store'):  # Skip .DS_Store files
                yield root, name


def get_format(path):
    file_format = "." + path.split(".")[-1]
    if file_format == ".xlsx":
        read_func = pd.read_excel
    elif file_format == ".csv":
        read_func = pd.read_csv
    else:
        raise ValueError(f"File format {file_format} not supported.")
        print("unected file: " + path )
    return file_format, read_func


def select_player(container, players, gender, position):

    # Make a copy of Players object
    player = copy.deepcopy(players)

    # Filter players by position and select a player with sidebar selectors
    with container:

        # Filter for player name
        player.select_and_filter(
            column_name="player_name",
            label="Player",
        )

        # Return data point

        player = player.to_data_point(gender, position)

    return player


def create_chat(to_hash, chat_class, *args, **kwargs):
    chat_hash_state = hash(to_hash)
    chat = chat_class(chat_hash_state, *args, **kwargs)
    return chat

sidebar_container = add_common_page_elements()

st.divider()

embeddings = Embeddings()

directory= st.text_input("Directory to embedd", "")
st.write("Starting to embedd " + directory)

path_describe = os.path.normpath("data/describe/"+directory)
path_embedded = os.path.normpath("data/embeddings/"+directory)


st.write("Updating all embeddings...")
for root, name in file_walk(path_describe):
    print_path = os.path.join(root, name).replace(path_describe, "")[1:]
    embed(os.path.join(root, name),embeddings)

