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
from classes.chat import PlayerChat
from classes.description import PlayerDescription
from classes.visual import DistributionPlot

from utils.page_components import add_common_page_elements, select_two_players, create_chat


sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.subheader('Player Comparison')
minimal_minutes = 300
players = PlayerStats(minimal_minutes=minimal_minutes)

metrics = [
    "npxG_adjusted_per90", "goals_adjusted_per90", "assists_adjusted_per90", 
    "key_passes_adjusted_per90", "smart_passes_adjusted_per90", 
    "final_third_passes_adjusted_per90", "final_third_receptions_adjusted_per90", 
    "ground_duels_won_adjusted_per90", "air_duels_won_adjusted_per90"
]
players.calculate_statistics(metrics=metrics)
player1, player2 = select_two_players(sidebar_container, players, gender="male", position="Forward")

st.write(players.df)

st.divider()
st.subheader('Ask TwelveGPT')

#to_hash = (player1.id, player2.id)
to_hash = (player1.id,)


# Now create the chat as type PlayerChat
chat = create_chat(to_hash, PlayerChat, player1, players)

# Now we want to add basic content to chat if it's empty
if chat.state == "empty":

    # Make a plot of the distribution of the metrics for all players
    # We reverse the order of the elements in metrics for plotting (because they plot from bottom to top)
    visual = DistributionPlot(metrics[::-1])
    visual.add_title_from_player(player1)
    visual.add_players(players, metrics=metrics)
    visual.add_player(player1, len(players.df), metrics=metrics)

    # Now call the description class to get the summary of the player
    description = PlayerDescription(player1)
    summary = description.stream_gpt()

    # Add the visual and summary to the chat
    chat.add_message(
        "Please can you summarise " + player1.name + " for me?",
        role="user",
        user_only=False,
        visible=False,
    )
    chat.add_message(visual)
    chat.add_message(summary)

    chat.state = "default"

# Now we want to get the user input, display the messages and save the state
chat.get_input()
chat.display_messages()
chat.save_state()