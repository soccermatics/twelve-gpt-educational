"""
Entrypoint for streamlit app.
Runs top to bottom every time the user interacts with the app (other than imports and cached functions).
"""

# Library imports
import traceback
import copy

import streamlit as st

from classes.data_source import PlayerStats
from classes.data_point import Player
from classes.visual import DistributionPlot, ComparisonDistributionPlot
from classes.description import PlayerDescriptionComparison
from classes.chat import PlayerChat,PlayerChatComparison

from utils.page_components import add_common_page_elements, create_chat
from utils.page_components import select_two_players

sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()

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

if not player1 or not player2:
    st.write("Please select two players to compare.")
    st.stop()

st.write(players.df)

to_hash = (player1.id, player2.id)

chat = create_chat(to_hash, PlayerChatComparison, player1, player2,players)

if chat.state == "empty":

    #visual = ComparisonDistributionPlot(metrics[::-1])
    #visual.add_title_from_players(player1, player2)
    #visual.add_players(player1, player2, len(players.df), metrics)
    
    # Make a plot of the distribution of the metrics for all players
    #visual = DistributionPlot(metrics[::-1])
    #visual.add_title_from_player(player1, player2)  # Assuming this method can handle two players
    #visual.add_players(players, metrics=metrics)
    #visual.add_player(player1, len(players.df), metrics=metrics)
    #visual.add_player(player2, len(players.df), metrics=metrics)
    
    # Call the description comparison class to get the summary of both players
    description = PlayerDescriptionComparison(player1, player2)
    summary = description.stream_gpt()

    # Add the visual and summary to the chat
    chat.add_message(f"Please can you compare {player1.name} and {player2.name} for me?", role="user", user_only=False, visible=False)
    #chat.add_message(visual)
    chat.add_message(summary)

    chat.state = "default"

chat.get_input()
chat.display_messages()
chat.save_state()