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
from classes.visual import DistributionPlot
from classes.description import (
    PlayerDescription,
)
from classes.chat import PlayerChat

from utils.page_components import (
    add_common_page_elements,
    select_player,
    create_chat,
)

sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()


# minimal_minutes is the minimum number of minutes a player must have played to be included in the analysis
minimal_minutes = 300
players = PlayerStats(minimal_minutes=minimal_minutes)


# Define the metrics we are interested in and calculates them
metrics = [
    "npxG_adjusted_per90",
    "goals_adjusted_per90",
    "assists_adjusted_per90",
    "key_passes_adjusted_per90",
    "smart_passes_adjusted_per90",
    "final_third_passes_adjusted_per90",
    "final_third_receptions_adjusted_per90",
    "ground_duels_won_adjusted_per90",
    "air_duels_won_adjusted_per90",
]
players.calculate_statistics(metrics=metrics)

# Now select the focal player
player = select_player(sidebar_container, players, gender="male", position="Forward")

st.write("This app can only handle three or four users at a time. Please [download](https://github.com/soccermatics/twelve-gpt-educational) and run on your own computer with your own Gemini key.")

# Read in model card text
with open("model cards/model-card-football-scout.md", 'r',encoding='utf-8') as file:
    # Read the contents of the file
    model_card_text = file.read()


st.expander("Model card for Football Scout", expanded=False).markdown(model_card_text)

st.expander("Dataframe used", expanded=False).write(players.df)

# Chat state hash determines whether or not we should load a new chat or continue an old one
# We can add or remove variables to this hash to change conditions for loading a new chat
to_hash = (player.id,)
# Now create the chat as type PlayerChat
chat = create_chat(to_hash, PlayerChat, player, players)

# Now we want to add basic content to chat if it's empty
if chat.state == "empty":

    # Make a plot of the distribution of the metrics for all players
    # We reverse the order of the elements in metrics for plotting (because they plot from bottom to top)
    visual = DistributionPlot(metrics[::-1])
    visual.add_title_from_player(player)
    visual.add_players(players, metrics=metrics)
    visual.add_player(player, len(players.df), metrics=metrics)

    # Now call the description class to get the summary of the player
    description = PlayerDescription(player)
    summary = description.stream_gpt()

    # Add the visual and summary to the chat
    chat.add_message(
        "Please can you summarise " + player.name + " for me?",
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
