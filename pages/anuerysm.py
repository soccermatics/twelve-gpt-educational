"""
Entrypoint for streamlit app.
Runs top to bottom every time the user interacts with the app (other than imports and cached functions).
"""

# Library imports
import traceback
import copy

import streamlit as st

from classes.chat import ModelChat
from classes.data_source import Model
from classes.data_point import Individual
from classes.visual import DistributionModelPlot
from classes.description import (
    IndividualDescription,
)
from classes.chat import PlayerChat

from utils.page_components import (
    add_common_page_elements,
    select_individual,
    create_chat,
)

sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()

individuals = Model()

individuals.weight_contributions()

# Now select the focal player
individual = select_individual(sidebar_container, individuals)

# Read in model card text
#with open("model cards/model-card-football-scout.md", 'r') as file:

model_card_text = "We need to write this."

st.expander("Model card", expanded=False).markdown(model_card_text)

st.expander("Dataframe used", expanded=False).write(individuals.df)

# Chat state hash determines whether or not we should load a new chat or continue an old one
# We can add or remove variables to this hash to change conditions for loading a new chat
to_hash = (individual.id,)
# Now create the chat as type PlayerChat
chat = create_chat(to_hash, ModelChat, individual, individuals)

metrics =  individuals.parameters['Parameter']

# Now we want to add basic content to chat if it's empty
if chat.state == "empty":

    # Make a plot of the distribution of the metrics for all players
    # We reverse the order of the elements in metrics for plotting (because they plot from bottom to top)
    visual = DistributionModelPlot(metrics)
    visual.add_title('Evaluation of individual','')
    visual.add_individuals(individuals, metrics=metrics)
    visual.add_individual(individual, len(individuals.df), metrics=metrics)

    # Now call the description class to get the summary of the player
    description = IndividualDescription(individual,metrics,parameter_explanation=individuals.parameter_explanation)
    summary = description.stream_gpt()

    # Add the visual and summary to the chat
    chat.add_message(
        "Please can you summarise this individual for me?",
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
