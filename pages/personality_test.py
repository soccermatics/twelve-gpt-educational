import streamlit as st
import json
import openai
from itertools import groupby
from types import GeneratorType
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from classes.data_source import PersonStat
from classes.data_point import Person
from classes.description import PersonDescription
from classes.visual import DistributionPlot,DistributionPlotPersonality

from settings import GPT_BASE, GPT_VERSION, GPT_KEY, GPT_ENGINE

from classes.chat import PersonChat
import utils.sentences as sentences

from utils.page_components import (add_common_page_elements)

import traceback
import copy


from utils.page_components import (add_common_page_elements,select_person,create_chat,)

sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()

persons = PersonStat()
# Define the metrics we are interested in and calculates them
metrics = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']
persons.calculate_statistics(metrics=metrics)

with st.expander("Dataframe"):
    st.write(persons.df)


person = select_person(sidebar_container, persons)

#description =  PersonDescription(person)
#st.write( description.get_description(person))





# Chat state hash determines whether or not we should load a new chat or continue an old one
# We can add or remove variables to this hash to change conditions for loading a new chat
to_hash = (person.id,)
# Now create the chat as type PersonChat
chat = create_chat(to_hash, PersonChat, person, persons)

# Now we want to add basic content to chat if it's empty
if chat.state == "empty":

    # Make a plot of the distribution of the metrics for all players
    # We reverse the order of the elements in metrics for plotting (because they plot from bottom to top)
    visual = DistributionPlotPersonality(metrics[::-1])
    visual.add_title_from_person(person)
    visual.add_persons(persons,metrics=metrics)
    visual.add_person(person, len(persons.df),metrics=metrics)

    # Now call the description class to get the summary of the player
    description =  PersonDescription(person)
    summary = description.stream_gpt()

    # Add the visual and summary to the chat
    chat.add_message(
        "Please can you summarise " + person.name + " for me?",
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
