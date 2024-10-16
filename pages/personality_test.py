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


import utils.sentences as sentences

from utils.page_components import (add_common_page_elements)

import traceback
import copy


from utils.page_components import (add_common_page_elements,select_person,create_chat,)

sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()

#data = pd.read_csv("data/events/dataset.csv",encoding='unicode_escape')
person_stat = PersonStat()

with st.expander("Dataframe"):
    st.write(person_stat.df)


person = select_person(sidebar_container, person_stat)
description =  PersonDescription(person)
st.write( description.get_description(person))
metrics = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']


visual = DistributionPlotPersonality(metrics)
visual.add_title_from_person(person)
visual.add_persons(persons=person_stat,metrics=metrics)
visual.add_person(person)


'''
# Now select the candidate
player = select_player(sidebar_container, person) # TO CHANGE

st.write("This app can only handle three or four users at a time. Please [download](https://github.com/soccermatics/twelve-gpt-educational) and run on your own computer with your own Gemini key.")

st.expander("Dataframe used", expanded=False).write(data)

# Chat state hash determines whether or not we should load a new chat or continue an old one
# We can add or remove variables to this hash to change conditions for loading a new chat
to_hash = (person.id,)

# Now create the chat as type PlayerChat
chat = create_chat(to_hash, PlayerChat, person, person) # TO CHANGE

# Now we want to add basic content to chat if it's empty
if chat.state == "empty":

    # Make a plot of the distribution of the metrics for all players
    # We reverse the order of the elements in metrics for plotting (because they plot from bottom to top)

    visual =DistributionPlotPersonality(metrics)
    visual.add_title_from_person(person)
    visual.add_persons(metrics=metrics)
    visual.add_person(person)
    # Now call the description class to get the summary of the player
    description = PersonDescription(person)
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
chat.save_state()'''
