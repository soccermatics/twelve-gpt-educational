
# Library imports
import streamlit as st
import pandas as pd
import argparse
import tiktoken
import os
from utils.utils import normalize_text

from classes.data_source import Arguments

from classes.description import (
    TrolleyDescription
)
from classes.chat import TrolleyChat

from utils.page_components import (
    add_common_page_elements,
    create_chat,
)

from classes.visual import TreePlot

sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()

arguments=  Arguments()

overall='1.'
current = '1.'
userStance = 'Pro'
# make a dictionary to give opposite Pro and Con arguments
stanceSwap = {'Pro': 'Con', 'Con': 'Pro'}
stanceFullName = {'Pro': 'in support of', 'Con': 'against'}

# Get the overall argument from currentArguments
overallArgument = arguments.df[arguments.df['assistant']==overall].iloc[0]['user']

text = 'Thank you for discussing the following thesis with me: ' + overallArgument
st.write(text)
st.write('You should argue ' + stanceFullName[userStance] + ' this thesis. I will argue ' + stanceFullName[stanceSwap[userStance]] + ' the thesis. I will start.')


if 'argumentsMade' not in st.session_state:
    st.session_state.argumentsMade = []

to_hash = (current, )

chat = create_chat(to_hash, TrolleyChat, arguments, overallArgument, stance=stanceFullName[stanceSwap[userStance]],argumentsMade=st.session_state.argumentsMade)

# Now we want to add basic content to chat if it's empty
if chat.state == "empty":

    # Make a plot 
    # visual = TreePlot(arguments.df['user'], arguments.df['assistant'])
    # visual.add_tree('Trolley Problem')
    

    # Gets the arguments at current level and supporting arguments one below.
    currentArguments= arguments.get_arguments(current,stanceSwap[userStance])
    description = TrolleyDescription(currentArguments, overallArgument,stanceFullName[stanceSwap[userStance]])
    summary = description.stream_gpt()

    # Add the visual and summary to the chat
    #chat.add_message(
    #    "Please can you summarise " + player.name + " for me?",
    #    role="user",
    #    user_only=False,
    #    visible=False,
    #)

    #chat.add_message(visual)
    chat.add_message(summary)

    chat.state = "default"

# Now we want to get the user input, display the messages and save the state
chat.get_input()
chat.display_messages()
st.session_state.argumentsMade = chat.argumentsMade
chat.save_state()

