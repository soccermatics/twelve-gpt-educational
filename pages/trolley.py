
# Library imports
import streamlit as st
import pandas as pd
import argparse
import tiktoken
import os
import numpy as np

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

arguments = Arguments()


overall='1.'

# Selcet random Pro or Con stance
if 'argumentsMade' not in st.session_state:
    st.session_state.userStance = np.random.choice(['Pro', 'Con'])
userStance = st.session_state.userStance

# make a dictionary to give opposite Pro and Con arguments
stanceSwap = {'Pro': 'Con', 'Con': 'Pro'}
stanceFullName = {'Pro': 'in support of', 'Con': 'against'}

# Get the overall thesis
overallThesis = arguments.df[arguments.df['assistant']==overall].iloc[0]['user']

displaytext= (
    "## The Argument Game\n\n"
    "Do you think you can see all sides of an argument? Let's find out! "
    "In the argument game you will be given a thesis and a stance. You will then be asked to argue for or against that thesis. "
    "Each argument you make will be given a score out of 10 based on its novelty. "
    "The aim is to get to 100 points. But watch out ... if you are too repetitive in your arguments the game will end before you get there!\n\n" 
    "**Good luck and let's get started!**\n\n"
    )

st.markdown(displaytext)

background = '**Background**: A runaway trolley runs down a track; ahead are five people awaiting certain death. You observe the scenario from nearby and see a lever next to you. If you pull the lever you can divert the trolley to a different set of tracks. Yet, on that other track is a single person. The train cannot be stopped.'
st.markdown(background)
text = '**Thesis**: ' + overallThesis
st.markdown(text)
st.markdown(' You should argue **' + stanceFullName[userStance] + '** this thesis. I will argue ' + stanceFullName[stanceSwap[userStance]] + ' the thesis. I will begin.')

if 'argumentsMade' not in st.session_state:
    st.session_state.argumentsMade = []

if 'totalscore' not in st.session_state:
    st.session_state.totalscore = 0

if 'gameOver' not in st.session_state:
    st.session_state.gameOver = False

to_hash = (overall)

chat = create_chat(to_hash, TrolleyChat, arguments, userStance,overallThesis,argumentsMade=st.session_state.argumentsMade,totalscore=st.session_state.totalscore,gameOver=st.session_state.gameOver)

# Now we want to add basic content to chat if it's empty
if chat.state == "empty":

    # Gets the arguments at current level and supporting arguments one below.
    currentArguments= arguments.get_arguments(overall,stanceSwap[userStance])
    description = TrolleyDescription(currentArguments, overallThesis,stanceFullName[stanceSwap[userStance]])
    summary = description.stream_gpt()

    chat.add_message(summary)

    chat.state = "default"

# Now we want to get the user input, display the messages and save the state
chat.get_input()
chat.display_messages()
st.session_state.totalscore =  chat.totalscore
st.session_state.argumentsMade = chat.argumentsMade
st.session_state.gameOver = chat.gameOver
chat.save_state()

