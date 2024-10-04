"""
Entrypoint for streamlit app.
Runs top to bottom every time the user interacts with the app (other than imports and cached functions).
"""
from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
print(path_root)
sys.path.append(str(path_root))
# Library imports
import traceback
import copy

import streamlit as st


from utils.page_components import (
    add_common_page_elements,
)

sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()

displaytext= ('''## About Twelve GPT Educational ''')

st.markdown(displaytext)

displaytext= (
    '''TwelveGPT Educational is a framework for creating data-driven chatbots. The design and code for this project was by Matthias Green, David Sumpter and Ágúst Pálmason Morthens. \n\n'''
    '''The code is set up in a general way, to allow users to build bots which talk about data. '''
    '''The football scout bot displays a distribution plot of a football player's performance in various metrics. It then starts a chat giving an AI generated summary of the player's performance and asks a variety of questions about the player. \n\n'''
    '''This is **not** the [Twelve GPT product](https://twelve.football/), but rather a (very) stripped down version of our code '''
    '''to help people who would like to learn how to build bots to talk about football data. There are lots of things which [Twelve GPT](https://twelve.football/) can do, which TwelveGPT Educational cannot do. But we want more people to learn about the methods we use and to do this **TwelveGPT Educational** is an excellent alternative. We have thus used the the GNU GPL license which requires that all the released improved versions are also be free software. This will allow us to learn from each other in developing better. \n\n '''
    '''If you work for a footballing organisation and would like to see a demo of the full Twelve GPT product then please email us at hello@twelve.football. '''
 )

st.markdown(displaytext)