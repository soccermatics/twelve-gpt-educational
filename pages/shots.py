# Library imports
from pathlib import Path
import sys
# path_root = Path(__file__).parents[1]
# print(path_root)
# sys.path.append(str(path_root))

#importing necessary libraries
from mplsoccer import Sbopen
import pandas as pd
import numpy as np
import warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os
import random as rn
#warnings not visible on the course webpage
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

#setting random seeds so that the results are reproducible on the webpage
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(1)
rn.seed(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import pandas as pd
import numpy as np
import argparse
import tiktoken
import os
from utils.utils import normalize_text

#from classes.data_source import PlayerStats
#from classes.data_point import Player
from classes.data_source import Shots
from classes.visual import ShotVisual
from classes.chat import Chat
from classes.description import ShotDescription



from utils.page_components import (
    add_common_page_elements
)


from classes.chat import PlayerChat

from utils.page_components import add_common_page_elements
from utils.utils import select_player, create_chat

sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()


st.markdown("# This is the shots and xG explanation page")

st.markdown("### Shots and xG")
shots = Shots()
shots_df= shots.df_shots

df_contributions = shots.df_contributions

# Create a dropdown to select a shot ID from the available shot IDs in shots.df_shots['id']
shot_id = st.sidebar.selectbox("Select Shot ID", shots_df['id'].unique())
st.markdown("#### Selected Shot Data")
st.write(shots_df[shots_df['id']== shot_id]) 
st.markdown("#### Feature Contributions")
st.write(df_contributions[df_contributions['id']== shot_id])

visuals = ShotVisual(metric=None)
visuals.add_shot(shots, shot_id)
descriptions = ShotDescription(shots, shot_id)
with st.expander("Messages"):
    st.write(descriptions.messages)

summaries = descriptions.stream_gpt()

chat = create_chat(tuple(shots_df['id'].unique()), Chat)

chat.add_message(visuals)
if summaries:
    chat.add_message(summaries)



chat.state = "default"
chat.display_messages()
chat.save_state()

#visual.add_title_from_match_player(match, selected_players)








