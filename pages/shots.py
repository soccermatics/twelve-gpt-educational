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
import json
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
from classes.visual import ShotVisual, DistributionPlot, ShotContributionPlot
from classes.chat import Chat
from classes.description import ShotDescription

# Function to load and inject custom CSS from an external file
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


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

st.markdown("## Shot commentator")

#parser = Sbopen()
#df_match = parser.match(competition_id=55, season_id=282)
#match_ids = df_match['match_id'].unique()
competitions = {
    "EURO Men 2024": "data/match_id_to_name_EURO_2024.json",
    "EURO Men 2020": "data/match_id_to_name_EURO_2020.json",
    "National Women's Soccer League (NWSL) 2018": "data/match_id_to_name_NWSL.json",
    "FIFA 2022": "data/match_id_to_name_FIFA_2022.json",
    "Women's Super League (FAWSL) 2017-18": "data/match_id_to_name_FAWSL.json",
    "Africa Cup of Nations (AFCON) 2023": "data/match_id_to_name_AFCON_2023.json"
}

# Select a competition
selected_competition = st.sidebar.selectbox("Select a Competition", options=competitions.keys())

# Load the JSON file corresponding to the selected competition
file_path = competitions[selected_competition]

with open(file_path, 'r') as f:
    id_to_match_name = json.load(f)



selected_match_name = st.sidebar.selectbox(
    "Select a Match", 
    options=id_to_match_name.values())

match_name_to_id = {v: k for k, v in id_to_match_name.items()}
selected_match_id = match_name_to_id[selected_match_name]

shots = Shots(selected_competition, selected_match_id)
shots_df= shots.df_shots
df_contributions = shots.df_contributions
st.write(shots_df)


excluded_columns = ['xG', 'id', 'match_id']
metrics = [col for col in df_contributions.columns if col not in excluded_columns]

# Create a dropdown to select a shot ID from the available shot IDs in shots.df_shots['id']

id_to_number = {shot_id: idx + 1 for idx, shot_id in enumerate(shots_df['id'])}
number_to_id = {v: k for k, v in id_to_number.items()}


# selected_number= st.sidebar.selectbox("Select a Shot:",
#     options=list(number_to_id.keys()),  
#     format_func=lambda x: f"Shot #{x}")

# shot_id = number_to_id[selected_number]

shots_df['player_minute'] = shots_df['player_name'] + " - " + shots_df['minute'].astype(str)
selected_player_minute = st.sidebar.selectbox(
    "Select a Shot (Player - Minute):",
    options=shots_df['player_minute'].unique())
selected_shot = shots_df[shots_df['player_minute'] == selected_player_minute]

#st.write(selected_shot)


if not selected_shot.empty:
    shot_id = selected_shot.iloc[0]['id']  # Retrieve the shot_id for the selection
else:
    st.warning("No matching shot found.")

# Read in model card text
with open("model cards/model-card-shot-xG-analysis.md", "r") as file:
     # Read the contents of the file
    model_card_text = file.read()

load_css("model cards/style/python-code.css")
st.expander("Model card", expanded=False).markdown(model_card_text)

#st.markdown("#### Selected Shot Data")
#shot = shots_df[shots_df['id']== shot_id]
#st.write(shot) 
#st.markdown("#### Feature Contributions")
#st.write(df_contributions[df_contributions['id']== shot_id])

to_hash = (selected_match_id, shot_id)



visuals = ShotVisual(metric=None)
visuals.add_shot(shots, shot_id)
visuals2= ShotContributionPlot(df_contributions=df_contributions, df_shots= shots_df, metrics=metrics)
visuals2.add_shots(shots_df, metrics, id_to_number= id_to_number)
visuals2.add_shot(contribution_df=df_contributions, shots_df= shots_df, shot_id=shot_id, metrics=metrics, id_to_number= id_to_number)


descriptions = ShotDescription(shots, shot_id, selected_competition)

summaries = descriptions.stream_gpt()
chat = create_chat(to_hash, Chat)

#chat = create_chat(tuple(shots_df['id'].unique()), Chat)

chat.add_message(visuals2)
chat.add_message(visuals)
if summaries:
    chat.add_message(summaries)



chat.state = "default"
chat.display_messages()
chat.save_state()

#visual.add_title_from_match_player(match, selected_players)








