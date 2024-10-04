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
import argparse
import tiktoken
import os
from utils.utils import normalize_text

from classes.data_source import PlayerStats
from classes.data_point import Player
from classes.data_source import PlayerStats, Shots


from utils.page_components import (
    add_common_page_elements
)

sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()


st.write("To make your own page create a page_name.py file and link to it in add_page_selector() in utils/page_components.py")


shots = Shots()
shots_df= shots.df_shots

st.write(shots_df.head(10))  # Display selected columns





