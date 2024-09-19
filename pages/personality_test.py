import streamlit as st
import json
import openai
from itertools import groupby
from types import GeneratorType
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from classes.data_source import PersonalityStats
from classes.data_point import Person
from classes.personality_description import*

from settings import GPT_BASE, GPT_VERSION, GPT_KEY, GPT_ENGINE


import utils.sentences as sentences

from utils.page_components import (add_common_page_elements)


sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()



st.write(PersonDescription.get_description(4))
