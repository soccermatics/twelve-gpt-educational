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




data = pd.read_csv("data/events/dataset.csv",encoding='unicode_escape')
person_stat = PersonStat()
person = person_stat.to_data_point(3)
description =  PersonDescription(person, person_stat)

st.write(description.get_description(person))
