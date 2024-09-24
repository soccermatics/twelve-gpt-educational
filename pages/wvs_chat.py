# Library imports
import streamlit as st
from utils.utils import select_country, create_chat

from classes.data_source import CountryStats

# from classes.chat import WVSChat

from utils.page_components import add_common_page_elements


# def show():
sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()

countries = CountryStats()

metrics = [m for m in countries.df.columns if m not in ["country"]]

countries.calculate_statistics(metrics=metrics)

country = select_country(sidebar_container, countries)

st.write(
    "This app can only handle three or four users at a time. Please [download](https://github.com/soccermatics/twelve-gpt-educational) and run on your own computer with your own Gemini key."
)

st.expander("Dataframe used", expanded=False).write(countries.df)


# Chat state hash determines whether or not we should load a new chat or continue an old one
# We can add or remove variables to this hash to change conditions for loading a new chat
to_hash = (country.id,)

# chat = create_chat(to_hash, WVSChat, country, countries)


# st.write("Under construction")
