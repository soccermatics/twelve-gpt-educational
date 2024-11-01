from classes.data_source import CountryStats

countries = CountryStats()

metrics = [m for m in countries.df.columns if m not in ["country"]]

countries.calculate_statistics(metrics=metrics)

# # save countries.df to csv
# countries.df.to_csv("data/wvs/countries.csv", index=False)


import streamlit as st
from utils.utils import select_country, create_chat


from classes.chat import WVSChat
from classes.visual import DistributionPlot

from utils.page_components import add_common_page_elements

from classes.description import (
    CountryDescription,
)


# Function to load and inject custom CSS from an external file
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


import json
import pandas as pd

# def show():
sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

country = select_country(sidebar_container, countries)

st.divider()

st.write(
    "This app can only handle three or four users at a time. Please [download](https://github.com/soccermatics/twelve-gpt-educational) and run on your own computer with your own Gemini key."
)

# Read in model card text
with open("model cards/model-card-wvs-chat.md", "r", encoding="utf8") as file:
    # Read the contents of the file
    model_card_text = file.read()

####
import base64
import re


# Function to convert local images to base64
def convert_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
    return base64.b64encode(img_data).decode("utf-8")


# Pattern for Markdown image links
image_pattern = r"!\[(.*?)\]\((.*?)\)"


# Replace image links in text
def replace_images_in_text(text):
    def replacer(match):
        alt_text = match.group(1)
        link = match.group(2)

        if link.startswith("http"):
            # If it's a URL, return markdown for web image
            return f"![{alt_text}]({link})"
        else:
            # If it's a local file, convert to base64 and use HTML <img>
            try:
                data_url = convert_to_base64(link)
                return f'<img src="data:image/gif;base64,{data_url}" alt="{alt_text}" style="width:100%; max-width:900px;">'
            except FileNotFoundError:
                return f"![{alt_text}](Image not found: {link})"

    # Replace all image links with the appropriate HTML or markdown
    return re.sub(image_pattern, replacer, text)


# Process the text with image replacements
processed_text = replace_images_in_text(model_card_text)

####

load_css("model cards/style/python-code.css")
st.expander("Model card", expanded=False).markdown(
    processed_text,  # model_card_text,
    unsafe_allow_html=True,
)

st.expander("Dataframe used", expanded=False).write(countries.df)

with open("data/wvs/description_dict.json", "r") as f:
    description_dict = json.load(f)

thresholds_dict = dict(
    (
        metric,
        [
            2,
            1,
            -1,
            -2,
        ],
    )
    for metric in metrics
)

# check that the metrics exactly match the keys of the description_dict
if set(metrics) == set(description_dict.keys()):
    pass
else:
    raise ValueError(
        "The metrics do not match the keys of the description_dict. If you recently update the data then likely need to update the description_dict."
    )

# Chat state hash determines whether or not we should load a new chat or continue an old one
# We can add or remove variables to this hash to change conditions for loading a new chat
to_hash = (country.id,)

chat = create_chat(
    to_hash, WVSChat, country, countries, description_dict, thresholds_dict
)

# Now we want to add basic content to chat if it's empty
if chat.state == "empty":

    # Make a plot of the distribution of the metrics for all players
    # We reverse the order of the elements in metrics for plotting (because they plot from bottom to top)
    visual = DistributionPlot(
        metrics[::-1], labels=["Low", "Average", "High"], plot_type="wvs"
    )
    visual.add_title_from_player(country)
    visual.add_players(countries, metrics=metrics)
    visual.add_player(country, len(countries.df), metrics=metrics)

    # Now call the description class to get the summary of the country
    description = CountryDescription(
        country, description_dict=description_dict, thresholds_dict=thresholds_dict
    )
    summary = description.stream_gpt()

    # Add the visual and summary to the chat
    chat.add_message(
        "Please can you summarise " + country.name + " for me?",
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
chat.save_state()
