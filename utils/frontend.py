import base64
from pathlib import Path

import streamlit as st
import pandas as pd
import requests


def set_page_config():
    """
    Sets the page configuration for the app.
    """
    st.set_page_config(
        layout="centered",
        page_title="TwelveGPT Scout",
        page_icon="data/ressources/img/twelve_chat_logo.svg",
        initial_sidebar_state="expanded",
        menu_items={
            "Report a bug": "mailto:matthias@twelve.football?subject=Bug report"
        },
    )


def add_page_selector():
    st.image("data/ressources/img/twelve_new_logo.svg")
    st.write("")
    if st.session_state.user_info:
        st.page_link("app.py", label="Scout Player")
        st.page_link("pages/find_players.py", label="Find Players")
        st.page_link("pages/match_analyst.py", label="Match Analyst")
        #st.page_link("pages/player_career.py", label="Player Career")
        st.page_link("pages/player_analyst.py", label="Player Analyst")
        #st.page_link("pages/match_reporter.py", label="Match Reporter")
        #st.page_link("pages/live_scout.py", label="Live Scout")
        #st.page_link("pages/test_chat.py", label="Test Chat")
        st.page_link("pages/league_analyst.py", label="League Analyst")
        st.page_link("pages/player_career.py", label="Player Career")
        st.page_link("pages/shot_commentator.py", label="Shot Commentator")
    # st.page_link("https://twelve.football", label="Twelve Homepage")
        #st.page_link("https://twelve.football", label="Twelve Homepage")
        st.page_link("pages/team_analyst.py", label="Season Analyst")
    st.divider()


@st.cache_data
def insert_local_css():
    """
    Injects the local CSS file into the app.
    Replaces the logo and font URL placeholders in the CSS file with base64 encoded versions.
    """
    with open("data/style.css", "r") as f:
        css = f.read()

    logo_url = (
        "url(data:image/png;base64,"
        + base64.b64encode(
            Path('data/ressources/img/twelve_logo_light.png')
            .read_bytes()
        ).decode()
        + ")"
    )
    font_url_medium = (
        "url(data:font/otf;base64,"
        + base64.b64encode(
            Path('data/ressources/fonts/Gilroy-Medium.otf')
            .read_bytes()
        ).decode()
        + ")"
    )
    font_url_light = (
        "url(data:font/otf;base64,"
        + base64.b64encode(
            Path('data/ressources/fonts/Gilroy-Light.otf')
            .read_bytes()
        ).decode()
        + ")"
    )
    
    css = css.replace("replace_logo_url", logo_url)
    css = css.replace("replace_font_url_medium", font_url_medium)
    css = css.replace("replace_font_url_light", font_url_light)

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def style_info_table(styler):
    styler.set_caption("Player Info")
    styler.set_table_styles(
        [
            {"selector": "thead", "props": [("display", "none")]},
        ]
    )
    return styler


def display_player_info(player):
    info_table = pd.DataFrame(
        {
            "Full name": [player.long_name],
            "Age": [player.age],
        }
    ).T

    st.table(info_table.style.pipe(style_info_table))


def email_form(player_id, competition_id, year):
    with st.form(key="email"):
        email_address = st.text_input(
            label="Email address",
            label_visibility="collapsed",
            value="example@email.com",
        )
        if st.form_submit_button(label="Email PDF report"):
            requests.post(
                "email_endpoint",
                data={
                    "player_id": player_id,
                    "competition_id": competition_id,
                    "year": year,
                    "email_address": email_address,
                },
            )
