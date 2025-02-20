"""
Page components for pages/*.py
"""

# Stdlib imports
import base64
from pathlib import Path

import streamlit as st

# from pages import about, football_scout, embedder, wvs_chat, own_page


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
            Path("data/ressources/img/twelve_logo_light.png").read_bytes()
        ).decode()
        + ")"
    )
    font_url_medium = (
        "url(data:font/otf;base64,"
        + base64.b64encode(
            Path("data/ressources/fonts/Gilroy-Medium.otf").read_bytes()
        ).decode()
        + ")"
    )
    font_url_light = (
        "url(data:font/otf;base64,"
        + base64.b64encode(
            Path("data/ressources/fonts/Gilroy-Light.otf").read_bytes()
        ).decode()
        + ")"
    )

    css = css.replace("replace_logo_url", logo_url)
    css = css.replace("replace_font_url_medium", font_url_medium)
    css = css.replace("replace_font_url_light", font_url_light)

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def set_page_config():
    """
    Sets the page configuration for the app.
    """
    st.set_page_config(
        layout="centered",
        page_title="TwelveGPT Scout",
        page_icon="data/ressources/img/TwelveEdu.png",
        initial_sidebar_state="expanded",
        menu_items={
            "Report a bug": "mailto:matthias@twelve.football?subject=Bug report"
        },
    )


def add_page_selector():
    st.image("data/ressources/img/TwelveEdu.png")
    st.page_link("pages/about.py", label="About")
    #st.page_link("pages/football_scout.py", label="Football Scout")
    #st.page_link("pages/embedder.py", label="Embdedding Tool")
    #st.page_link("pages/wvs_chat.py", label="World Value Survey")
    #st.page_link("pages/own_page.py", label="Your Own Page")

    # st.image("data/ressources/img/TwelveEdu.png")

    # # Define the available pages using their module names, not file paths
    # pages = {
    #     "About": about,
    #     "Football Scout": football_scout,
    #     "Embedder": embedder,
    #     "World Values Survey": wvs_chat,
    #     "Your Own Page": own_page,
    #     # Add other pages here
    # }

    # # Sidebar for page selection with default set to "About"
    # selected_page = st.sidebar.radio(
    #     "Select a page",
    #     list(pages.keys()),
    #     index=0,  # 'index=0' selects "About" by default
    # )

    # # Load and display the selected page's content by calling its `show` function
    # page = pages[selected_page]
    # # page.show()  # Assume each page has a `show()` function to display its content

    st.page_link("pages/shots.py", label="Shot xG Analysis")
    st.page_link("pages/eval.py", label="Shot Evaluation")
    

def add_common_page_elements():
    """
    Sets page config, injects local CSS, adds page selector and login button.
    Returns a container that MUST be used instead of st.sidebar in the rest of the app.

    Returns:
        sidebar_container: A container in the sidebar to hold all other sidebar elements.
    """
    # Set page config must be the first st. function called
    set_page_config()
    # Insert local CSS as fast as possible for better display
    insert_local_css()
    # Create a page selector
    page_selector_container = st.sidebar.container()
    sidebar_container = st.sidebar.container()

    page_selector_container = st.sidebar.container()
    sidebar_container = st.sidebar.container()

    with page_selector_container:
        add_page_selector()

    sidebar_container.divider()

    return sidebar_container
