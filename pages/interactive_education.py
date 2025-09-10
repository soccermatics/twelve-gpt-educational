
# Library imports
import streamlit as st
import pandas as pd
import argparse
import tiktoken
import os
import numpy as np
import json
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import time
from io import BytesIO
import uuid
from dotenv import load_dotenv,find_dotenv
from dotenv import dotenv_values
import sys
import dropbox
from dropbox import DropboxOAuth2FlowNoRedirect
from utils.utils import normalize_text
from io import BytesIO

from classes.data_source import Arguments

from classes.data_source import Lesson

from classes.description import (
    TrolleyDescription, 
)
from classes.description import (
    LessonDescription, 
)
from classes.chat import TrolleyChat, LessonChat

from utils.page_components import (
    add_common_page_elements,
    create_chat,
)

from classes.visual import TreePlot

sidebar_container = add_common_page_elements()
#page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()

lesson = Lesson()

with open("model cards/model-card-CProgrammingagent.md", 'r',encoding='utf-8') as file:
    # Read the contents of the file
    model_card_text = file.read()


st.expander("Model card for CProgramming agent", expanded=False).markdown(model_card_text)


overall='1'

# Selcet random topics 
if 'argumentsMade' not in st.session_state:
    st.session_state.userStance = np.random.choice(['defination','implementation', 'implementation','defination','implementation','condtion'])
userStance = st.session_state.userStance
# make a dictionary to give opposite Pro and Con arguments
 #stanceSwap = {'Pro': 'Con', 'Con': 'Pro'}
stanceFullName = {'defination':'implementation', 'implementation':'defination', 'implementation':'condtion'}

# Get the overall thesis
overallThesis = lesson.df[lesson.df['step']==overall].iloc[0]['assistant']
#st.write(overallThesis)
currentState=lesson.df[lesson.df['step']==overall].iloc[0]['topic']
#st.write(currentState)
displaytext= (
    "## The programming lesson chat\n\n"
    "Do you want to learn about programming concepts in an interactive manner! "
    "In this chat, we are going to teach you programming by prompting you to respond to questions. "
    "Each response you make will be used to determine what you will be asked to do next. "
    "The aim is to get you to understand the main concepts of programming without just copy pasting! \n\n " 
    "**Enjoy the lesson and let's get started!** \n\n "
    )

st.markdown(displaytext)

background = '**Background**: You will be prompted to do a task, based on your response you will be guided.'
st.markdown(background)
#text = '**Thesis**: ' + overallThesis
#st.markdown(text)
st.markdown(' You should respond to the questions asked with all honesty.')

if 'argumentsMade' not in st.session_state:
    st.session_state.argumentsMade = []
if 'gameOver' not in st.session_state:
    st.session_state.gameOver = False

to_hash = (overall)
chat = create_chat(to_hash, LessonChat, overallThesis, lesson, gameOver=st.session_state.gameOver)
# Now we want to add basic content to chat if it's empty
if chat.state == "empty":

    #Gets the arguments at current level and supporting arguments one below.
    currentArguments= lesson.get_arguments(overall,overallThesis )
    #st.write(currentArguments)
    description = LessonDescription(chat.state, currentArguments,overallThesis)
    summary = description.stream_gpt()

    #chat.add_message("What do you know about for loops as used in C programming langaguge?")
    chat.add_message(overallThesis)

    chat.state = "default"

# Now we want to get the user input, display the messages and save the state
#st.write(chat.state)

chat.get_input()
chat.display_messages()
st.session_state.arguments = chat.arguments
st.session_state.gameOver = chat.gameOver
chat.save_state()
#conn = st.connection("gsheets", type=GSheetsConnection)
#st.write(st.session_state)
#______________________________________________________________________________
# load_dotenv(find_dotenv())
# #tesu=load_dotenv()
# #st.write(tesu)
# path = "a.env"  #try .path[0] if 1 doesn't work
# load_dotenv(path)
# #config = dotenv_values(".env")
# #st.write(config)
# app_key= os.getenv("APP_KEY")
# app_secret = os.getenv("APP_SECRET")
# access_token=os.getenv("ACCESS_TOKEN")
# # Dropbox API credentials

# try:
#     dbx = dropbox.Dropbox(access_token)
#     st.write("Successfully connected ")
# except Exception as e:
#     st.error(f"Failed to connect to Dropbox: {e}")
# #dbx = dropbox.Dropbox(access_token)

# dropbox_file_path = "/uploaded_dataframe.csv"


# if "messages" not in st.session_state:
#     st.session_state["messages"] = st.session_state["messages_to_display"],

# # Function to upload file to Dropbox


# def upload_dataframe_to_dropbox(df, dropbox_path):
    
#     # Convert DataFrame to CSV in-memory
#     csv_buffer = BytesIO()
#     df.to_csv(csv_buffer, index=False)
#     csv_buffer.seek(0)  # Reset buffer pointer to the beginning

#     try:
#         # Upload CSV content to Dropbox
#         dbx.files_upload(
#             csv_buffer.read(),
#             dropbox_path,
#             mode=dropbox.files.WriteMode.overwrite,
#         )
#         return True, f"File successfully uploaded to Dropbox at {dropbox_path}"
#     except dropbox.exceptions.ApiError as e:
#         return False, f"Dropbox API error: {e}"
#     except Exception as e:
#         return False, f"Unexpected error: {e}"

def save_chat_history():
    # Create the filename using the session ID
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    filename = f"chat_{st.session_state['session_id']}.csv"
    dropbox_file_path=f"/{filename}.csv"
    
    # Convert chat history to a DataFrame
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []  # Initialize with an empty list
    chat_df = pd.DataFrame(st.session_state["chat_history"])
    
    # Save the DataFrame to a CSV file
    #file_content=chat_df.to_csv(filename, index=False)
    
    #upload_to_dropbox(st.session_state["chat_history"], filename)
    success, message = upload_dataframe_to_dropbox(chat_df, dropbox_file_path)
    if success:
        #st.success(message)
        st.write("Saved")
    else:
        st.error(message)

    return filename


# if st.button("Finish"):
#     if st.session_state["messages"]:
#         # Append user input to chat history
#         if "chat_history" not in st.session_state:
#             st.session_state["chat_history"] = [] 
#         st.session_state["chat_history"].append({"role": "user", "content":st.session_state["messages"]})
        
#         # Save chat history after every interaction
#     save_chat_history()
#     st.write("Congratulations on finishing the for loop lesson")


