import streamlit as st
import json
import openai
from itertools import groupby
from types import GeneratorType
import pandas as pd
import numpy as np
import utils.sentences as sentences
from utils.gemini import convert_messages_format

openai.api_type = "azure"

class Chat:
    function_names = []
    def __init__(self, chat_state_hash, state="empty"):

        if "chat_state_hash" not in st.session_state or chat_state_hash != st.session_state.chat_state_hash:
            # st.write("Initializing chat")
            st.session_state.chat_state_hash = chat_state_hash
            st.session_state.messages_to_display = []
            st.session_state.chat_state = state

        # Set session states as attributes for easier access
        self.messages_to_display = st.session_state.messages_to_display
        self.state = st.session_state.chat_state
    
    def instruction_messages(self):
        """
        Sets up the instructions to the agent. Should be overridden by subclasses.
        """
        return []

    def add_message(self, content, role="assistant", user_only=True, visible = True):
        """
        Used by app.py to start off the conversation with plots and descriptions.
        """
        message = {"role": role, "content": content}
        self.messages_to_display.append(message)

    def get_input(self):
        """
        Get input from streamlit."""
  
        if x := st.chat_input(placeholder=f"What else would you like to know?"):
            if len(x) > 500:
                st.error(f"Your message is too long ({len(x)} characters). Please keep it under 500 characters.")

            self.handle_input(x)
                

    def handle_input(self, input):
        """
        The main function that calls the GPT-4 API and processes the response.
        """

        # Get the instruction messages. 
        messages = self.instruction_messages()

        # Add a copy of the user messages. This is to give the assistant some context.
        messages = messages + self.messages_to_display.copy()

        # Get relevent information from the user input and then generate a response.
        # This is not added to messages_to_display as it is not a message from the assistant.
        get_relevant_info = self.get_relevant_info(input)

        # Now add the user input to the messages. Don't add system information and system messages to messages_to_display.
        self.messages_to_display.append({"role": "user", "content": input})
                         
        messages.append({"role": "user", "content": f"Here is the relevant information to answer the users query: {get_relevant_info}\n\n```User: {input}```"})

        # Remove all items in messages where content is not a string
        messages = [message for message in messages if isinstance(message["content"], str)]

        # Show the messages in an expander
        st.expander("GPT Messages", expanded=False).write(messages)  

        # Check if use gemini is set to true
        if USE_GEMINI:
            import google.generativeai as genai
            converted_msgs = convert_messages_format(messages)

            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name=GEMINI_CHAT_MODEL,
                system_instruction=converted_msgs["system_instruction"]
            )
            chat = model.start_chat(history=converted_msgs["history"])
            response = chat.send_message(content=converted_msgs["content"])

            answer = response.text
        else:
            # Call the GPT-4 API
            openai.api_base = GPT_BASE
            openai.api_version = GPT_VERSION
            openai.api_key = GPT_KEY

            response = openai.ChatCompletion.create(
                engine=GPT_ENGINE,
                messages=messages
                )
        
            answer=response['choices'][0]['message']['content']
        message = {"role": "assistant", "content": answer}
        
        # Add the returned value to the messages.
        self.messages_to_display.append(message)
   
    def display_content(self,content):
        """
        Displays the content of a message in streamlit. Handles plots, strings, and StreamingMessages.
        """
        if isinstance(content, str):
            st.write(content)

        # Visual
        elif isinstance(content, Visual):
            content.show()

        else:
            # So we do this in case
            try: content.show()
            except: 
                try: st.write(content.get_string())
                except:
                    raise ValueError(f"Message content of type {type(content)} not supported.")


    def display_messages(self):
        """
        Displays visible messages in streamlit. Messages are grouped by role.
        If message content is a Visual, it is displayed in a st.columns((1, 2, 1))[1].
        If the message is a list of strings/Visuals of length n, they are displayed in n columns. 
        If a message is a generator, it is displayed with st.write_stream
        Special case: If there are N Visuals in one message, followed by N messages/StreamingMessages in the next, they are paired up into the same N columns.
        """
        # Group by role so user name and avatar is only displayed once

        #st.write(self.messages_to_display)

        for key, group in groupby(self.messages_to_display, lambda x: x["role"]):
            group = list(group)

            if key == "assistant":
                avatar = "data/ressources/img/twelve_chat_logo.svg"
            else:
                try:
                    avatar = st.session_state.user_info["picture"]
                except:
                    avatar = None

            message=st.chat_message(name=key, avatar=avatar)   
            with message:
                for message in group:
                    content = message["content"]
                    self.display_content(content)
                

    def save_state(self):
        """
        Saves the conversation to session state.
        """
        st.session_state.messages_to_display = self.messages_to_display
        st.session_state.chat_state = self.state

class cpchat(chat):
    