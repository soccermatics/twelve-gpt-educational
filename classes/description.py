import math
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Optional

import pandas as pd
import tiktoken
import openai
import numpy as np

import utils.sentences as sentences
from utils.gemini import convert_messages_format
from classes.data_point import Player


from settings import USE_GEMINI

if USE_GEMINI:
    from settings import USE_GEMINI, GEMINI_API_KEY, GEMINI_CHAT_MODEL
else:
    from settings import GPT_BASE, GPT_VERSION, GPT_KEY, GPT_ENGINE

import streamlit as st

openai.api_type = "azure"


class Description(ABC):
    gpt_examples_base = "data/gpt_examples"
    describe_base = "data/describe"

    @property
    @abstractmethod
    def gpt_examples_path(self) -> str:
        """
        Path to excel files containing examples of user and assistant messages for the GPT to learn from.
        """

    @property
    @abstractmethod
    def describe_paths(self) -> Union[str, List[str]]:
        """
        List of paths to excel files containing questions and answers for the GPT to learn from.
        """

    def __init__(self):
        self.synthesized_text = self.synthesize_text()
        self.messages = self.setup_messages()

    def synthesize_text(self) -> str:
        """
        Return a data description that will be used to prompt GPT.

        Returns:
        str
        """

    def get_prompt_messages(self) -> List[Dict[str, str]]:
        """
        Return the prompt that the GPT will see before self.synthesized_text.

        Returns:
        List of dicts with keys "role" and "content".
        """

    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a data analysis bot. "
                    "You provide succinct and to the point explanations about data using data. "
                    "You use the information given to you from the data and answers "
                    "to earlier user/assistant pairs to give summaries of players."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about the data for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro


    def get_messages_from_excel(self,
        paths: Union[str, List[str]],
    ) -> List[Dict[str, str]]:
        """
        Turn an excel file containing user and assistant columns with str values into a list of dicts.

        Arguments:
        paths: str or list of str
            Path to the excel file containing the user and assistant columns.

        Returns:
        List of dicts with keys "role" and "content".

        """

        # Handle list and str paths arg
        if isinstance(paths, str):
            paths = [paths]
        elif len(paths) == 0:
            return []

        # Concatenate dfs read from paths
        df = pd.read_excel(paths[0])
        for path in paths[1:]:
            df = pd.concat([df, pd.read_excel(path)])

        if df.empty:
            return []

        # Convert to list of dicts
        messages = []
        for i, row in df.iterrows():
            if i == 0:
                messages.append({"role": "user", "content": row["user"]})
            else:
                messages.append({"role": "user", "content": row["user"]})
            messages.append({"role": "assistant", "content": row["assistant"]})

        return messages

    def setup_messages(self) -> List[Dict[str, str]]:
        messages = self.get_intro_messages()
        try:
            paths=self.describe_paths
            messages += self.get_messages_from_excel(paths)
        except FileNotFoundError as e:  # FIXME: When merging with new_training, add the other exception
            print(e)
        messages += self.get_prompt_messages()

        messages = [message for message in messages if isinstance(message["content"], str)]


        try:
            messages += self.get_messages_from_excel(
                paths=self.gpt_examples_path,
                
            )
        except FileNotFoundError as e:  # FIXME: When merging with new_training, add the other exception
            print(e)

        messages += [{"role": "user", "content": f"Now do the same thing with the following: ```{self.synthesized_text}```"}]
        return messages

    def stream_gpt(self, temperature=1):
        """
        Run the GPT model on the messages and stream the output.

        Arguments:
        temperature: optional float
            The temperature of the GPT model.
        
        Yields:
            str
        """

        st.expander("Description messages", expanded=False).write(self.messages)

        if USE_GEMINI:
            import google.generativeai as genai
            converted_msgs = convert_messages_format(self.messages)

            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name=GEMINI_CHAT_MODEL,
                system_instruction=converted_msgs["system_instruction"]
            )
            chat = model.start_chat(history=converted_msgs["history"])
            response = chat.send_message(content=converted_msgs["content"])

            answer = response.text
        else:
            # Use OpenAI API
            openai.api_base = GPT_BASE
            openai.api_version = GPT_VERSION
            openai.api_key = GPT_KEY

            response = openai.ChatCompletion.create(
                engine=GPT_ENGINE,
                messages=self.messages,
                temperature= temperature,
                )
        
            answer=response['choices'][0]['message']['content']

        return answer


class PlayerDescription(Description):

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/Forward.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/Forward.xlsx"]

    def __init__(self, player: Player):
        self.player = player
        super().__init__()


    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a UK-based football scout. "
                    "You provide succinct and to the point explanations about football players using data. "
                    "You use the information given to you from the data and answers "
                    "to earlier user/assistant pairs to give summaries of players."
                ),
            },
            {
                "role": "user",
                "content": "Do you refer to the game you are an expert in as soccer or football?",
            },
            {
                "role": "assistant",
                "content": (
                    "I refer to the game as football. "
                    "When I say football, I don't mean American football, I mean what Americans call soccer. "
                    "But I always talk about football, as people do in the United Kingdom."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about football for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def synthesize_text(self):

        player=self.player
        metrics = self.player.relevant_metrics
        description = f"Here is a statistical description of {player.name}, who played for {player.minutes_played} minutes as a {player.position}. \n\n "

        subject_p, object_p, possessive_p = sentences.pronouns(player.gender)
        
        for metric in metrics:

            description += f"{subject_p.capitalize()} was "
            description += sentences.describe_level(player.ser_metrics[metric +"_Z"]) 
            description += " in " + sentences.write_out_metric(metric)
            description += " compared to other players in the same playing position. "                            

        #st.write(description)

        return description

    def get_prompt_messages(self):
        prompt = (
            f"Please use the statistical description enclosed with ``` to give a concise, 4 sentence summary of the player's playing style, strengths and weaknesses. "
            f"The first sentence should use varied language to give an overview of the player. "
            "The second sentence should describe the player's specific strengths based on the metrics. "
            "The third sentence should describe aspects in which the player is average and/or weak based on the statistics. "
            "Finally, summarise exactly how the player compares to others in the same position. "
        )
        return [{"role": "user", "content": prompt}]


class TrolleyDescription(Description):

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/Trolley.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/TrolleyTree.xlsx"]

    def __init__(self, currentArguments,overallArgument,stance):
        self.currentArguments = currentArguments
        self.overallArgument = overallArgument
        self.stance = stance
        
        super().__init__()


    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
            "role": "system", "content": (
                "You are talking to a human user about the following thesis: " + self.overallArgument + ". "
                " You are currently arguing " + self.stance + " thesis."
                )
            },
            {
                "role": "user",
                "content": "Are you aggresive when you argue?",
            },
            {
                "role": "assistant",
                "content": (
                    "No. I am not aggresive when I argue. I try to make my arguments politely and with an ac"
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about the question we will discuss for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def synthesize_text(self):

        description = f"Here are some arguments {self.stance} the thesis: {self.overallArgument}. \n\n "

        for i,argument in self.currentArguments.iterrows():
            
            description += argument['user'] + ". "

        return description

    def get_prompt_messages(self):
        prompt = (
            f"Please use the information enclosed with ``` to give a concise, 2-3 sentence "
            f"argument {self.stance} the thesis that {self.overallArgument}."
            f"The first sentence should layout the strongest argument out of all of those provided {self.stance} the thesis. Then the remaining one or two sentences should support"
            f"your main point using arguments provided. Be forceful but polite and only outline your own argument, not objections to that argument. Only argue {self.stance} the thesis. Address the user directly. Do not give a prelude to what you are going to do or respond to this request with words like 'certainly'."
        )
        return [{"role": "user", "content": prompt}]
#_____________________________________________________________________________________________________________________________
class LessonDescription(Description):
    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/CProgramming.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/LessonTree.xlsx"]

    def __init__(self, currentState,topic,studentResponse):
        self.currentState = currentState
        self.topic = topic
        self.studentResponse = studentResponse
        
        super().__init__()


    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
            "role": "system", "content": (
                "You are an instructor bot teaching a human learner in a socratic way "
                " You instruct on  for loop topic."
                "Instead of providing a direct answer, guide the user into thinking and explaining the concepts about the topic"
                "Depeding on the user respose, generate questions to test if the user has understood the concept"
                )
            },
            {
                "role": "user",
                "content": "Are you fun when instracting?",
            },
            {
                "role": "assistant",
                "content": (
                    "I can be fun if you are fun to be thought, I don't expect to be giving you the answers, but I hope you learn the key concepts for loops in C "
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about the topic we will discuss for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def synthesize_text(self):

        message=pd.read_excel(self.gpt_examples_path)

        description = f"Here are some examples {message['user']} on how you can respond \n\n "

        for i,argument in message.iterrows():
            
            description += argument['assistant'] + ". "

        return description

    def get_prompt_messages(self):
        prompt = (
            f"Please use the information enclosed with ``` to give a concise, 2 sentence response to the user. "
            #f"instructions {self.topic} the the topic is {self.currentState}."
            f"The first sentence should be an appreciation of what the user has answered on {self.topic} topic. The next statement should be a question asking the user on concepts that build up to the current topic"
            f"your main response should be one question asking the user on knowledge based on the previous response"
            f"Assess the user understading of the topic based on their response, if there is a gap ask the user a question that will help them understand the gap in knowledge "
       
        )
        return [{"role": "user", "content": prompt}]
