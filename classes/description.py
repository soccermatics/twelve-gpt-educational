import math
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Optional

import pandas as pd
import tiktoken
import openai
import numpy as np

import utils.sentences as sentences
from classes.data_point import Player, Person
from classes.data_source import PersonStat

from settings import GPT_BASE, GPT_VERSION, GPT_KEY, GPT_ENGINE, GPT_DEFAULT

import streamlit as st
import random
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


    def get_messages_from_excel(self, paths: Union[str, List[str]],) -> List[Dict[str, str]]:
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
        openai.api_base = GPT_BASE
        openai.api_version = GPT_VERSION
        openai.api_key = GPT_KEY

        st.expander("Description messages", expanded=False).write(self.messages)

        response = openai.ChatCompletion.create(
            engine=GPT_ENGINE,
            messages=self.messages,
            temperature= temperature,
            )
    
        answer=response['choices'][0]['message']['content']

        return answer


class PlayerDescription(Description):
    output_token_limit = 150

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


# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------


class PersonDescription():
    output_token_limit = 150

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/Forward.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/Forward.xlsx"]

    def __init__(self, person: Person):
        self.person = person
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
                    "You are a recruiter. "
                    "You use the information given to you from the data and answers "
                    "to earlier user/assistant pairs to give summaries of candidates."
                ),
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
                    "content": "First, could you answer some questions about a candidate for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro



    def categorie_description(self, value):
        text = ''
        if value <= -2:
            text = 'The candidate is extremely '
        elif (value < -2) & (value <= -1):
            text = 'The candidate is very '
        elif (value > -1) & (value <= -0.5):
            text = 'The candidate is quite '
        elif (value > -0.5) & (value <= 0.5):
            text = 'The candidate is relatively '
        elif (value > 0.5) & (value <= 1):
            text = 'The candidate is quite'
        elif (value > 1) & (value <= 2):
            text = 'The candidate is very '
        elif value > 2:
            text = 'The candidate is extremely '    
        return text

    def all_max_indices(self, row):
        max_value = row.max()
        return list(row[row == max_value].index)

    def all_min_indices(self, row):
        min_value = row.min()
        return list(row[row == min_value].index)



    def get_description(self, person):
        # here we need the dataset to check the min and max score of the person
        
        person_metrics = person.ser_metrics
        person_stat = PersonStat()
        questions = person_stat.get_questions()

        
        name = person.name
        extraversion = person_metrics['extraversion']
        neuroticism = person_metrics['neuroticism']
        agreeableness = person_metrics['agreeableness']
        conscientiousness = person_metrics['conscientiousness']
        openness = person_metrics['openness']


   

      
        
        text = []

            
        # extraversion
        cat_0 = 'solitary and reserved. '
        cat_1 = 'outgoing and energetic. '
                
        if extraversion > 0:
            text_t = self.categorie_description(extraversion) + cat_1
            if extraversion > 1:
                index_max = person_metrics[0:10].idxmax()
                text_2 = 'In particular they said that ' + questions[index_max][0]+'. '
                text_t +=  text_2
        else:
            text_t = self.categorie_description(extraversion) + cat_0
            if extraversion < -1:
                index_min = person_metrics[0:10].idxmin()
                text_2 = 'In particular they said that ' + questions[index_min][0]+'. '
                text_t += text_2
        text.append(text_t)
            
        # neuroticism
        cat_0 = 'resilient and confident. '
        cat_1 = 'sensitive and nervous. '
            
        if neuroticism > 0:
            text_t = self.categorie_description(neuroticism) + cat_1  \
                    + 'The candidate tends to feel more negative emotions, anxiety. '
            if neuroticism > 1:
                index_max = person_metrics[10:20].idxmax()
                text_2 = 'In particular they said that ' + questions[index_max][0]+'. '
                text_t += text_2
                
        else:
            text_t = self.categorie_description(neuroticism) + cat_0  \
                    + 'The candidate tends to feel less negative emotions, anxiety. '
            if neuroticism < -1:
                index_min = person_metrics[10:20].idxmin()
                text_2 = 'In particular they said that ' + questions[index_min][0]+'. '
                text_t += text_2
        text.append(text_t)
            
        # agreeableness  
        cat_0 = 'critical and rational. '
        cat_1 = 'friendly and compassionate. '
            
        if agreeableness > 0:
            text_t = self.categorie_description(agreeableness) + cat_1  \
                    + 'The candidate tends to be more cooperative, polite, kind and friendly. '
            if agreeableness > 1:
                index_max = person_metrics[20:30].idxmax()
                text_2 = 'In particular they said that ' + questions[index_max][0] +'. '
                text_t += text_2

        else:
            text_t = self.categorie_description(agreeableness) + cat_0  \
                    + 'The candidate tends to be less cooperative, polite, kind and friendly. '
            if agreeableness < -1:
                index_min = person_metrics[20:30].idxmin()
                text_2 = 'In particular they said that ' + questions[index_min][0] +'. '
                text_t += text_2
        text.append(text_t)
       
       # conscientiousness
        cat_0 = 'extravagant and careless. '
        cat_1 = 'efficient and organized. '
            
        if conscientiousness > 0:
            text_t = self.categorie_description(conscientiousness) + cat_1  \
                    + 'The candidate tends to be more careful or diligent. '
            if conscientiousness > 1:
                index_max = person_metrics[30:40].idxmax()
                text_2 = 'In particular they said that ' + questions[index_max][0] +'. '
                text_t += text_2
        else:
            text_t = self.categorie_description(conscientiousness) + cat_0  \
                    + 'The candidate tends to be less careful or diligent. '
            if conscientiousness < -1:
                index_min = person_metrics[30:40].idxmin()
                text_2 = 'In particular they said that ' + questions[index_min][0] +'. '
                text_t += text_2
        text.append(text_t)
        
        # openness
        cat_0 = 'consistent and cautious. '
        cat_1 = 'inventive and curious. '

        if openness > 0:
            text_t = self.categorie_description(openness) + cat_1  \
                    + 'The candidate tends to be more open. '
            if openness > 1:
                index_max = person_metrics[40:50].idxmax()
                text_2 = 'In particular they said that ' + questions[index_max][0] +'. '
                text_t += text_2
        else:
            text_t = self.categorie_description(openness) + cat_0  \
                    + 'The candidate tends to be less open. '
            if openness < -1:
                index_min = person_metrics[40:50].idxmin()
                text_2 = 'In particular they said that ' + questions[index_min][0] +'. '
                text_t += text_2
        text.append(text_t)
        
        text = ''.join(text)
        text = text.replace(',','')
        return text

    
    def get_prompt_messages(self):
        prompt = (
            f"Please use the statistical description enclosed with ``` to give a concise, 4 sentence summary of the player's playing style, strengths and weaknesses. "
            f"The first sentence should use varied language to give an overview of the player. "
            "The second sentence should describe the player's specific strengths based on the metrics. "
            "The third sentence should describe aspects in which the player is average and/or weak based on the statistics. "
            "Finally, summarise exactly how the player compares to others in the same position. "
        )
        return [{"role": "user", "content": prompt}]
