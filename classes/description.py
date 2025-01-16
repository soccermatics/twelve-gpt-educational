import math
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Optional

import pandas as pd
import tiktoken
import openai
import numpy as np

import utils.sentences as sentences
from utils.gemini import convert_messages_format
from classes.data_point import Player, Country


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

    def get_messages_from_excel(
        self,
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
            paths = self.describe_paths
            messages += self.get_messages_from_excel(paths)
        except (
            FileNotFoundError
        ) as e:  # FIXME: When merging with new_training, add the other exception
            print(e)
        
        # Ensure messages are in the correct format after getting from excel
        messages = [msg for msg in messages if isinstance(msg, dict) and "content" in msg and isinstance(msg["content"], str)]
        
        messages += self.get_prompt_messages()  # Adding prompt messages
        
        try:
            messages += self.get_messages_from_excel(paths=self.gpt_examples_path)
        except FileNotFoundError as e:  # FIXME: When merging with new_training, add the other exception
            print(e)

        # Ensure that synthesized_text is defined and has a string value
        synthesized_text = getattr(self, 'synthesized_text', '')
        if isinstance(synthesized_text, str):
            messages.append({"role": "user", "content": f"Now do the same thing with the following: ```{synthesized_text}```"})

        # Filter again to ensure no non-string content is present
        messages = [msg for msg in messages if isinstance(msg, dict) and "content" in msg and isinstance(msg["content"], str)]
        
        messages += self.get_prompt_messages()

        messages = [
            message for message in messages if isinstance(message["content"], str)
        ]

        try:
            messages += self.get_messages_from_excel(
                paths=self.gpt_examples_path,
            )
        except (
            FileNotFoundError
        ) as e:  # FIXME: When merging with new_training, add the other exception
            print(e)

        messages += [
            {
                "role": "user",
                "content": f"Now do the same thing with the following: ```{self.synthesized_text}```",
            }
        ]
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
        st.expander("Chat transcript", expanded=False).write(self.messages)

        if USE_GEMINI:
            import google.generativeai as genai

            converted_msgs = convert_messages_format(self.messages)

            # # save converted messages to json
            # import json
            # with open("data/wvs/msgs_0.json", "w") as f:
            #     json.dump(converted_msgs, f)

            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name=GEMINI_CHAT_MODEL,
                system_instruction=converted_msgs["system_instruction"],
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
                temperature=temperature,
            )

            answer = response["choices"][0]["message"]["content"]

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

        player = self.player
        metrics = self.player.relevant_metrics
        description = f"Here is a statistical description of {player.name}, who played for {player.minutes_played} minutes as a {player.position}. \n\n "

        subject_p, object_p, possessive_p = sentences.pronouns(player.gender)

        for metric in metrics:

            description += f"{subject_p.capitalize()} was "
            description += sentences.describe_level(player.ser_metrics[metric + "_Z"])
            description += " in " + sentences.write_out_metric(metric)
            description += " compared to other players in the same playing position. "

        # st.write(description)

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





class ShotDescription(Description):

    output_token_limit = 500

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/action/shots.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/action/shots.xlsx"]
    
    def __init__(self, shots, shot_id, competition):
        self.shots = shots
        self.shot_id = shot_id
        self.competition = competition
        super().__init__()

    def synthesize_text(self):

        shots = self.shots
        shot_data = shots.df_shots[shots.df_shots['id'] == self.shot_id]  # Fix here to use self.shot_id

        if shot_data.empty:
            raise ValueError(f"No shot found with ID {self.shot_id}")
        
        player_name = shot_data['player_name'].iloc[0]
        team_name = shot_data['team_name'].iloc[0]

        start_x = shot_data['start_x'].iloc[0]
        start_y = shot_data['start_y'].iloc[0]
        xG = shot_data['xG'].iloc[0]
        goal_status = shot_data['goal'].fillna(False).iloc[0]
        
        # Map goal boolean to readable category
        labels = {False: "didn't result in a goal.", True: 'was a goal!'}
        goal_status_text = labels[goal_status]
        #angle_to_goal = shot_data['angle_to_goal'].iloc[0]
        distance_to_goal = shot_data['euclidean_distance_to_goal'].iloc[0]
        distance_to_nearest_opponent = shot_data['distance_to_nearest_opponent'].iloc[0]
        gk_dist_to_goal = shot_data['goalkeeper_distance_to_goal'].iloc[0]
        minute= shot_data['minute'].iloc[0]

        # Give a detailed description of the contributions to the shot
        shot_contributions = self.shots.df_contributions[self.shots.df_contributions['id'] == self.shot_id]

        shot_features = {
            'vertical_distance_to_center': shot_data['vertical_distance_to_center'].iloc[0],
            'euclidean_distance_to_goal': distance_to_goal,
            'nearby_opponents_in_3_meters': shot_data['nearby_opponents_in_3_meters'].iloc[0],
            'opponents_in_triangle': shot_data['opponents_in_triangle'].iloc[0],
            'goalkeeper_distance_to_goal': gk_dist_to_goal,
            #'header': shot_data['header'].iloc[0],
            'distance_to_nearest_opponent': distance_to_nearest_opponent,
            'angle_to_goalkeeper': shot_data['angle_to_goalkeeper'].iloc[0],
            'shot_with_left_foot': shot_data['shot_with_left_foot'].iloc[0],
            'shot_after_throw_in': shot_data['shot_after_throw_in'].iloc[0],
            'shot_after_corner': shot_data['shot_after_corner'].iloc[0],
            'shot_after_free_kick': shot_data['shot_after_free_kick'].iloc[0],
            'shot_during_regular_play': shot_data['shot_during_regular_play'].iloc[0],
            'pattern': shot_data['play_pattern_name'].iloc[0],
        }

        feature_descriptions = sentences.describe_shot_features(shot_features, self.competition)


        shot_description = (
            f"{player_name}'s shot from {team_name} {goal_status_text} "
            f"This shot had an xG value of {xG:.2f}, which means that we estimate the chance of scoring from this situation as {xG * 100:.0f}%. "
            f"{sentences.describe_xg(xG)} "
            #f"The distance to goal was {distance_to_goal:.1f} meters and the distance to the nearest opponent was {distance_to_nearest_opponent:.1f} meters."
        )
        shot_description += '\n'.join(feature_descriptions) + '\n'  # Add the detailed descriptions of the shot features

        shot_description += '\n' + sentences.describe_shot_contributions(shot_contributions, shot_features)

        with st.expander("Synthesized Text"):
            st.write(shot_description)
        
        return shot_description 
    

    def get_prompt_messages(self):
        prompt = (
            "You are a football commentator. You should write in an exciting and engaging way about a shot"
            f"You should giva a four sentence summary of the shot taken by the player. "
            "The first sentence should say whether it was a good chance or not, state the expected goals value and also state if it was a goal. "
            "The second and third sentences should describe the most important factors that contributed to the quality of the chance. "
            "If it was a good chance these two sentences chould explain what contributing factors made the shot dangerous. "
            "If it wasn't particularly good chance then these two sentences chould explain why it wasn't a good chance. "
            "Depedning on the quality of the chance, the final sentence should either praise the player or offer advice about what to think about when shooting."
            )
        return [{"role": "user", "content": prompt}]



class CountryDescription(Description):
    output_token_limit = 150

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/WVS_examples.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/WVS_qualities.xlsx"]

    def __init__(self, country: Country, description_dict, thresholds_dict):
        self.country = country
        self.description_dict = description_dict
        self.thresholds_dict = thresholds_dict

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
                    "You are a data analyst and a social scientist. "
                    "You provide succinct and to the point explanations about countries using metrics derived from data collected in the World Value Survey. "
                    "You use the information given to you from the data and answers to earlier questions to give summaries of how countries score in various metrics that attempt to measure the social values held by the population of that country."
                ),
            },
            # {
            #     "role": "user",
            #     "content": "Do you refer to the game you are an expert in as soccer or football?",
            # },
            # {
            #     "role": "assistant",
            #     "content": (
            #         "I refer to the game as football. "
            #         "When I say football, I don't mean American football, I mean what Americans call soccer. "
            #         "But I always talk about football, as people do in the United Kingdom."
            #     ),
            # },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about a the World Value Survey for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def synthesize_text(self):

        country = self.country
        metrics = self.country.relevant_metrics
        description = f"Here is a statistical description of the core values of {country.name.capitalize()}. \n\n"

        # subject_p, object_p, possessive_p = sentences.pronouns(country.gender)

        for metric in metrics:

            # # TODO: customize this text?
            # description += f"{country.name.capitalize()} was found to be "
            # description += sentences.describe_level(
            #     country.ser_metrics[metric + "_Z"],
            #     thresholds=self.thresholds_dict[metric],
            #     words=self.description_dict[metric],
            # )
            # description += " in " + metric.lower()  # .replace("_", " ")
            # description += " compared to other countries in the same survey. "

            description += f"{country.name.capitalize()} was found to "
            description += sentences.describe_level(
                country.ser_metrics[metric + "_Z"],
                thresholds=self.thresholds_dict[metric],
                words=self.description_dict[metric],
            )
            description += " compared to other countries in the same survey. "

        # st.write(description)

        return description

    def get_prompt_messages(self):
        prompt = (
            f"Please use the statistical description enclosed with ``` to give a concise, 4 sentence summary of the social values held by population of the country. "
            # f"The first sentence should use varied language to give an overview of the player. "
            # "The second sentence should describe the player's specific strengths based on the metrics. "
            # "The third sentence should describe aspects in which the player is average and/or weak based on the statistics. "
            # "Finally, summarise exactly how the player compares to others in the same position. "
        )
        return [{"role": "user", "content": prompt}]
