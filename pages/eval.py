import streamlit as st
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path
import sys

from mplsoccer import Sbopen
import pandas as pd
import numpy as np
import json
import warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os
import random as rn
#warnings not visible on the course webpage
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')
import re
import utils.sentences as sentences

USE_GEMINI = True
if USE_GEMINI:
    from settings import USE_GEMINI, GEMINI_API_KEY, GEMINI_CHAT_MODEL
else:
    from settings import GPT_BASE, GPT_VERSION, GPT_KEY, GPT_ENGINE


from settings import USE_GEMINI

#setting random seeds so that the results are reproducible on the webpage
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(1)
rn.seed(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import pandas as pd
import numpy as np
import argparse
import tiktoken
import os
from utils.utils import normalize_text
import time
import google.api_core.exceptions

from classes.data_source import Shots
from classes.visual import ShotVisual, DistributionPlot, ShotContributionPlot
from classes.chat import Chat
from classes.description import ShotDescription

# Function to load and inject custom CSS from an external file
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


from utils.page_components import (
    add_common_page_elements
)


from classes.chat import PlayerChat

from utils.page_components import add_common_page_elements
from utils.utils import select_player, create_chat

sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()

st.markdown("## Shot commentator")


competitions = {
    "EURO Men 2024": "data/match_id_to_name_EURO_2024.json",
    "EURO Men 2020": "data/match_id_to_name_EURO_2020.json",
    "National Women's Soccer League (NWSL) 2018": "data/match_id_to_name_NWSL.json",
    "FIFA 2022": "data/match_id_to_name_FIFA_2022.json",
    "Women's Super League (FAWSL) 2017-18": "data/match_id_to_name_FAWSL.json",
    "Africa Cup of Nations (AFCON) 2023": "data/match_id_to_name_AFCON_2023.json"
}



# Define the function to generate descriptions
class GenerateDescriptions:
    def __init__(self, competition_file, selected_competition):
        with open(competition_file, 'r') as f:
            self.id_to_match_name = json.load(f)
        self.selected_competition = selected_competition

    def process_shots(self, shots, selected_match_id):
        """
        Generate descriptions for all shots with and without using the prompt.
        """
        results_with_prompt = []
        results_without_prompt = []
        feature_values = []

        shot_features = [
            'vertical_distance_to_center', 'euclidean_distance_to_goal', 'nearby_opponents_in_3_meters',  # Example shot features
            'opponents_in_triangle', 'goalkeeper_distance_to_goal', 'distance_to_nearest_opponent', 'angle_to_goalkeeper', 'shot_with_left_foot',
            'shot_after_throw_in', 'shot_after_corner', 'shot_after_free_kick'
        ]

        for shot_id in tqdm(shots.df_shots['id'], desc="Processing Shots"):
        #for shot_id in tqdm(shots.df_shots['id'].head(3), desc="Processing Shots"):

            contribution = shots.df_contributions.loc[shots.df_contributions['id'] == shot_id, 'euclidean_distance_to_goal_contribution'].values
            # Determine the contribution sign (positive, negative, or not contributing)
            if contribution.size > 0:
                if contribution[0] > 0.1:
                    contribution_sign = 'positive'
                elif contribution[0] < -0.1:   
                    contribution_sign = 'negative'
                else:
                    contribution_sign = 'not contributing'
            else:
                contribution_sign = 'not contributing'


            # Generate description with prompt
            to_hash = (selected_match_id, shot_id)
            descriptions_with_prompt = ShotDescription(shots, shot_id, self.selected_competition)
            retries = 3
            for attempt in range(retries):
                try:
                    summaries = descriptions_with_prompt.stream_gpt()
                    # Run the tests on the generated text
                    #test_1_result = self.run_test_1(summaries)
                    test_2_result = self.run_test_2(summaries, factor="euclidean distance to goal")

                    results_with_prompt.append({
                        "shot_id": shot_id,
                        "description": summaries,
                        #"test_1_score": test_1_result,  # Store the score
                        "test_2_result": test_2_result,
                        "ground_truth_contribution_sign": contribution_sign
                    })
                    #results_with_prompt.append(summaries)
                    break  # Exit loop if successful
                except google.api_core.exceptions.ResourceExhausted as e:
                    print(f"Rate limit exceeded, retrying in {2**attempt} seconds...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    if attempt == retries - 1:
                        print("Max retries reached. Skipping this shot.")
                    continue  # Try again if the exception is caught


            # Generate description without prompt
            descriptions_with_prompt.get_prompt_messages = lambda: []  # Disable prompt function
            descriptions_without_prompt_text = descriptions_with_prompt.synthesize_text()

            # Run the tests on the generated text
            #test_1_result_no_prompt = self.run_test_1(descriptions_without_prompt_text)
            test_2_result_no_prompt = self.run_test_2(descriptions_without_prompt_text, factor="euclidean distance to goal")

            results_without_prompt.append({
                "shot_id": shot_id,
                "description": descriptions_without_prompt_text,
                #"test_1_score": test_1_result_no_prompt,  # Store the score
                "test_2_result": test_2_result_no_prompt,
                "ground_truth_contribution_sign": contribution_sign
            })

            # Extract shot features for the current shot and append them to feature_values
            # shot_feature_values = shots.df_shots.loc[shots.df_shots['id'] == shot_id, shot_features].values.flatten()
            # feature_values.append(list(shot_feature_values) + [contribution_sign, test_2_result])



        # Create DataFrames
        df_with_prompt = pd.DataFrame(results_with_prompt)
        df_without_prompt = pd.DataFrame(results_without_prompt)
        # feature_columns = shot_features + ['ground_truth_contribution_sign', 'test_2_result']
        # feature_values_df = pd.DataFrame(feature_values, columns=feature_columns)

        return df_with_prompt, df_without_prompt
    

    def run_test_1(self, shot_description_text):
        """
        Rank the text for how interesting and engaging it is using the `stream_gpt` method.
        """
        
        test_message = f"Rank this text on a scale from 0 to 5 for how interesting and engaging it is:\n\n{shot_description_text}"

        retries = 5  # Increase the retry count
        for attempt in range(retries):
            try:
                response = self.stream_gpt(test_message)  # Generate response from stream_gpt

                if response:
                    match = re.search(r'\b(\d)\b', response.strip())  # Looks for a single digit (0-5)
                    if match:
                        return int(match.group(1))   # Return the extracted score
                    else:
                        print("Error: Could not extract a valid score from the response.")
                        return 0
                else:
                    print("Error: No response from stream_gpt.")
                    return "No valid description", 0  # Return default values if response is empty
                
            except google.api_core.exceptions.ResourceExhausted as e:
                print(f"Rate limit exceeded, retrying in {2**attempt} seconds...")
                time.sleep(2 ** attempt)  # Exponential backoff
                if attempt == retries - 1:
                    print("Max retries reached. Skipping this test.")

    def run_test_2(self, shot_description_text, factor):
        """
        Evaluate the impact of a specific factor (positive or negative) using the `stream_gpt` method.
        """
        test_message = (
            f"In the following text, was {factor} a positive, negative, or not contributing factor? "
            f"Respond with one of ['positive', 'negative', 'not contributing'].\n\n{shot_description_text}"
        )
        retries = 5  # Increase the retry count
        for attempt in range(retries):
            try:
                response = self.stream_gpt(test_message)  # Generate response from stream_gpt
                return response  # Return the response
            except google.api_core.exceptions.ResourceExhausted as e:
                print(f"Rate limit exceeded, retrying in {2**attempt} seconds...")
                time.sleep(2 ** attempt)  # Exponential backoff
                if attempt == retries - 1:
                    print("Max retries reached. Skipping this test.")

    def stream_gpt(self, text, temperature=1, USE_GEMINI = True):
        """
        Helper function to stream responses from Gemini or other model.
        """
        USE_GEMINI = True
        if USE_GEMINI:
            import google.generativeai as genai

            # Make sure you configure and get the responses from Gemini model.
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name=GEMINI_CHAT_MODEL,
                system_instruction="Provide a helpful response."
            )
            chat = model.start_chat()
            response = chat.send_message(content=text)

            return response.text
        else:
            # If using OpenAI or another model, adapt accordingly.
            return "response from other model"  # Replace with actual response handling



# Streamlit page for generating DataFrames
def generate_descriptions_page():
    st.title("Generate Descriptions for Shots")

    # Sidebar for competition and match selection
    selected_competition = st.sidebar.selectbox("Select a Competition", options=competitions.keys())
    file_path = competitions[selected_competition]

    with open(file_path, 'r') as f:
        id_to_match_name = json.load(f)

    selected_match_name = st.sidebar.selectbox(
        "Select a Match",
        options=id_to_match_name.values()
    )

    match_name_to_id = {v: k for k, v in id_to_match_name.items()}
    selected_match_id = match_name_to_id[selected_match_name]

    # Load shots for the selected match
    shots = Shots(selected_competition, selected_match_id)

    # Generate descriptions
    generator = GenerateDescriptions(file_path, selected_competition)
    df_with_prompt, df_without_prompt = generator.process_shots(shots, selected_match_id)


    # # Display DataFrames
    # st.subheader("Case 5: Feature Values")
    # feature_values_df['ground_truth_contribution_sign'] = feature_values_df['ground_truth_contribution_sign'].str.strip().str.lower()
    # feature_values_df['test_2_result'] = feature_values_df['test_2_result'].str.strip().str.lower()
    # feature_values_df['correct_prediction'] = feature_values_df['ground_truth_contribution_sign'] == feature_values_df['test_2_result']
    # st.dataframe(feature_values_df)
    # average_score = feature_values_df['test_1_score'].astype(float).mean()  # Ensure it's float for accurate calculation
    # st.subheader("Average Test 1 Score for Descriptions with Prompt")
    # st.write(f"The average score from Test 1 is: {average_score:.2f}")

    # # Calculate accuracy
    # accuracy = feature_values_df['correct_prediction'].mean() * 100  # Percentage of correct predictions


    # Display DataFrames
    st.subheader("Case 3 and 4: LLM text with and without examples")
    df_with_prompt['ground_truth_contribution_sign'] = df_with_prompt['ground_truth_contribution_sign'].str.strip().str.lower()
    df_with_prompt['test_2_result'] = df_with_prompt['test_2_result'].str.strip().str.lower()
    df_with_prompt['correct_prediction'] = df_with_prompt['ground_truth_contribution_sign'] == df_with_prompt['test_2_result']
    st.dataframe(df_with_prompt)
    #average_score = df_with_prompt['test_1_score'].astype(float).mean()  # Ensure it's float for accurate calculation
    #st.subheader("Average Test 1 Score for Descriptions with Prompt")
    #st.write(f"The average score from Test 1 is: {average_score:.2f}")

    # Calculate accuracy
    accuracy = df_with_prompt['correct_prediction'].mean() * 100  # Percentage of correct predictions

    # Display accuracy
    st.subheader("Accuracy of Test 2 Results")
    st.write(f"The accuracy of Test 2 (ground truth vs prediction) is: {accuracy:.2f}%")

    # st.subheader("Case 1 and 2: Synthetized text with and without contributions")
    # df_without_prompt['ground_truth_contribution_sign'] = df_without_prompt['ground_truth_contribution_sign'].str.strip().str.lower()
    # df_without_prompt['test_2_result'] = df_without_prompt['test_2_result'].str.strip().str.lower()
    # df_without_prompt['correct_prediction'] = df_without_prompt['ground_truth_contribution_sign'] == df_without_prompt['test_2_result']
    # st.dataframe(df_without_prompt)
    # #average_score_no_prompt = df_without_prompt['test_1_score'].astype(float).mean()  # Ensure it's float for accurate calculation
    # #st.subheader("Average Test 1 Score for Descriptions without Prompt")
    # #st.write(f"The average score from Test 1 is: {average_score_no_prompt:.2f}")
    # # Calculate accuracy
    # accuracy = df_without_prompt['correct_prediction'].mean() * 100  # Percentage of correct predictions

    # # Display accuracy
    # st.subheader("Accuracy of Test 2 Results")
    # st.write(f"The accuracy of Test 2 (ground truth vs prediction) is: {accuracy:.2f}%")

    # Option to download results as CSV
    st.download_button(
        label="Download Descriptions with Prompt",
        data=df_with_prompt.to_csv(index=False).encode('utf-8'),
        file_name=f"descriptions_with_prompt_{selected_match_name}.csv",
        mime="text/csv"
    )

    # st.download_button(
    #     label="Download Descriptions without Prompt",
    #     data=df_without_prompt.to_csv(index=False).encode('utf-8'),
    #     file_name=f"descriptions_without_prompt_{selected_match_name}.csv",
    #     mime="text/csv"
    # )
    # st.download_button(
    #     label="Download Descriptions Feature Values",
    #     data=feature_values_df.to_csv(index=False).encode('utf-8'),
    #     file_name=f"feature_values_{selected_match_name}.csv",
    #     mime="text/csv"
    # )

# Add this page to your Streamlit app's navigation
if __name__ == "__main__":
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Generate Descriptions"])

    if page == "Generate Descriptions":
        generate_descriptions_page()