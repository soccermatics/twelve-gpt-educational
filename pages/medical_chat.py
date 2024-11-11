"""
Entrypoint for streamlit app.
Runs top to bottom every time the user interacts with the app (other than imports and cached functions).
"""

# Library imports
import traceback
import copy
import pandas as pd
import json

import streamlit as st

from classes.chat import ModelChat
from classes.data_source import Model
from classes.data_point import Individual
from classes.train_model import TrainModel
from classes.visual import DistributionModelPlot
from classes.description import (
    IndividualDescription,
)
from classes.chat import PlayerChat

from utils.page_components import (
    add_common_page_elements,
    select_individual,
    create_chat,
)

sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

# Read in model card text
#with open("model cards/model-card-football-scout.md", 'r') as file:

model_card_text = "We need to write this."

st.expander("Model card", expanded=False).markdown(model_card_text)
global data, model_features
# data=None
# model_features= None

def setup_model(train=False):
    st.write("Upload a CSV file to use as the data source.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="data_file")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Upload a CSV file to explaining the features.")
        feature_file = st.file_uploader("Choose a CSV file", type="csv", key="feature_file")
        if feature_file is not None:
            feature_data = pd.read_csv(feature_file)
            if not train:
                st.write("Upload a CSV file of the weights (coefficients) found in your model")
                weights_file = st.file_uploader("Choose a CSV file", type="csv", key="weight_file")
                if weights_file is not None:
                    weights_data = pd.read_csv(weights_file)
                    parameter_explanations = {row['Parameter']: row['Explanation'] for _, row in feature_data.iterrows()}
                    weights_data['Explanation'] = weights_data['Parameter'].map(parameter_explanations)
                    st.write("Feature Explanation:", weights_data)
                    return (data, weights_data)
            else:
                categorical_interpretations=None
                has_categorical = st.radio("Does your data have categorical features?", ("Yes", "No"))
                if has_categorical == "Yes":
                    st.write("Upload a JSON file detailing the interpretations of the categorical data.")
                    json_file = st.file_uploader("Choose a JSON file", type="json", key="category_json_file")
                    if json_file is not None:
                        categorical_interpretations = json.load(json_file)
                        # st.write("Categorical Interpretations:", categorical_interpretations)
                if has_categorical == "No" or categorical_interpretations is not None:
                    st.write("Choose target column")
                    target = st.radio("Target column", data.columns)
                    features = [col for col in data.columns if col != target]
                    # model=TrainModel(data, target, features)
                    try:
                        model=TrainModel(data, target, features)
                    except Exception as e:
                        st.error(f"An error occurred while training the model: {e} Pick a traget column that is binary.")
                        return
                    # model.selectFeatures()
                    # merge explantions with coef_df on matching feature names
                    coef_explanations = {row['Parameter']: row['Explanation'] for _, row in feature_data.iterrows()}
                    model.coef_df['Explanation'] = model.coef_df['Parameter'].map(coef_explanations)
                    st.write("Training Output:", model.coef_df)
                    # Keep only the Intercept, target, and parameters in model.coef_df in data
                    columns_to_keep = [target] + [param for param in model.coef_df['Parameter'].tolist() if param in data.columns]
                    data = data[columns_to_keep]
                    data = data.head(200)
                    if categorical_interpretations is not None:
                        return (data, model.coef_df, categorical_interpretations, target)
                    else:
                        return (data, model.coef_df)
    
def setup_chat(data, model_features, categorical_interpretations=None, target=None):
    model=Model()
    model.set_data(data, model_features)
    model.process_data()
    model.weight_contributions()
    st.write("Dataframe used:", model.df)
    
    # # Now select the focal player
    columns = ["ID", target]
    individual = select_individual(sidebar_container, model, columns=columns)
    thresholds = model.most_variable_data()            
    # st.write("Feature Contributions Variance")
    # st.write(model.std_contributions.sort_values(ascending=False))
    st.write("The most variable feature is",model.std_contributions.idxmax())
    st.write("The thresholds value are",str(thresholds))

    # Chat state hash determines whether or not we should load a new chat or continue an old one
    # We can add or remove variables to this hash to change conditions for loading a new chat
    to_hash = (individual.id,)
    # Now create the chat as type PlayerChat
    chat = create_chat(to_hash, ModelChat, individual, model)

    metrics =  model.parameters['Parameter']

    # Now we want to add basic content to chat if it's empty
    if chat.state == "empty":

        # Make a plot of the distribution of the metrics for all players
        # We reverse the order of the elements in metrics for plotting (because they plot from bottom to top)
        visual = DistributionModelPlot(thresholds,metrics, model_features=model_features)
        visual.add_title('Evaluation of individual','')
        visual.add_individuals(model, metrics=metrics, target=target)
        visual.add_individual(individual, len(model.df), metrics=metrics)

        # Now call the description class to get the summary of the player
        description = IndividualDescription(individual,metrics,parameter_explanation=model.parameter_explanation, categorical_interpretations= categorical_interpretations , thresholds = [10, 5, 2, -2,-5,-10])
        summary = description.stream_gpt()

        # Add the visual and summary to the chat
        chat.add_message(
            "Please can you summarise this individual for me?",
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
st.divider()
# Ask the user whether they want to train a model or use an existing trained model
model_option = st.radio("Do you want to train a new model or use an existing trained model?", ("Train a new model", "Use an existing trained model"))

if model_option == "Train a new model":
    st.write("You chose to train a new model. Please upload the raw data, and the explanations of the parameters in the data in CSV format.")
    # trainModel()
    result = setup_model(train=True)
    if result is not None:
        data, model_features, categorical_interpretations, target = result
        setup_chat(data, model_features, categorical_interpretations, target)
    
else:
    result = setup_model(train=False)
    if result is not None:
        data, model_features = result
        setup_chat(data, model_features)
    # data, model_features= setup_model(train=False)
    # setup_chat(data, model_features)



