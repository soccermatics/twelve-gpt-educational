import streamlit as st
import json
import openai
from itertools import groupby
from types import GeneratorType
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from classes.data_source import PersonalityStats
from classes.data_point import Person

from settings import GPT_BASE, GPT_VERSION, GPT_KEY, GPT_ENGINE


import utils.sentences as sentences

from utils.page_components import (add_common_page_elements)


sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()


def radarPlot(data_p):
    # Data import
    data_r = data_p.to_list()  
    labels = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    df = pd.DataFrame({'data': data_r,'label': labels})
    
    # Create the radar plot
    fig = px.line_polar(df, r='data', theta='label', line_close=True, markers=True)
    fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0, 40])),showlegend=True, title= 'Candidate profile')
    fig.update_traces(fill='toself', hoverinfo='r', marker=dict(size=5))
    # Display the plot in Streamlit
    st.plotly_chart(fig) 


def violin_and_point_plot(data, point_data):
    # Create a figure object
    fig = go.Figure()

    # Labels for the columns
    labels = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']

    # Loop through each label to add a violin plot trace
    for label in labels:
        fig.add_trace(go.Violin(
            x=data[label],  # Use x for the data
            name=label,      # Label each violin plot correctly
            box_visible=True,
            meanline_visible=True,
            line_color='black',  # Color of the violin outline
            fillcolor='rgba(0,100,200,0.3)',  # Color of the violin fill
            opacity=0.6,
            orientation='h'  # Set orientation to horizontal
        )
    )
    for label, value in point_data.items():
        fig.add_trace(
            go.Scatter(x=[value], y=[label], mode='markers', marker=dict(color='red', size=10, symbol='cross'), name=f'{label} Candidate Point'))

    # Update layout for better visualization
    fig.update_layout(
        title='Distribution of Personality Traits',
        xaxis_title='Score',  
        yaxis_title='Trait',
        xaxis=dict(range=[0, 40]),
        violinmode='overlay', 
        showlegend=True)

    # Display the plot in Streamlit
    st.plotly_chart(fig)


# Upload the dataset
data = pd.read_csv("data/events/dataset.csv",encoding='unicode_escape')
# Reduce the dataset
df = data.iloc[0:1000, -6:-1]
# Pick one candidate
data_p = df.iloc[0, -5:]

radarPlot(data_p)
violin_and_point_plot(df, data_p)
