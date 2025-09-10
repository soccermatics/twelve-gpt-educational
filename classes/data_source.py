from pandas.core.api import DataFrame as DataFrame
import streamlit as st
import requests
import pandas as pd
import numpy as np
import copy
import json
import datetime
from scipy.stats import zscore
import os 

from itertools import accumulate
from pathlib import Path
import sys
import pyarrow.parquet as pq


import classes.data_point as data_point
#from classes.wyscout_api import WyNot


# Base class for all data 
class Data():
    """
    Get, process, and manage various forms of data.
    """
    data_point_class = None

    def __init__(self):
        self.df = self.get_processed_data()

    def get_raw_data(self) -> pd.DataFrame:
        raise NotImplementedError("Child class must implement get_raw_data(self)")

    def process_data(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Child class must implement process_data(self, df_raw)")

    def get_processed_data(self):

        raw = self.get_raw_data()
        return self.process_data(raw)
    
    def select_and_filter(self, column_name, label):
        
        df = self.df
        selected_id = st.sidebar.selectbox(label, df[column_name].unique())
        self.df = df[df[column_name] == selected_id]


# Base class for stat related data sources
# Calculates zscores, ranks and pct_ranks
class Stats(Data):
    """
    Builds upon DataSource for data sources which have metrics and info
    """

    def __init__(self):
        # Dataframe specs:
        # df_info: index = player, columns = basic info
        # df_metrics: index = player/team_id, columns = multiindex (Raw, Z, Rank), (metrics)
        self.df = self.get_processed_data()
        self.metrics = []
        self.negative_metrics = []

    def get_metric_zscores(self,df):
 
        df_z = df.apply(zscore, nan_policy="omit")

        # Rename every column to include "Z" at the end
        df_z.columns = [f"{col}_Z" for col in df_z.columns]

        # Here we get opposite value of metrics if their weight is negative
        for metric in set(self.negative_metrics).intersection(self.metrics):
            df_z[metric] = df_z[metric] * -1
        return df_z

    def get_ranks(self,df):
        df_ranks = df.rank(ascending=False)

        # Rename every column to include "Ranks" at the end
        df_ranks.columns = [f"{col}_Ranks" for col in df_ranks.columns]

        return df_ranks

    def get_pct_ranks(self,df):
        df_pct = df.rank(pct=True) * 100
        # Rename every column to include "Pct_Ranks" at the end
        df_pct.columns = [f"{col}_Pct_Ranks" for col in df_pct.columns]

        return df_pct

    def calculate_statistics(self,metrics,negative_metrics=[]):
        self.metrics=metrics
        self.negative_metrics=negative_metrics
        
        df=self.df
        # Add zscores, rankings and qualities
        df_metric_zscores = self.get_metric_zscores(df[metrics])
        # Here we want to use df_metric_zscores to get the ranks and pct_ranks due to negative metrics
        df_metric_ranks = self.get_ranks(df[metrics])

        # Add ranks and pct_ranks as new columns
        self.df = pd.concat([df, df_metric_zscores, df_metric_ranks], axis=1)
        

class PlayerStats(Stats):
    data_point_class = data_point.Player
    # This can be used if some metrics are not good to perform, like tackles lost.
    negative_metrics = []

    def __init__(self, minimal_minutes=300):
        self.minimal_minutes = minimal_minutes

        super().__init__()


    def get_raw_data(self):

        df = pd.read_csv("data/events/Forwards.csv",encoding='unicode_escape')

        return df

    def process_data(self, df_raw):
        df_raw = df_raw.rename(columns={"shortName": "player_name"})

        df_raw = df_raw.replace({-1: np.nan})
        # Remove players with low minutes
        df_raw = df_raw[(df_raw.Minutes >= self.minimal_minutes)]

        if len(df_raw) < 10:  # Or else plots won't work
            raise Exception("Not enough players with enough minutes")

        return df_raw

    def to_data_point(self,gender,position) -> data_point.Player:
        
        id = self.df.index[0]

        #Reindexing dataframe
        self.df.reset_index(drop=True, inplace=True)

        name=self.df['player_name'][0]
        minutes_played=self.df['Minutes'][0]
        self.df=self.df.drop(columns=["player_name", "Minutes"])

        # Convert to series
        ser_metrics = self.df.squeeze()
        
        return self.data_point_class(id=id,name=name,minutes_played=minutes_played,gender=gender,position=position,ser_metrics=ser_metrics,relevant_metrics=self.metrics)

# Base class for an argument
# This class consists of a list of verbal arguments and where they fit into a larger discusion.
# Code 1.1 means the first argument at the top level. 
# These can either be pro or con arguments.
class Arguments(Data):
    """
    Builds upon DataSource for data sources which have metrics and info
    """

    def __init__(self):
        # Dataframe specs:
        # df_info: index = player, columns = basic info
        # df_metrics: index = player/team_id, columns = multiindex (Raw, Z, Rank), (metrics)
        self.df = self.get_processed_data()

    def get_raw_data(self):

        df = pd.read_csv("data/trolley/Trolley.csv",encoding='unicode_escape')

        return df

    def process_data(self, df_raw):

        # Assuming df is your DataFrame
        df = df_raw.sort_values('assistant')
        overall = []

        for _, row in df.iterrows():
            parts = row['assistant']
            category = row['category']

            opposite_dict={'Pro':'Con','Con':'Pro'}

            current_view =''
            for i in range(int(len(parts)/2)):
                prefix = parts[:i*2+2]

                
                new_view=df[df['assistant']==prefix]['category'].values[0]
                if new_view == 'Thesis':
                    current_view = new_view
                if current_view == 'Thesis':
                    current_view = new_view
                elif current_view == 'Con' and new_view == 'Pro':
                    current_view = 'Con'
                elif current_view == 'Con' and new_view == 'Con':
                    current_view = 'Pro'
                elif current_view == 'Pro' and new_view == 'Con':
                    current_view = 'Con'
                elif current_view == 'Pro' and new_view == 'Pro':
                    current_view = 'Pro'

            overall.append(current_view)  

        df['overall'] = overall

        return df

        
    def get_arguments(self, argument,stance):
        
        df = self.df
        
        # df['assistant'] contains the tree structure.
        # df['category'] contains the stance.
        
        # Find all rows of dateframe where df['assistant'] starts with 'argument'
        argument_df = df[df['assistant'].str.startswith(argument)]
        # Find all rows of dateframe where df['assistant'] is longer but no more than two characters longer than 'argument'
        argument_df = argument_df[argument_df['assistant'].str.len() <= len(argument)+2]
        argument_df = argument_df[argument_df['assistant'].str.len() > len(argument)]
        argument_df = argument_df[argument_df['category']==stance]
        # Unique list of all 'assistant' values
        list_of_arguments = argument_df['assistant'].unique()

        # For each argument in the list, find all rows of dateframe where df['assistant'] is within 2 and 4 and they are 'Pro
        for subargument in list_of_arguments:
            argument_df2 = df[df['assistant'].str.startswith(subargument)]
            argument_df2 = argument_df2[argument_df2['assistant'].str.len() <= len(argument)+4]
            argument_df2 = argument_df2[argument_df2['assistant'].str.len() > len(argument) + 2 ]
            argument_df2 = argument_df2[argument_df2['category']=='Pro']
            # Add these to the arguments.
            argument_df = pd.concat([argument_df,argument_df2])

        return argument_df
    #________________________________________________________________________________________________________

class Lesson(Data):
    
    #Builds upon DataSource for data sources which have metrics and info
    

    def __init__(self):
            # Dataframe specs:
            # df_info: index = player, columns = basic info
            # df_metrics: index = player/team_id, columns = multiindex (Raw, Z, Rank), (metrics)
         self.df = self.get_processed_data()

    def get_raw_data(self):
        
        df = pd.read_csv("data/CP programming agent/Cpprogramming.csv",encoding='unicode_escape')

        return df

    def process_data(self, df_raw):

            # Assuming df is your DataFrame
        df = df_raw.sort_values('step', ascending=False)
        overall = []

        for _, row in df.iterrows():
            parts = row[['step']]
            category = row['topic']
            current_view =''
            for i in range(int(len(parts)/2)):
                prefix = parts[:i*2+2]
                prefix=prefix.rstrip('')
                df['step'] = df['step'].astype(str)
                new_view=df[df['step']==prefix]['topic'].values[0]
                if new_view == 'Start':
                    current_view = new_view
                if current_view == 'Start':
                    current_view = new_view
                elif current_view == 'Defination' and new_view == 'Syntax':
                    current_view = 'Syntax'
                elif current_view == 'Syntax' and new_view == 'Defination':
                    current_view = 'Defination'
                elif current_view == 'Syntax' and new_view == 'Basic Implementation':
                    current_view = 'Basic Implementation'
                elif current_view == 'Basic Implementation' and new_view == 'loops with array':
                    current_view = 'loops with array'
                    #elif current_view == 'Pro' and new_view == 'Pro':
                       # current_view = 'Pro'


            overall.append(current_view)  

        df['overall'] = overall
       

        return df

            
    def get_arguments(self, argument,stance):
            
        df = self.df
            
        #df['step'] #contains the tree structure.
        #df['topic'] contains the topics.
            
        # Find all rows of dateframe where df['assistant'] starts with 'topic'
        argument_df = df[df['step'].str.startswith(argument)]
        # Find all rows of dateframe where df['assistant'] is longer but no more than two characters longer than 'argument'
        #argument_df = argument_df[argument_df['assistant'].str.len() <= len(argument)+2]
        argument_df = argument_df[argument_df['assistant'].str.len() > len(argument)]
        argument_df = argument_df[argument_df['assistant']==stance]
        # Unique list of all 'assistant' values
        list_of_arguments = argument_df['step'].unique()
        #st.write(list_of_arguments)
        # For each argument in the list, find all rows of dateframe where df['assistant'] is within 2 and 4 and they are 'Pro
        for subargument in list_of_arguments:
            argument_df2 = df[df['step'].str.startswith(subargument)]
            argument_df2 = argument_df2[argument_df2['step'].str.len() <= len(argument)+4]
            argument_df2 = argument_df2[argument_df2['step'].str.len() > len(argument) + 2 ]
            argument_df2 = argument_df2[argument_df2['topic']=='defination']
            # Add these to the arguments.
            argument_df = pd.concat([argument_df,argument_df2])
        #st.write(argument_df)

        return argument_df
        