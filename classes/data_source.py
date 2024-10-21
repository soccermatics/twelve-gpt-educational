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




class PersonStat(Stats):   
    data_point_class = data_point.Person
    negative_metrics = []

    def __init__(self):
        super().__init__()
    
    def get_raw_data(self):
        #df = pd.read_csv('data/data-final.csv',sep='\t',encoding='unicode_escape').sample(frac=0.0001)
        df = pd.read_csv('data/data_raw.csv',sep='\t',encoding='unicode_escape')
        return df
      

    def get_questions(self):
        ''' This function is to have access to the questions'''
        
        # Groups and Questions modify version
        # (1) extraversion, (2) neuroticism, (3) agreeableness, (4)conscientiousness , and (5) openness
        ext_questions = {'EXT1' : ['they are the life of the party',1],
                         'EXT2' : ['they dont talk a lot',-1],
                         'EXT3' : ['they feel comfortable around people',1],
                         'EXT4' : ['they keep in the background',-1],
                         'EXT5' : ['they start conversations',1],
                         'EXT6' : ['they have little to say',-1],
                         'EXT7' : ['they talk to a lot of different people at parties',1],
                         'EXT8' : ['they dont like to draw attention to themself',-1],
                         'EXT9' : ['they dont mind being the center of attention',1],
                         'EXT10': ['they are quiet around strangers',-1]}
        
        est_questions = {'EST1' : ['they get stressed out easily',-1],
                         'EST2' : ['they are relaxed most of the time',1],
                         'EST3' : ['they worry about things',-1],
                         'EST4' : ['they seldom feel blue',1],
                         'EST5' : ['they are easily disturbed',-1],
                         'EST6' : ['they get upset easily',-1],
                         'EST7' : ['they change their mood a lot',-1],
                         'EST8' : ['they have frequent mood swings',-1],
                         'EST9' : ['they get irritated easily',-1],
                         'EST10': ['they often feel blue',-1]}
        
        agr_questions = {'AGR1' : ['they feel little concern for others',-1],
                         'AGR2' : ['they interested in people',1],
                         'AGR3' : ['they insult people',-1],
                         'AGR4' : ['they sympathize with others feelings',1],
                         'AGR5' : ['they are not interested in other peoples problems',-1],
                         'AGR6' : ['they have a soft heart',1],
                         'AGR7' : ['they not really interested in others',-1],
                         'AGR8' : ['they take time out for others',1],
                         'AGR9' : ['they feel others emotions',1],
                         'AGR10': ['they make people feel at ease',1]}
    
        csn_questions = {'CSN1' : ['they are always prepared',1],
                         'CSN2' : ['they leave their belongings around',-1],
                         'CSN3' : ['they pay attention to details',1],
                         'CSN4' : ['they make a mess of things',-1],
                         'CSN5' : ['they get chores done right away',1],
                         'CSN6' : ['they often forget to put things back in their proper place',-1],
                         'CSN7' : ['they like order',1],
                         'CSN8' : ['they shirk their duties',-1],
                         'CSN9' : ['they follow a schedule',1],
                         'CSN10' : ['they are exacting in their work',1]}

        opn_questions = {'OPN1' : ['they have a rich vocabulary',1],
                         'OPN2' : ['they have difficulty understanding abstract ideas',-1],
                         'OPN3' : ['they have a vivid imagination',1],
                         'OPN4' : ['they are not interested in abstract ideas',-1],
                         'OPN5' : ['they have excellent ideas',1],
                         'OPN6' : ['they do not have a good imagination',-1],
                         'OPN7' : ['they are quick to understand things',1],
                         'OPN8' : ['they use difficult words',1],
                         'OPN9' : ['they spend time reflecting on things',1],
                         'OPN10': ['they are full of ideas',1]}
    
        questions = ext_questions | est_questions | agr_questions | csn_questions  | opn_questions
        return questions


    
        
    def process_data(self, df_raw):
        ''' This fonction get the person or candidate data with a number id or a list, and return a dataframe of the person '''
        questions = self.get_questions()
        
        # First we want to check if the user want a certain candidate from the dataset 
        # or if the user did the test so it return a list
        if isinstance(df_raw, list):
            matching = ['EXT1', 'EXT2', 'EXT3', 'EXT4', 'EXT5', 'EXT6', 'EXT7', 'EXT8', 'EXT9',
           'EXT10', 'EST1', 'EST2', 'EST3', 'EST4', 'EST5', 'EST6', 'EST7', 'EST8',
           'EST9', 'EST10', 'AGR1', 'AGR2', 'AGR3', 'AGR4', 'AGR5', 'AGR6', 'AGR7',
           'AGR8', 'AGR9', 'AGR10', 'CSN1', 'CSN2', 'CSN3', 'CSN4', 'CSN5', 'CSN6',
           'CSN7', 'CSN8', 'CSN9', 'CSN10', 'OPN1', 'OPN2', 'OPN3', 'OPN4', 'OPN5',
           'OPN6', 'OPN7', 'OPN8', 'OPN9', 'OPN10']
            df_raw =pd.DataFrame([df_raw], columns= [column for column in matching])
            
            
        else:
            df_raw.drop(df_raw.columns[50:107], axis=1, inplace=True)
            df_raw.drop(df_raw.columns[50:], axis=1, inplace=True) # here 50 to remove the country
            df_raw.dropna(inplace=True)

           

            # Group Names and Columns
            #EXT = [column for column in df_raw if column.startswith('EXT')]
            #EST = [column for column in df_raw if column.startswith('EST')]
            #AGR = [column for column in df_raw if column.startswith('AGR')]
            #CSN = [column for column in df_raw if column.startswith('CSN')]
            #OPN = [column for column in df_raw if column.startswith('OPN')]

            #matching = EXT+EST+AGR+CSN+OPN

            # Here we update the dataframe by applying the new coefficient
        for column in df_raw.columns:
            df_raw[column] = df_raw[column] * questions[column][1]

        # reference to scoring: https://sites.temple.edu/rtassessment/files/2018/10/Table_BFPT.pdf 
        df_raw['extraversion'] = df_raw.iloc[:, 0:10].sum(axis=1) + 20
        df_raw['neuroticism'] = df_raw.iloc[:, 10:20].sum(axis=1) +38
        df_raw['agreeableness'] = df_raw.iloc[:, 20:30].sum(axis=1) +14 
        df_raw['conscientiousness'] = df_raw.iloc[:, 30:40].sum(axis=1) + 14
        df_raw['openness'] = df_raw.iloc[:, 40:50].sum(axis=1) + 8
        df_raw['name'] = df_raw.index.to_series().apply(lambda idx: 'C_' + str(idx))
             
        return df_raw
        
    
    def to_data_point(self) -> data_point.Person:
        
        id = self.df.index[0]
        name = self.df['name'].values[0]

        #Reindexing dataframe
        self.df.reset_index(drop=True, inplace=True)

        self.df=self.df.drop(columns=["name"])

        # Convert to series
        ser_metrics = self.df.squeeze()
        
        return self.data_point_class(id=id,name=name,ser_metrics=ser_metrics)

