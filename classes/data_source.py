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
#importing necessary libraries
from mplsoccer import Sbopen
import pandas as pd
import numpy as np
import warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import random as rn
#warnings not visible on the course webpage
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')
import pickle 
from joblib import load


import classes.data_point as data_point

# from classes.wyscout_api import WyNot


# Base class for all data
class Data:
    """
    Get, process, and manage various forms of data.
    """

    data_point_class = None

    def __init__(self):
        self.df = self.get_processed_data()

    def get_raw_data(self) -> pd.DataFrame:
        raise NotImplementedError("Child class must implement get_raw_data(self)")

    def process_data(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(
            "Child class must implement process_data(self, df_raw)"
        )

    def get_processed_data(self):

        raw = self.get_raw_data()
        return self.process_data(raw)

    def select_and_filter(self, column_name, label, default_index=0):

        df = self.df
        selected_id = st.selectbox(label, df[column_name].unique(), index=default_index)
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

    def get_metric_zscores(self, df):

        df_z = df.apply(zscore, nan_policy="omit")

        # Rename every column to include "Z" at the end
        df_z.columns = [f"{col}_Z" for col in df_z.columns]

        # Here we get opposite value of metrics if their weight is negative
        for metric in set(self.negative_metrics).intersection(self.metrics):
            df_z[metric] = df_z[metric] * -1
        return df_z

    def get_ranks(self, df):
        df_ranks = df.rank(ascending=False)

        # Rename every column to include "Ranks" at the end
        df_ranks.columns = [f"{col}_Ranks" for col in df_ranks.columns]

        return df_ranks

    def get_pct_ranks(self, df):
        df_pct = df.rank(pct=True) * 100
        # Rename every column to include "Pct_Ranks" at the end
        df_pct.columns = [f"{col}_Pct_Ranks" for col in df_pct.columns]

        return df_pct

    def calculate_statistics(self, metrics, negative_metrics=[]):
        self.metrics = metrics
        self.negative_metrics = negative_metrics

        df = self.df
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

        df = pd.read_csv("data/events/Forwards.csv", encoding="unicode_escape")

        return df

    def process_data(self, df_raw):
        df_raw = df_raw.rename(columns={"shortName": "player_name"})

        df_raw = df_raw.replace({-1: np.nan})
        # Remove players with low minutes
        df_raw = df_raw[(df_raw.Minutes >= self.minimal_minutes)]

        if len(df_raw) < 10:  # Or else plots won't work
            raise Exception("Not enough players with enough minutes")

        return df_raw

    def to_data_point(self, gender, position) -> data_point.Player:

        id = self.df.index[0]

        # Reindexing dataframe
        self.df.reset_index(drop=True, inplace=True)

        name = self.df["player_name"][0]
        minutes_played = self.df["Minutes"][0]
        self.df = self.df.drop(columns=["player_name", "Minutes"])

        # Convert to series
        ser_metrics = self.df.squeeze()

    



class CountryStats(Stats):
    data_point_class = data_point.Country
    # This can be used if some metrics are not good to perform, like tackles lost.
    negative_metrics = []

    def __init__(self):

        self.drill_down = self.get_drill_down_dict()
        self.drill_down_threshold = 1

        super().__init__()

    def get_drill_down_data(self, file_path):
        df = self.process_data(self.get_z_scores(pd.read_csv(file_path)))

        return dict(zip(df.country.values, df.drill_down_metric.values))

    def get_drill_down_data_values(self, file_path, metric_name):

        df = self.process_data(pd.read_csv(file_path))

        # create a value column that has the values from the columns is given by the dict self.drill_down_metric_country_question
        # where the dict has format {country: question}
        df["value_low"] = df.apply(
            lambda x: x[
                self.drill_down_metric_country_question[metric_name][x["country"]][0]
            ],
            axis=1,
        )

        df["value_high"] = df.apply(
            lambda x: x[
                self.drill_down_metric_country_question[metric_name][x["country"]][1]
            ],
            axis=1,
        )

        values = [
            (floor(l), ceil(h)) for l, h in zip(df["value_low"], df["value_high"])
        ]

        return dict(zip(df.country.values, values))

    def get_drill_down_dict(
        self,
    ):

        # read all .csv files from path ending in _pre.csv
        path = "data/wvs/intermediate_data/"
        all_files = os.listdir(path)

        self.drill_down_metric_country_question = dict(
            (
                "_".join(file.split("_")[:-1]),
                self.get_drill_down_data(path + file),
            )
            for file in all_files
            if file.endswith("_pre.csv")
        )

        drill_down_data_raw = dict(
            (
                "_".join(file.split("_")[:-1]),
                self.get_drill_down_data_values(
                    path + file, "_".join(file.split("_")[:-1])
                ),
            )
            for file in all_files
            if file.endswith("_raw.csv")
        )

        metrics = [m for m in self.drill_down_metric_country_question.keys()]
        countries = [
            k for k in self.drill_down_metric_country_question[metrics[0]].keys()
        ]

        drill_down = [
            (
                country,
                dict(
                    [
                        (
                            metric,
                            (
                                self.drill_down_metric_country_question[metric][
                                    country
                                ],
                                drill_down_data_raw[metric][country],
                            ),
                        )
                        for metric in metrics
                    ]
                ),
            )
            for country in countries
        ]

        return dict(drill_down)

    def get_z_scores(self, df, metrics=None, negative_metrics=[]):

        if metrics is None:
            metrics = [m for m in df.columns if m not in ["country"]]

        df_z = df[metrics].apply(zscore, nan_policy="omit")

        # Rename every column to include "Z" at the end
        df_z.columns = [f"{col}_Z" for col in df_z.columns]

        # Here we get opposite value of metrics if their weight is negative
        for metric in set(negative_metrics).intersection(metrics):
            df_z[metric] = df_z[metric] * -1

        # find the columns that has greatest magnitude
        drill_down_metrics_high = df[metrics].idxmax(axis=1)
        drill_down_metrics_low = df[metrics].idxmin(axis=1)

        drill_down_metrics = [
            (l, h) for l, h in zip(drill_down_metrics_low, drill_down_metrics_high)
        ]

        ### Using the question with the greatest raw magnitude vs using the question with the greatest z-score magnitude

        # # find the columns that has greatest magnitude
        # drill_down_metrics = df[metrics].abs().idxmax(axis=1)

        # # find the columns that end with "_Z" and has greatest magnitude
        # drill_down_metrics = (
        #     df_z[df_z.columns[df_z.columns.str.endswith("_Z")]]
        #     .abs()
        #     .idxmax(axis=1)
        #     .apply(lambda x: "_".join(x.split("_")[:-1]))
        # )

        ###

        df_z["drill_down_metric"] = drill_down_metrics

        # # Here we want to use df_metric_zscores to get the ranks and pct_ranks due to negative metrics
        # df_ranks = df[metrics].rank(ascending=False)

        # # Rename every column to include "Ranks" at the end
        # df_ranks.columns = [f"{col}_Ranks" for col in df_ranks.columns]

        # Add ranks and pct_ranks as new columns
        # return pd.concat([df, df_z, df_ranks], axis=1)
        return pd.concat([df, df_z], axis=1)

    def select_random(self):
        # return the index of the random sample
        return self.df.sample(1).index[0]

    def get_raw_data(self):

        df = pd.read_csv("data/wvs/wave_7.csv")

        return df

    def process_data(self, df_raw):

        # raise error if df_raw["country"] contains any NaN values
        if df_raw["country"].isnull().values.any():
            raise ValueError("Country column contains NaN values")
        # raise error if df_raw["country"] contains any empty strings
        if (df_raw["country"] == "").values.any():
            raise ValueError("Country column contains empty strings")

        # raise error if df_raw["country"] contains any duplicates
        if df_raw["country"].duplicated().any():
            raise ValueError("Country column contains duplicates")

        # # raise warning is any nan values are present in the dataframe
        # if df_raw.isnull().values.any():
        #     st.warning("Data contains NaN values")

        if len(df_raw) < 10:  # Or else plots won't work
            raise Exception("Not enough data points")

        return df_raw

    def to_data_point(self) -> data_point.Country:

        id = self.df.index[0]

        # Reindexing dataframe
        self.df.reset_index(drop=True, inplace=True)

        name = self.df["country"][0]
        self.df = self.df.drop(columns=["country"])

        # Convert to series
        ser_metrics = self.df.squeeze()

        # get the names of columns in ser_metrics than end in "_Z" with abs value greater than 1.5
        drill_down_metrics = ser_metrics[
            ser_metrics.index.str.endswith("_Z")
            & (ser_metrics.abs() >= self.drill_down_threshold)
        ].index.tolist()
        drill_down_metrics = [
            "_".join(x.split("_")[:-1]).lower() for x in drill_down_metrics
        ]

        drill_down_values = dict(
            [
                (key, value)
                for key, value in self.drill_down[name].items()
                if key.lower() in drill_down_metrics
            ]
        )

        return self.data_point_class(
            id=id,
            name=name,
            ser_metrics=ser_metrics,
            relevant_metrics=self.metrics,
            drill_down_metrics=drill_down_values,
        )






class Shots(Data):
    def __init__(self):
        #self.raw_hash_attrs = (competition.id, match.id, team.id)
        #self.proc_hash_attrs = (competition.id, match.id, team.id)
        #self.competition = competition
        #self.match = match
        #self.team = team
        self.df_shots = self.get_processed_data()  # Process the raw data directly
        self.xG_Model = self.load_model()  # Load the model once
        self.parameters = self.read_model_params()
        #self.df_cum_xG, self.df_contributions = self.get_xG_contributions()
        self.df_contributions = self.weight_contributions()
        # Add total xG to df_shots
        #self.df_shots["xG"] = self.xG_Model.predict(self.df_shots)
    #@st.cache_data(hash_funcs={"classes.data_source.Shots": lambda self: hash(self.raw_hash_attrs)}, ttl=5*60)


    def get_raw_data(self):
        parser = Sbopen()
        #get list of games during Indian Super League season
        df_match = parser.match(competition_id=55, season_id=282)

        # matches = df_match.match_id.unique()
        matches= [3942819]
        shot_df = pd.DataFrame()
        track_df = pd.DataFrame()
        #store data in one dataframe
        for match in matches:
            #open events
            df_event = parser.event(match)[0]
            #open 360 data
            df_track = parser.event(match)[2]
            #get shots
            shots = df_event.loc[df_event["type_name"] == "Shot"]
            shots.x = shots.x.apply(lambda cell: cell*105/120)
            shots.y = shots.y.apply(lambda cell: cell*68/80)
            df_track.x = df_track.x.apply(lambda cell: cell*105/120)
            df_track.y = df_track.y.apply(lambda cell: cell*68/80)
            #append event and trackings to a dataframe
            shot_df = pd.concat([shot_df, shots], ignore_index = True)
            track_df = pd.concat([track_df, df_track], ignore_index = True)

        #reset indicies
        shot_df.reset_index(drop=True, inplace=True)
        track_df.reset_index(drop=True, inplace=True)
        #filter out non open-play shots
        shot_df = shot_df.loc[shot_df["sub_type_name"] == "Open Play"]
        #filter out shots where goalkeeper was not tracked
        gks_tracked = track_df.loc[track_df["teammate"] == False].loc[track_df["position_name"] == "Goalkeeper"]['id'].unique()
        shot_df = shot_df.loc[shot_df["id"].isin(gks_tracked)]

        df_raw = (shot_df, track_df) 

        return df_raw


    def process_data(self, df_raw: tuple) -> pd.DataFrame:

        test_shot, track_df = df_raw

        #ball_goalkeeper distance
        def dist_to_gk(test_shot, track_df):
            #get id of the shot to search for tracking data using this index
            test_shot_id = test_shot["id"]
            #check goalkeeper position
            gk_pos = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False].loc[track_df["position_name"] == "Goalkeeper"][["x", "y"]]
            #calculate distance from event to goalkeeper position
            dist = np.sqrt((test_shot["x"] - gk_pos["x"])**2 + (test_shot["y"] - gk_pos["y"])**2)
            return dist.iloc[0]

        #ball goalkeeper y axis
        def y_to_gk(test_shot, track_df):
            #get id of the shot to search for tracking data using this index
            test_shot_id = test_shot["id"]
            #calculate distance from event to goalkeeper position
            gk_pos = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False].loc[track_df["position_name"] == "Goalkeeper"][["y"]]
            #calculate distance from event to goalkeeper position in y axis
            dist = abs(test_shot["y"] - gk_pos["y"])
            return dist.iloc[0]

        #number of players less than 3 meters away from the ball
        def three_meters_away(test_shot, track_df):
            #get id of the shot to search for tracking data using this index
            test_shot_id = test_shot["id"]
            #get all opposition's player location
            player_position = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False][["x", "y"]]
            #calculate their distance to the ball
            dist = np.sqrt((test_shot["x"] - player_position["x"])**2 + (test_shot["y"] - player_position["y"])**2)
            #return how many are closer to the ball than 3 meters
            return len(dist[dist<3])

        #number of players inside a triangle
        def players_in_triangle(test_shot, track_df):
            #get id of the shot to search for tracking data using this index
            test_shot_id = test_shot["id"]
            #get all opposition's player location
            player_position = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False][["x", "y"]]
            #checking if point inside a triangle
            x1 = 105
            y1 = 34 - 7.32/2
            x2 = 105
            y2 = 34 + 7.32/2
            x3 = test_shot["x"]
            y3 = test_shot["y"]
            xp = player_position["x"]
            yp = player_position["y"]
            c1 = (x2-x1)*(yp-y1)-(y2-y1)*(xp-x1)
            c2 = (x3-x2)*(yp-y2)-(y3-y2)*(xp-x2)
            c3 = (x1-x3)*(yp-y3)-(y1-y3)*(xp-x3)
            #get number of players inside a triangle
            return len(player_position.loc[((c1<0) & (c2<0) & (c3<0)) | ((c1>0) & (c2>0) & (c3>0))])

        #goalkeeper distance to goal
        def gk_dist_to_goal(test_shot, track_df):
            #get id of the shot to search for tracking data using this index
            test_shot_id = test_shot["id"]
            #get goalkeeper position
            gk_pos = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False].loc[track_df["position_name"] == "Goalkeeper"][["x", "y"]]
            #calculate their distance to goal
            dist = np.sqrt((105 -gk_pos["x"])**2 + (34 - gk_pos["y"])**2)
            return dist.iloc[0]


        # Distance to the nearest opponent
        def nearest_opponent_distance(test_shot, track_df):
            # Get the ID of the shot to search for tracking data using this index
            test_shot_id = test_shot["id"]
            
            # Get all opposition's player locations (non-teammates)
            opponent_position = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False][["x", "y"]]
            
            # Calculate the Euclidean distance to each opponent
            distances = np.sqrt((test_shot["x"] - opponent_position["x"])**2 + (test_shot["y"] - opponent_position["y"])**2)
            
            # Return the minimum distance (i.e., nearest opponent)
            return distances.min() if len(distances) > 0 else np.nan

        # Calculate the angle to the nearest opponent
        def nearest_opponent_angle(test_shot, track_df):
            # Get the ID of the shot to search for tracking data using this index
            test_shot_id = test_shot["id"]
            
            # Get all opposition's player locations (non-teammates)
            opponent_position = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False][["x", "y"]]
            
            # Check if there are any opponents
            if opponent_position.empty:
                return np.nan
            
            # Calculate the Euclidean distance to each opponent
            distances = np.sqrt((test_shot["x"] - opponent_position["x"])**2 + (test_shot["y"] - opponent_position["y"])**2)
            
            # Find the index of the nearest opponent
            nearest_index = distances.idxmin()
            
            # Get the coordinates of the nearest opponent
            nearest_opponent = opponent_position.loc[nearest_index]
            
            # Calculate the angle to the nearest opponent using arctan2
            angle = np.degrees(np.arctan2(nearest_opponent["y"] - test_shot["y"], nearest_opponent["x"] - test_shot["x"]))
            
            # Normalize angles to be within 0-360 degrees
            angle = angle % 360
            
            return angle



        def angle_to_gk(test_shot, track_df):
            #get id of the shot to search for tracking data using this index
            test_shot_id = test_shot["id"]
            #check goalkeeper position
            gk_pos = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False].loc[track_df["position_name"] == "Goalkeeper"][["x", "y"]]
            angle = np.degrees(np.arctan2(gk_pos["y"] - test_shot["y"], gk_pos["x"] - test_shot["x"]))
            angle = angle % 360
            return angle.iloc[0]



        # Distance to the nearest teammate
        def nearest_teammate_distance(test_shot, track_df):
            # Get the ID of the shot to search for tracking data using this index
            test_shot_id = test_shot["id"]
            
            # Get all teammates' locations (excluding the shooter)
            teammate_position = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == True][["x", "y"]]
            
            # Calculate the Euclidean distance to each teammate
            distances = np.sqrt((test_shot["x"] - teammate_position["x"])**2 + (test_shot["y"] - teammate_position["y"])**2)
            
            # Return the minimum distance (i.e., nearest teammate)
            return distances.min() if len(distances) > 0 else np.nan


        # Transposing teammates' and opponents' coordinates into the dataframe
        def transpose_player_positions(test_shot, track_df):
            test_shot_id = test_shot["id"]

            # Separate teammates and opponents
            teammates = track_df[(track_df["id"] == test_shot_id) & (track_df["teammate"] == True)]
            opponents = track_df[(track_df["id"] == test_shot_id) & (track_df["teammate"] == False)]

            # Transpose and flatten coordinates for teammates
            teammate_x = teammates["x"].values
            teammate_y = teammates["y"].values
            teammate_coords = {
                f"teammate_{i+1}_x": x for i, x in enumerate(teammate_x)
            }
            teammate_coords.update({
                f"teammate_{i+1}_y": y for i, y in enumerate(teammate_y)
            })

            # Transpose and flatten coordinates for opponents
            opponent_x = opponents["x"].values
            opponent_y = opponents["y"].values
            opponent_coords = {
                f"opponent_{i+1}_x": x for i, x in enumerate(opponent_x)
            }
            opponent_coords.update({
                f"opponent_{i+1}_y": y for i, y in enumerate(opponent_y)
            })

            # Combine both dictionaries into a single row dictionary
            return {**teammate_coords, **opponent_coords}
        
        model_vars = test_shot[["id", "index", "x", "y"]].copy()
        model_vars["header"] = test_shot.body_part_name.apply(lambda cell: 1 if cell == "Head" else 0)
        model_vars["left_foot"] = test_shot.body_part_name.apply(lambda cell: 1 if cell == "Left Foot" else 0)
        model_vars["throw in"] = test_shot.play_pattern_name.apply(lambda cell: 1 if cell == "From Throw In" else 0)
        model_vars["corner"] = test_shot.play_pattern_name.apply(lambda cell: 1 if cell == "From Corner" else 0)
        model_vars["free_kich"] = test_shot.play_pattern_name.apply(lambda cell: 1 if cell == "From Free Kick" else 0)
        model_vars["goal"] = test_shot.outcome_name.apply(lambda cell: 1 if cell == "Goal" else 0)

        # Add necessary features and correct transformations
        model_vars["goal_smf"] = model_vars["goal"].astype(object)
        model_vars['start_x'] = model_vars.x
        model_vars['start_y'] = model_vars.y
        model_vars["x"] = model_vars.x.apply(lambda cell: 105 - cell)  # Adjust x for goal location
        model_vars["c"] = model_vars.y.apply(lambda cell: abs(34 - cell))

        # Calculate angle and distance
        model_vars["angle_to_goal"] = np.where(np.arctan(7.32 * model_vars["x"] / (model_vars["x"]**2 + model_vars["c"]**2 - (7.32/2)**2)) >= 0,
                                    np.arctan(7.32 * model_vars["x"] / (model_vars["x"]**2 + model_vars["c"]**2 - (7.32/2)**2)),
                                    np.arctan(7.32 * model_vars["x"] / (model_vars["x"]**2 + model_vars["c"]**2 - (7.32/2)**2)) + np.pi) * 180 / np.pi

        model_vars["distance to goal"] = np.sqrt(model_vars["x"]**2 + model_vars["c"]**2)

        # Add other features (assuming your earlier functions return correct results)
        model_vars["dist_to_gk"] = test_shot.apply(dist_to_gk, track_df=track_df, axis=1)
        model_vars["gk_distance_y"] = test_shot.apply(y_to_gk, track_df=track_df, axis=1)
        model_vars["close_players"] = test_shot.apply(three_meters_away, track_df=track_df, axis=1)
        model_vars["triangle zone"] = test_shot.apply(players_in_triangle, track_df=track_df, axis=1)
        model_vars["gk distance to goal"] = test_shot.apply(gk_dist_to_goal, track_df=track_df, axis=1)
        model_vars["distance to nearest opponent"] = test_shot.apply(nearest_opponent_distance, track_df=track_df, axis=1)
        model_vars["angle_to_nearest_opponent"] = test_shot.apply(nearest_opponent_angle, track_df=track_df, axis=1)
        model_vars["angle_to_gk"] = test_shot.apply(angle_to_gk, track_df=track_df, axis=1)
        model_vars['distance from touchline']= model_vars['c']**2
        model_vars['angle to gk and opponent']= model_vars['angle_to_goal']*model_vars['angle_to_gk']*model_vars['angle_to_nearest_opponent']
        # Merge player position data by applying the transpose_player_positions function
        player_position_features = test_shot.apply(transpose_player_positions, track_df=track_df, axis=1)

        # Convert player_position_features (which is a Series of dicts) into a DataFrame and concatenate with model_vars
        player_position_df = pd.DataFrame(list(player_position_features))
        model_vars = pd.concat([model_vars.reset_index(drop=True), player_position_df.reset_index(drop=True)], axis=1)



        # Binary features
        model_vars["is_closer"] = np.where(model_vars["gk distance to goal"] > model_vars["distance to goal"], 1, 0)
        model_vars["header"] = test_shot.body_part_name.apply(lambda cell: 1 if cell == "Head" else 0)
        model_vars['goal'] = test_shot.outcome_name.apply(lambda cell: 1 if cell == "Goal" else 0)
        model_vars['Intercept'] = 1
        #model_vars = sm.add_constant(model_vars)
        #model_vars.rename(columns={'const': 'Intercept'}, inplace=True)  # Rename 'const' to 'Intercept'

        # model_vars.dropna(inplace=True)

        return model_vars
    

    def read_model_params(self):
        parameters = pd.read_excel("data/model_params.xlsx")
        return parameters


    def weight_contributions(self):
        
        df_shots = self.df_shots
        parameters = self.parameters
        self.intercept = parameters[parameters['Parameter'] == 'Intercept']['Value'].values[0]
        # Drop intercept
        self.parameters = parameters[parameters['Parameter'] != 'Intercept']

        for i,row in self.parameters.iterrows():
            df_shots[row['Parameter']+'_contribution'] = df_shots[row['Parameter']] * row['Value']
            #Remove the mean
            df_shots[row['Parameter']+'_contribution'] = df_shots[row['Parameter']+'_contribution'] - df_shots[row['Parameter']+'_contribution'].mean()
        
        df_contribution = df_shots[['id'] + [col for col in df_shots.columns if 'contribution' in col]]

        # Calculate linear combination directly from original feature values and their coefficients
        linear_combination = self.intercept + sum(
            df_shots[param['Parameter']] * param['Value']
            for _, param in self.parameters.iterrows()
        )
        
        # Apply logistic function to get xG values
        df_contribution['xG'] = 1 / (1 + np.exp(-linear_combination))
        df_shots['xG'] = 1 / (1 + np.exp(-linear_combination))

        #self.parameter_explanation = parameters.set_index('Parameter')['Explanation'].to_dict()
        #st.write(self.parameter_explanation)

        return df_contribution



    def get_xG_contributions(self, df_shots=None):
        if df_shots is None:
            df_shots = self.df_shots

        # Get model parameters including intercept
        model_params = self.model_params  # This includes the intercept, assumed to be labeled 'const'
        #model_params = [param if param != "Intercept" else "const" for param in self.xG_Model.params.index.tolist()]

        intercept = self.xG_Model.params[0]  # Intercept is the first element in the model params

        # Initialize lists to store cumulative xG and contributions for each shot
        cumulative_xG_list = []
        contributions_list = []
        shot_ids = []

        # Loop over each shot in the DataFrame
        for _, shot in df_shots.iterrows():
            cumulative_xG_shot = []  # Cumulative xG for this shot
            prev_xG = 1 / (1 + np.exp(-intercept))  # xG with just intercept
            shot_ids.append(shot['id'])  # Store the shot ID
            

            
            # First, calculate the xG for intercept only and store
            cumulative_xG_shot.append(prev_xG)

            # Now iteratively add features and calculate xG
            for i, feature in enumerate(model_params[1:], start=1):  # model_params[1:] skips intercept
                # Get current subset of parameters
                relevant_params = model_params[:i+1]  # Include intercept and features up to the current one
                relevant_weights = self.xG_Model.params[:i+1]  # Corresponding weights
                
                # Linear combination for the relevant features
                linear_combination = np.dot(shot[relevant_params], relevant_weights)
                
                # Calculate xG using the sigmoid function
                current_xG = 1 / (1 + np.exp(-linear_combination))
                
                # Append the current xG to the cumulative list
                cumulative_xG_shot.append(current_xG)

            # Now that cumulative xG is calculated for this shot, we calculate contributions
            contributions_shot = np.diff(cumulative_xG_shot, prepend=prev_xG)

            # Append cumulative xG and contributions for this shot
            cumulative_xG_list.append(cumulative_xG_shot)
            contributions_list.append(contributions_shot)


        # Create DataFrames for cumulative xG and contributions
        df_cum_xG = pd.DataFrame(cumulative_xG_list, index=df_shots.index, columns=model_params)
        df_contributions = pd.DataFrame(contributions_list, index=df_shots.index, columns=model_params)
        df_contributions['shot_id'] = shot_ids

        # Output the tables with an introduction
        #st.markdown("### Cumulative Expected Goals (xG) for Each Feature")
        #st.write(df_cum_xG)
        #st.markdown("### Contributions of Each Feature to xG")
        #st.write(df_contributions)

        return df_cum_xG, df_contributions




 
    @staticmethod
    def load_model():

        # Load model from data/...
        saved_model_path = "data/xG_model.sav"
        model = load(saved_model_path)
        st.markdown("### Model Summary")
        st.write(model.summary())   
        #predictions = model.predict(df_shots)
        return model

        












