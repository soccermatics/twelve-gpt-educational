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

        return self.data_point_class(
            id=id,
            name=name,
            minutes_played=minutes_played,
            gender=gender,
            position=position,
            ser_metrics=ser_metrics,
            relevant_metrics=self.metrics,
        )


class CountryStats(Stats):
    data_point_class = data_point.Country
    # This can be used if some metrics are not good to perform, like tackles lost.
    negative_metrics = []

    def __init__(self):

        self.core_value_dict = pd.read_csv("data/wvs/values.csv")
        # convert to dictionary using the value and name columns
        self.core_value_dict = self.core_value_dict.set_index("value")["name"].to_dict()

        self.drill_down = self.get_drill_down_dict()

        super().__init__()

    def get_drill_down_data(self, file_path):
        df = self.process_data(
            self.get_z_scores(pd.read_csv(file_path).drop(columns=["score"]))
        )

        return dict(zip(df.country.values, df.drill_down_metric.values))

    def get_drill_down_data_values(self, file_path, metric_name):

        df = self.process_data(pd.read_csv(file_path).drop(columns=["score"]))

        # create a value column that has the values from the columns is given by the dict self.drill_down_metric_country_question
        # where the dict has format {country: question}
        df["value"] = df.apply(
            lambda x: x[
                self.drill_down_metric_country_question[metric_name][x["country"]]
            ],
            axis=1,
        )

        return dict(zip(df.country.values, df.value.values))

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

        # find the columns that end with "_Z" and has greatest magnitude
        drill_down_metrics = (
            df_z[df_z.columns[df_z.columns.str.endswith("_Z")]]
            .abs()
            .idxmax(axis=1)
            .apply(lambda x: "_".join(x.split("_")[:-1]))
        )
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
        df_raw = df_raw.rename(columns=self.core_value_dict)

        country_codes_dict = {
            "AFG": "Afghanistan",
            "ALB": "Albania",
            "DZA": "Algeria",
            "AND": "Andorra",
            "AGO": "Angola",
            "ATG": "Antigua and Barbuda",
            "ARG": "Argentina",
            "ARM": "Armenia",
            "AUS": "Australia",
            "AUT": "Austria",
            "AZE": "Azerbaijan",
            "BHS": "Bahamas",
            "BHR": "Bahrain",
            "BGD": "Bangladesh",
            "BRB": "Barbados",
            "BLR": "Belarus",
            "BEL": "Belgium",
            "BLZ": "Belize",
            "BEN": "Benin",
            "BTN": "Bhutan",
            "BOL": "Bolivia",
            "BIH": "Bosnia and Herzegovina",
            "BWA": "Botswana",
            "BRA": "Brazil",
            "BRN": "Brunei",
            "BGR": "Bulgaria",
            "BFA": "Burkina Faso",
            "BDI": "Burundi",
            "CPV": "Cabo Verde",
            "KHM": "Cambodia",
            "CMR": "Cameroon",
            "CAN": "Canada",
            "CAF": "Central African Republic",
            "TCD": "Chad",
            "CHL": "Chile",
            "CHN": "China",
            "COL": "Colombia",
            "COM": "Comoros",
            "COG": "Congo",
            "CRI": "Costa Rica",
            "HRV": "Croatia",
            "CUB": "Cuba",
            "CYP": "Cyprus",
            "CZE": "Czechia",
            "DNK": "Denmark",
            "DJI": "Djibouti",
            "DMA": "Dominica",
            "DOM": "Dominican Republic",
            "ECU": "Ecuador",
            "EGY": "Egypt",
            "SLV": "El Salvador",
            "GNQ": "Equatorial Guinea",
            "ERI": "Eritrea",
            "EST": "Estonia",
            "SWZ": "Eswatini",
            "ETH": "Ethiopia",
            "FJI": "Fiji",
            "FIN": "Finland",
            "FRA": "France",
            "GAB": "Gabon",
            "GMB": "Gambia",
            "GEO": "Georgia",
            "DEU": "Germany",
            "GHA": "Ghana",
            "GRC": "Greece",
            "GRD": "Grenada",
            "GTM": "Guatemala",
            "GIN": "Guinea",
            "GNB": "Guinea-Bissau",
            "GUY": "Guyana",
            "HTI": "Haiti",
            "HND": "Honduras",
            "HUN": "Hungary",
            "ISL": "Iceland",
            "IND": "India",
            "IDN": "Indonesia",
            "IRN": "Iran",
            "IRQ": "Iraq",
            "IRL": "Ireland",
            "ISR": "Israel",
            "ITA": "Italy",
            "JAM": "Jamaica",
            "JPN": "Japan",
            "JOR": "Jordan",
            "KAZ": "Kazakhstan",
            "KEN": "Kenya",
            "KIR": "Kiribati",
            "PRK": "Korea, North",
            "KOR": "Korea, South",
            "KWT": "Kuwait",
            "KGZ": "Kyrgyzstan",
            "LAO": "Laos",
            "LVA": "Latvia",
            "LBN": "Lebanon",
            "LSO": "Lesotho",
            "LBR": "Liberia",
            "LBY": "Libya",
            "LIE": "Liechtenstein",
            "LTU": "Lithuania",
            "LUX": "Luxembourg",
            "MDG": "Madagascar",
            "MWI": "Malawi",
            "MYS": "Malaysia",
            "MDV": "Maldives",
            "MLI": "Mali",
            "MLT": "Malta",
            "MHL": "Marshall Islands",
            "MRT": "Mauritania",
            "MUS": "Mauritius",
            "MEX": "Mexico",
            "FSM": "Micronesia",
            "MDA": "Moldova",
            "MCO": "Monaco",
            "MNG": "Mongolia",
            "MNE": "Montenegro",
            "MAR": "Morocco",
            "MOZ": "Mozambique",
            "MMR": "Myanmar",
            "NAM": "Namibia",
            "NRU": "Nauru",
            "NPL": "Nepal",
            "NLD": "Netherlands",
            "NZL": "New Zealand",
            "NIC": "Nicaragua",
            "NER": "Niger",
            "NGA": "Nigeria",
            "MKD": "North Macedonia",
            "NOR": "Norway",
            "OMN": "Oman",
            "PAK": "Pakistan",
            "PLW": "Palau",
            "PAN": "Panama",
            "PNG": "Papua New Guinea",
            "PRY": "Paraguay",
            "PER": "Peru",
            "PHL": "Philippines",
            "POL": "Poland",
            "PRT": "Portugal",
            "QAT": "Qatar",
            "ROU": "Romania",
            "RUS": "Russia",
            "RWA": "Rwanda",
            "KNA": "Saint Kitts and Nevis",
            "LCA": "Saint Lucia",
            "VCT": "Saint Vincent and the Grenadines",
            "WSM": "Samoa",
            "SMR": "San Marino",
            "STP": "Sao Tome and Principe",
            "SAU": "Saudi Arabia",
            "SEN": "Senegal",
            "SRB": "Serbia",
            "SYC": "Seychelles",
            "SLE": "Sierra Leone",
            "SGP": "Singapore",
            "SVK": "Slovakia",
            "SVN": "Slovenia",
            "SLB": "Solomon Islands",
            "SOM": "Somalia",
            "ZAF": "South Africa",
            "SSD": "South Sudan",
            "ESP": "Spain",
            "LKA": "Sri Lanka",
            "SDN": "Sudan",
            "SUR": "Suriname",
            "SWE": "Sweden",
            "CHE": "Switzerland",
            "SYR": "Syria",
            "TWN": "Taiwan",
            "TJK": "Tajikistan",
            "TZA": "Tanzania",
            "THA": "Thailand",
            "TLS": "Timor-Leste",
            "TGO": "Togo",
            "TON": "Tonga",
            "TTO": "Trinidad and Tobago",
            "TUN": "Tunisia",
            "TUR": "Turkey",
            "TKM": "Turkmenistan",
            "TUV": "Tuvalu",
            "UGA": "Uganda",
            "UKR": "Ukraine",
            "ARE": "United Arab Emirates",
            "GBR": "United Kingdom",
            "USA": "United States",
            "URY": "Uruguay",
            "UZB": "Uzbekistan",
            "VUT": "Vanuatu",
            "VEN": "Venezuela",
            "VNM": "Vietnam",
            "YEM": "Yemen",
            "ZMB": "Zambia",
            "ZWE": "Zimbabwe",
            "HKG": "Hong Kong",
            "MAC": "Macau",
            "PRI": "Puerto Rico",
            "NIR": "Northern Ireland",
            "GRL": "Greenland",
            "CIV": "Ivory Coast",
            "COD": "Congo, Democratic Republic of the",
            "SWZ": "Eswatini",
        }

        country_names = df_raw["country"].values.tolist()
        # check if the country names are in the country_codes_dict
        if not set(country_names).issubset(set(country_codes_dict.keys())):
            # print the country names that are not in the country_codes_dict
            print(set(country_names) - set(country_codes_dict.keys()))
            raise ValueError("Country names do not match the country codes")

        df_raw["country"] = df_raw["country"].map(country_codes_dict)

        # df_raw = df_raw.replace({-1: np.nan})

        if len(df_raw) < 10:  # Or else plots won't work
            raise Exception("Not enough players with enough minutes")

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
            ser_metrics.index.str.endswith("_Z") & (ser_metrics.abs() > 1.0)
        ].index.tolist()
        drill_down_metrics = ["_".join(x.split("_")[:-1]) for x in drill_down_metrics]

        drill_down_values = dict(
            [
                (key, value)
                for key, value in self.drill_down[name].items()
                if key in drill_down_metrics
            ]
        )

        return self.data_point_class(
            id=id,
            name=name,
            ser_metrics=ser_metrics,
            relevant_metrics=self.metrics,
            drill_down_metrics=drill_down_values,
        )


class PersonStat(Stats):
    data_point_class = data_point.Person
    negative_metrics = []

    def __init__(self):
        super().__init__()

    def get_raw_data(self):
        # df = pd.read_csv('data/data-final.csv',sep='\t',encoding='unicode_escape').sample(frac=0.0001)
        df = pd.read_csv("data/data_raw.csv", sep="\t", encoding="unicode_escape")
        return df

    def get_questions(self):
        """This function is to have access to the questions"""

        # Groups and Questions modify version
        # (1) extraversion, (2) neuroticism, (3) agreeableness, (4)conscientiousness , and (5) openness
        ext_questions = {
            "EXT1": ["they are the life of the party", 1],
            "EXT2": ["they dont talk a lot", -1],
            "EXT3": ["they feel comfortable around people", 1],
            "EXT4": ["they keep in the background", -1],
            "EXT5": ["they start conversations", 1],
            "EXT6": ["they have little to say", -1],
            "EXT7": ["they talk to a lot of different people at parties", 1],
            "EXT8": ["they dont like to draw attention to themself", -1],
            "EXT9": ["they dont mind being the center of attention", 1],
            "EXT10": ["they are quiet around strangers", -1],
        }

        est_questions = {
            "EST1": ["they get stressed out easily", -1],
            "EST2": ["they are relaxed most of the time", 1],
            "EST3": ["they worry about things", -1],
            "EST4": ["they seldom feel blue", 1],
            "EST5": ["they are easily disturbed", -1],
            "EST6": ["they get upset easily", -1],
            "EST7": ["they change their mood a lot", -1],
            "EST8": ["they have frequent mood swings", -1],
            "EST9": ["they get irritated easily", -1],
            "EST10": ["they often feel blue", -1],
        }

        agr_questions = {
            "AGR1": ["they feel little concern for others", -1],
            "AGR2": ["they interested in people", 1],
            "AGR3": ["they insult people", -1],
            "AGR4": ["they sympathize with others feelings", 1],
            "AGR5": ["they are not interested in other peoples problems", -1],
            "AGR6": ["they have a soft heart", 1],
            "AGR7": ["they not really interested in others", -1],
            "AGR8": ["they take time out for others", 1],
            "AGR9": ["they feel others emotions", 1],
            "AGR10": ["they make people feel at ease", 1],
        }

        csn_questions = {
            "CSN1": ["they are always prepared", 1],
            "CSN2": ["they leave their belongings around", -1],
            "CSN3": ["they pay attention to details", 1],
            "CSN4": ["they make a mess of things", -1],
            "CSN5": ["they get chores done right away", 1],
            "CSN6": ["they often forget to put things back in their proper place", -1],
            "CSN7": ["they like order", 1],
            "CSN8": ["they shirk their duties", -1],
            "CSN9": ["they follow a schedule", 1],
            "CSN10": ["they are exacting in their work", 1],
        }

        opn_questions = {
            "OPN1": ["they have a rich vocabulary", 1],
            "OPN2": ["they have difficulty understanding abstract ideas", -1],
            "OPN3": ["they have a vivid imagination", 1],
            "OPN4": ["they are not interested in abstract ideas", -1],
            "OPN5": ["they have excellent ideas", 1],
            "OPN6": ["they do not have a good imagination", -1],
            "OPN7": ["they are quick to understand things", 1],
            "OPN8": ["they use difficult words", 1],
            "OPN9": ["they spend time reflecting on things", 1],
            "OPN10": ["they are full of ideas", 1],
        }

        questions = (
            ext_questions
            | est_questions
            | agr_questions
            | csn_questions
            | opn_questions
        )
        return questions

    def process_data(self, df_raw):
        """This fonction get the person or candidate data with a number id or a list, and return a dataframe of the person"""
        questions = self.get_questions()

        # First we want to check if the user want a certain candidate from the dataset
        # or if the user did the test so it return a list
        if isinstance(df_raw, list):
            matching = [
                "EXT1",
                "EXT2",
                "EXT3",
                "EXT4",
                "EXT5",
                "EXT6",
                "EXT7",
                "EXT8",
                "EXT9",
                "EXT10",
                "EST1",
                "EST2",
                "EST3",
                "EST4",
                "EST5",
                "EST6",
                "EST7",
                "EST8",
                "EST9",
                "EST10",
                "AGR1",
                "AGR2",
                "AGR3",
                "AGR4",
                "AGR5",
                "AGR6",
                "AGR7",
                "AGR8",
                "AGR9",
                "AGR10",
                "CSN1",
                "CSN2",
                "CSN3",
                "CSN4",
                "CSN5",
                "CSN6",
                "CSN7",
                "CSN8",
                "CSN9",
                "CSN10",
                "OPN1",
                "OPN2",
                "OPN3",
                "OPN4",
                "OPN5",
                "OPN6",
                "OPN7",
                "OPN8",
                "OPN9",
                "OPN10",
            ]
            df_raw = pd.DataFrame([df_raw], columns=[column for column in matching])

        else:
            df_raw.drop(df_raw.columns[50:107], axis=1, inplace=True)
            df_raw.drop(
                df_raw.columns[50:], axis=1, inplace=True
            )  # here 50 to remove the country
            df_raw.dropna(inplace=True)

            # Group Names and Columns
            # EXT = [column for column in df_raw if column.startswith('EXT')]
            # EST = [column for column in df_raw if column.startswith('EST')]
            # AGR = [column for column in df_raw if column.startswith('AGR')]
            # CSN = [column for column in df_raw if column.startswith('CSN')]
            # OPN = [column for column in df_raw if column.startswith('OPN')]

            # matching = EXT+EST+AGR+CSN+OPN

            # Here we update the dataframe by applying the new coefficient
        for column in df_raw.columns:
            df_raw[column] = df_raw[column] * questions[column][1]

        # reference to scoring: https://sites.temple.edu/rtassessment/files/2018/10/Table_BFPT.pdf
        df_raw["extraversion"] = df_raw.iloc[:, 0:10].sum(axis=1) + 20
        df_raw["neuroticism"] = df_raw.iloc[:, 10:20].sum(axis=1) + 38
        df_raw["agreeableness"] = df_raw.iloc[:, 20:30].sum(axis=1) + 14
        df_raw["conscientiousness"] = df_raw.iloc[:, 30:40].sum(axis=1) + 14
        df_raw["openness"] = df_raw.iloc[:, 40:50].sum(axis=1) + 8
        df_raw["name"] = df_raw.index.to_series().apply(lambda idx: "C_" + str(idx))

        return df_raw

    def to_data_point(self) -> data_point.Person:

        id = self.df.index[0]
        name = self.df["name"].values[0]

        # Reindexing dataframe
        self.df.reset_index(drop=True, inplace=True)

        self.df = self.df.drop(columns=["name"])

        # Convert to series
        ser_metrics = self.df.squeeze()

        return self.data_point_class(id=id, name=name, ser_metrics=ser_metrics)
