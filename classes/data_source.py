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

    def select_and_filter(self, column_name, label):

        df = self.df
        selected_id = st.selectbox(label, df[column_name].unique())
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

        super().__init__()

    def get_raw_data(self):

        df = pd.read_csv("data/wvs/wave_7.csv")

        return df

    def process_data(self, df_raw):
        df_raw = df_raw.rename(columns=self.core_value_dict)

        df_raw["country"] = df_raw["country"].map(
            {
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
            }
        )

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

        return self.data_point_class(
            id=id,
            name=name,
            ser_metrics=ser_metrics,
            relevant_metrics=self.metrics,
        )
