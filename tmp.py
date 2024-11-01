# %%
# Debug the data loading and processing

from classes.data_source import CountryStats
import copy
from classes.description import (
    CountryDescription,
)
import json

countries = CountryStats()
metrics = [m for m in countries.df.columns if m not in ["country"]]

countries.calculate_statistics(metrics=metrics)
country_names = countries.df["country"].values.tolist()

country = copy.deepcopy(countries)
country.df = country.df[country.df["country"] == "United States of America"]
country = country.to_data_point()


with open("data/wvs/description_dict.json", "r") as f:
    description_dict = json.load(f)

thresholds_dict = dict(
    (
        metric,
        [
            2,
            1,
            -1,
            -2,
        ],
    )
    for metric in metrics
)
description = CountryDescription(
    country, description_dict=description_dict, thresholds_dict=thresholds_dict
)


# %%
# Generate country specific data for evaluation

from classes.data_source import CountryStats
from classes.description import CountryDescription
import copy
import json
import pandas as pd

countries = CountryStats()

metrics = [m for m in countries.df.columns if m not in ["country"]]

countries.calculate_statistics(metrics=metrics)

country_names = countries.df["country"].values.tolist()

with open("data/wvs/description_dict.json", "r") as f:
    description_dict = json.load(f)

thresholds_dict = dict(
    (
        metric,
        [
            2,
            1,
            -1,
            -2,
        ],
    )
    for metric in metrics
)


def select_country(countries, country_name):

    countries = CountryStats()
    countries.calculate_statistics(metrics=metrics)
    # Make a copy of Players object
    country = copy.deepcopy(countries)

    # rnd = int(country.select_random()) # does not work because of page refresh!
    # Filter country by position and select a player with sidebar selectors
    country.df = country.df[country.df["country"] == country_name]
    # Return data point

    country = country.to_data_point()

    return country


texts = []
texts_empty = []
for country_name in country_names:
    try:
        tmp_country = select_country(countries, country_name)
        c_description = CountryDescription(
            tmp_country,
            description_dict=description_dict,
            thresholds_dict=thresholds_dict,
        )

        text = f"Now do the same thing with the following: ```{c_description.synthesize_text()}```"
        text_empty = f"Here is a statistical description of the core values of {tmp_country.name.capitalize()}. \n\n ``` ```"

        texts.append(text)
        texts_empty.append(text_empty)
    except:
        texts.append("")
        texts_empty.append("")
        print(f"Error with {country_name}")


# zip country names and texts into a dataframe and save
df = pd.DataFrame({"country": country_names, "text": texts, "text_empty": texts_empty})
df.to_csv("data/wvs/country_texts.csv", index=False)
# %%
