# %%
import pandas as pd

annotation_text = "<span style=''>{metric_name}: {data:.2f} per 90</span>"

# series with col "goals" and some dummy value
ser_plot = pd.Series({"goals": 0.5})
col = "goals"

print(annotation_text.format(metric_name="Goals", data=ser_plot[col]))
print()

# %%

description_dict = {
    "Traditional vs Secular Values": [
        "extremely secular",
        "very secular",
        "above averagely secular",
        "neither traditional nor secular",
        "above averagely traditional",
        "very traditional",
        "extremely traditional",
    ],
    "Survival vs Self-expression Values": [
        "extremely self-expression orientated",
        "very self-expression orientated",
        "above averagely self-expression orientated",
        "neither survival nor self-expression orientated",
        "some what survival orientated",
        "very survival orientated",
        "extremely survival orientated",
    ],
    "Neutrality": [
        "extremely high",
        "very high",
        "above average",
        "average",
        "below average",
        "very low",
        "extremely low",
    ],
    "Fairness": [
        "extremely high",
        "very high",
        "above average",
        "average",
        "below average",
        "very low",
        "extremely low",
    ],
    "Skeptisism": [
        "extremely high",
        "very high",
        "above average",
        "average",
        "below average",
        "very low",
        "extremely low",
    ],
    "Societal Tranquility": [
        "extremely high",
        "very high",
        "above average",
        "average",
        "below average",
        "very low",
        "extremely low",
    ],
}


# write to json
import json

with open("data/wvs/description_dict.json", "w") as f:
    json.dump(description_dict, f)

# %%

import pandas as pd

# read in "data/wvs/countries.csv"
df = pd.read_csv("data/wvs/countries.csv")
# select columns ending in "_Z"
df = df.loc[:, df.columns.str.endswith("_Z")]
# %%

import matplotlib.pyplot as plt

# for each column in df plot a histogram
for col in df.columns:
    plt.hist(df[col], bins=20)
    plt.title(col)
    plt.show()

# %%

threshold = 1.5

# construct a new df that check if the abs of each value is greater than threshold
df_abs = df.abs() > threshold
# sum the number of True values for each column
count = df_abs.sum(axis=1)

# plot the count as a histogram
plt.hist(count, bins=20)
# %%

import json

prompt_dict = {}
with open("data/wvs/prompt.json", "w") as f:
    json.dump(prompt_dict, f)

secrets_dict = {
    "GEMINI_API_KEY": "AIzaSyAC7c_jmNWYuJbvhMmNs7j23ifWZwF1VqI",
}
prompt_dict = {}
with open("data/wvs/secrets.json", "w") as f:
    json.dump(secrets_dict, f)


# %%

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
            2.5,
            1.5,
            0.5,
            -0.5,
            -1.5,
            -2.5,
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
        text_empty = f"Here is a statistical description of the core values of {tmp_country.name.capitalize()}. \n\n"

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
