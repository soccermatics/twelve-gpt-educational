# %%
# # Debug the data loading and processing

# from classes.data_source import CountryStats
# import copy
# from classes.description import (
#     CountryDescription,
# )
# import json

# countries = CountryStats()
# metrics = [m for m in countries.df.columns if m not in ["country"]]

# countries.calculate_statistics(metrics=metrics)
# country_names = countries.df["country"].values.tolist()

# country = copy.deepcopy(countries)
# country.df = country.df[country.df["country"] == "United States of America"]
# country = country.to_data_point()


# with open("../data/wvs/description_dict.json", "r") as f:
#     description_dict = json.load(f)

# thresholds_dict = dict(
#     (
#         metric,
#         [
#             2,
#             1,
#             -1,
#             -2,
#         ],
#     )
#     for metric in metrics
# )
# description = CountryDescription(
#     country, description_dict=description_dict, thresholds_dict=thresholds_dict
# )


# %%


def series_to_markdown(
    series, questions, header="| Factor | Z-score | Relevant question |"
):
    if questions is None:
        separator = "|:------|------:|"
        rows = [
            f"| {idx} | {val:.3f} |" for idx, val in zip(series.index, series.values)
        ]
    else:
        separator = "|:------|------:|:--------------------|"
        rows = [
            f"| {idx} | {val:.3f} | {q} |"
            for idx, val, q in zip(series.index, series.values, questions)
        ]
    return "\n".join([header, separator] + rows)


def get_questions(metric, c_description, entity):

    if metric.lower() in entity.drill_down_metrics:
        if entity.ser_metrics[metric + "_Z"] > 0:
            index = 1
        else:
            index = 0

        question, value = entity.drill_down_metrics[metric.lower()]
        question, value = question[index], value[index]
        description = "Question: '"
        description += c_description.relevant_questions[metric][question][0]
        description += "' Average answer:"
        description += c_description.relevant_questions[metric][question][1]
        description += " '"
        description += c_description.relevant_questions[metric][question][2][str(value)]
        description += "' "
        description += c_description.relevant_questions[metric][question][3]
        description += "."

    elif metric in entity.drill_down_metrics:

        if entity.ser_metrics[metric + "_Z"] > 0:
            index = 1
        else:
            index = 0

        question, value = entity.drill_down_metrics[metric]
        question, value = question[index], value[index]
        description = "Question: '"
        description += c_description.relevant_questions[metric][question][0]
        description += "' Average answer: "
        description += c_description.relevant_questions[metric][question][1]
        description += " '"
        description += c_description.relevant_questions[metric][question][2][str(value)]
        description += "' "
        description += c_description.relevant_questions[metric][question][3]
        description += "."
    else:
        description = ""
    return description


# %%

# Generate country specific data for evaluation

from classes.data_source import CountryStats
from classes.description import CountryDescription
import copy
import json
import pandas as pd
from tqdm import tqdm

countries = CountryStats()

metrics = [m for m in countries.df.columns if m not in ["country"]]

countries.calculate_statistics(metrics=metrics)

country_names = countries.df["country"].values.tolist()

with open("./data/wvs/description_dict.json", "r") as f:
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
tables = []
d_texts = []
for country_name in tqdm(country_names):
    # try:
    tmp_country = select_country(countries, country_name)
    c_description = CountryDescription(
        tmp_country,
        description_dict=description_dict,
        thresholds_dict=thresholds_dict,
    )

    text = f"```{c_description.synthesize_text()}```"
    data = c_description.country.ser_metrics
    # select only rows ending in "_Z"
    data = data[[col for col in data.index if col.endswith("_Z")]]
    # remove "_Z" from the index
    data.index = [col[:-2] for col in data.index]
    questions = [
        get_questions(x, c_description, c_description.country) for x in data.index
    ]

    table = f"```Here is a statistical description of the societal values of {tmp_country.name.capitalize()}.\n{series_to_markdown(data, questions)}```"
    text_empty = f"```Here is a statistical description of the societal values of {tmp_country.name.capitalize()}.```"
    d_text = f"```{c_description.synthesize_text()}\n{series_to_markdown(data, questions)}```"

    texts.append(text)
    tables.append(table)
    texts_empty.append(text_empty)
    d_texts.append(d_text)
    # except:
    #     texts.append("")
    #     tables.append("")
    #     texts_empty.append("")
    #     print(f"Error with {country_name}")

# %%

# zip country names and texts into a dataframe and save
df = pd.DataFrame(
    {
        "country": country_names,
        "text": texts,
        "table": tables,
        "text_empty": texts_empty,
        "description_text": d_texts,
    }
)
df.to_csv("evaluation/data/country_texts.csv", index=False)
# %%

# Generate country specific ground truth

from classes.data_source import CountryStats
from classes.description import CountryDescription
import copy
import json
import pandas as pd
import utils.sentences as sentences

countries = CountryStats()

metrics = [m for m in countries.df.columns if m not in ["country"]]

countries.calculate_statistics(metrics=metrics)

country_names = countries.df["country"].values.tolist()

description_dict = dict(
    (
        metric,
        [
            "far above average",
            "above average",
            "average",
            "below average",
            "far below average",
        ],
    )
    for metric in metrics
)


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


factors = []
for country_name in tqdm(country_names):
    tmp_country = select_country(countries, country_name)
    data = [tmp_country.name]
    for metric in tmp_country.relevant_metrics:
        try:
            text = sentences.describe_level(
                tmp_country.ser_metrics[metric + "_Z"],
                thresholds=thresholds_dict[metric],
                words=description_dict[metric],
            )
            data.append(text)
        except:
            data.append(None)
            print(f"Error with {country_name}")

    factors.append(data)

# create a dataframe with columns 'country' and and each factor from tmp_country.relevant_metrics
df = pd.DataFrame(factors, columns=["country"] + tmp_country.relevant_metrics)
df.to_csv("evaluation/data/country_ground_truth.csv", index=False)
# %%
# %%
# Generate player specific data for evaluation

from classes.data_source import PlayerStats
from classes.description import PlayerDescription
import copy
import json
import pandas as pd
import utils.sentences as sentences


players = PlayerStats()

metrics = [m for m in players.df.columns if m not in ["player_name"]]

players.calculate_statistics(metrics=metrics)

player_names = players.df["player_name"].values.tolist()

description_dict = dict(
    (
        metric,
        ["outstanding", "excellent", "good", "average", "below average", "poor"],
    )
    for metric in metrics
)

thresholds_dict = dict(
    (
        metric,
        [1.5, 1, 0.5, -0.5, -1],
    )
    for metric in metrics
)


def select_player(players, player_name):

    players = PlayerStats()
    players.calculate_statistics(metrics=metrics)
    # Make a copy of Players object
    player = copy.deepcopy(players)

    # rnd = int(player.select_random()) # does not work because of page refresh!
    # Filter player by position and select a player with sidebar selectors
    player.df = player.df[player.df["player_name"] == player_name]
    # Return data point

    player = player.to_data_point(gender="male", position="Forward")

    return player


name_map = {
    "npxG_adjusted_per90": "non-penalty expected goals",
    "goals_adjusted_per90": "goals",
    "assists_adjusted_per90": "assists",
    "key_passes_adjusted_per90": "key passes",
    "smart_passes_adjusted_per90": "smart passes",
    "final_third_passes_adjusted_per90": "final third passes",
    "final_third_receptions_adjusted_per90": "final third receptions",
    "ground_duels_won_adjusted_per90": "ground duels",
    "air_duels_won_adjusted_per90": "air duels",
}


texts = []
texts_empty = []
tables = []
d_texts = []
for player_name in tqdm(player_names):
    # try:
    tmp_player = select_player(players, player_name)
    c_description = PlayerDescription(
        tmp_player,
    )

    text = f"```{c_description.synthesize_text()}```"
    data = c_description.player.ser_metrics
    # select only rows ending in "_Z"
    data = data[[col for col in data.index if col.endswith("_Z")]]
    # remove "_Z" from the index
    data.index = [col[:-2] for col in data.index]
    # rename index to be more readable
    data.index = [name_map.get(col, col) for col in data.index]
    questions = None

    table = f"```Here is a statistical description of {tmp_player.name}.\n{series_to_markdown(data, questions, header='| Factor | Z-score |')}```\nIf no information is provided answer anyway, using your prior statistical knowledge."
    text_empty = f"```Here is a statistical description of {tmp_player.name}.```\nIf no information is provided answer anyway, using your prior statistical knowledge."
    d_text = f"```{c_description.synthesize_text()}\n{series_to_markdown(data, questions, header='| Factor | Z-score |')}```"
    texts.append(text)
    tables.append(table)
    texts_empty.append(text_empty)
    d_texts.append(d_text)
    # except:
    #     texts.append("")
    #     texts_empty.append("")
    #     print(f"Error with {player_name}")


# zip country names and texts into a dataframe and save
df = pd.DataFrame(
    {
        "player": player_names,
        "text": texts,
        "table": tables,
        "description_text": d_texts,
        "text_empty": texts_empty,
    }
)
df.to_csv("evaluation/data/player_texts.csv", index=False)

factors = []
for name in tqdm(player_names):
    tmp_player = select_player(players, name)
    data = [tmp_player.name]
    for metric in tmp_player.relevant_metrics:
        try:
            text = sentences.describe_level(
                tmp_player.ser_metrics[metric + "_Z"],
                thresholds=thresholds_dict[metric],
                words=description_dict[metric],
            )
            data.append(text)
        except:
            data.append(None)
            print(f"Error with {name}")

    factors.append(data)

# create a dataframe with columns 'player' and and each factor from tmp_player.relevant_metrics
df = pd.DataFrame(factors, columns=["player"] + tmp_player.relevant_metrics)
df.to_csv("evaluation/data/player_ground_truth.csv", index=False)
# %%

# Generate player specific data for evaluation

from classes.data_source import PersonStat
from classes.description import PersonDescription
import copy
import json
import pandas as pd
import utils.sentences as sentences
from tqdm import tqdm

people = PersonStat()

metrics = [m for m in people.df.columns if m not in ["name"]]

people.calculate_statistics(metrics=metrics)

people_names = people.df["name"].values.tolist()


def select_person(people, player_name):

    people = PersonStat()
    people.calculate_statistics(metrics=metrics)

    person = copy.deepcopy(people)

    person.df = person.df[person.df["name"] == player_name]

    person = person.to_data_point()

    return person


def get_questions(metric, c_description, entity):
    questions = PersonStat().get_questions()
    description = " "

    if metric == "extraversion":
        if entity.ser_metrics[metric + "_Z"] > 1:
            index = entity.ser_metrics[0:10].idxmax()
            description = "In particular they said that " + questions[index][0] + "."
        if entity.ser_metrics[metric + "_Z"] < -1:
            index = entity.ser_metrics[0:10].idxmin()
            description = "In particular they said that " + questions[index][0] + "."
    elif metric == "neuroticism":
        if entity.ser_metrics[metric + "_Z"] > 1:
            index = entity.ser_metrics[10:20].idxmax()
            description = "In particular they said that " + questions[index][0] + ". "
        if entity.ser_metrics[metric + "_Z"] < -1:
            index = entity.ser_metrics[10:20].idxmin()
            description = "In particular they said that " + questions[index][0] + "."
    elif metric == "agreeableness":
        if entity.ser_metrics[metric + "_Z"] > 1:
            index = entity.ser_metrics[20:30].idxmax()
            description = "In particular they said that " + questions[index][0] + "."
        if entity.ser_metrics[metric + "_Z"] < -1:
            index = entity.ser_metrics[20:30].idxmin()
            description = "In particular they said that " + questions[index][0] + "."
    elif metric == "conscientiousness":
        if entity.ser_metrics[metric + "_Z"] > 1:
            index = entity.ser_metrics[30:40].idxmax()
            description = "In particular they said that " + questions[index][0] + "."
        if entity.ser_metrics[metric + "_Z"] < -1:
            index = entity.ser_metrics[30:40].idxmin()
            description = "In particular they said that " + questions[index][0] + "."
    elif metric == "openness":
        if entity.ser_metrics[metric + "_Z"] > 1:
            index = entity.ser_metrics[40:50].idxmax()
            description = "In particular they said that " + questions[index][0] + "."
        if entity.ser_metrics[metric + "_Z"] < -1:
            index = entity.ser_metrics[40:50].idxmin()
            description = "In particular they said that " + questions[index][0] + "."

    return description


texts = []
texts_empty = []
tables = []
d_texts = []
for player_name in tqdm(people_names):
    # try:
    tmp_person = select_person(people, player_name)
    c_description = PersonDescription(
        tmp_person,
    )

    text = f"```{c_description.synthesize_text()}```"
    data = c_description.person.ser_metrics
    # select only rows ending in "_Z"
    cols = [
        "extraversion_Z",
        "neuroticism_Z",
        "agreeableness_Z",
        "conscientiousness_Z",
        "openness_Z",
    ]
    data = data[[col for col in data.index if col in cols]]
    # remove "_Z" from the index
    data.index = [col[:-2] for col in data.index]
    questions = [
        get_questions(x, c_description, c_description.person) for x in data.index
    ]

    table = f"```Here is a statistical description of the candidate.\n{series_to_markdown(data, questions)}```"
    text_empty = f"```The candidate is...```"
    d_text = f"```{c_description.synthesize_text()}\n{series_to_markdown(data, questions)}```"

    texts.append(text)
    tables.append(table)
    texts_empty.append(text_empty)
    d_texts.append(d_text)
    # except:
    #     texts.append("")
    #     texts_empty.append("")
    #     print(f"Error with {player_name}")


# zip country names and texts into a dataframe and save
df = pd.DataFrame(
    {
        "person": people_names,
        "text": texts,
        "text_empty": texts_empty,
        "table": tables,
        "description_text": d_texts,
    }
)
df.to_csv("evaluation/data/person_texts.csv", index=False)


factors = []
for name in tqdm(people_names):
    tmp_person = select_person(people, name)
    data = [tmp_person.name]

    # extraversion
    extraversion = tmp_person.ser_metrics["extraversion_Z"]
    cat_0 = "solitary and reserved"
    cat_1 = "outgoing and energetic"
    if extraversion > 0:
        data.append(cat_1)
    else:
        data.append(cat_0)

    # neuroticism
    neuroticism = tmp_person.ser_metrics["neuroticism_Z"]
    cat_0 = "resilient and confident"
    cat_1 = "sensitive and nervous"
    if neuroticism > 0:
        data.append(cat_1)
    else:
        data.append(cat_0)

    # agreeableness
    agreeableness = tmp_person.ser_metrics["agreeableness_Z"]
    cat_0 = "critical and rational"
    cat_1 = "friendly and compassionate"
    if agreeableness > 0:
        data.append(cat_1)
    else:
        data.append(cat_0)

    # conscientiousness
    conscientiousness = tmp_person.ser_metrics["conscientiousness_Z"]
    cat_0 = "extravagant and careless"
    cat_1 = "efficient and organized"

    if conscientiousness > 0:
        data.append(cat_1)
    else:
        data.append(cat_0)

    # openness
    openness = tmp_person.ser_metrics["openness_Z"]
    cat_0 = "consistent and cautious"
    cat_1 = "inventive and curious"

    if openness > 0:
        data.append(cat_1)
    else:
        data.append(cat_0)

    factors.append(data)

# create a dataframe with columns 'player' and and each factor from tmp_person.relevant_metrics
df = pd.DataFrame(
    factors,
    columns=["person"]
    + ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"],
)
df.to_csv("evaluation/data/person_ground_truth.csv", index=False)

# %%
