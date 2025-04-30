# Goal: Evaluate if the wordalisation accurately reflects teh synthetic text.
# Run this script to generate responses for the data points
# This code uses exponential backoff to handle rate limiting errors from the API.
# The code will also cycle through API keys if the max delay is reached.

# Required files:
# 1. secrets.json - containing a valid Gemini API key(s)
# 2. data/{ttype}_texts.csv - containing the entity names and the corresponding text to be used in 3. (see generate_data_for_evaluation.py)
# 3. prompts/prompt_v1_{ttype}.json - prompt to use for generating responses (copy of chat used in app with modification to discourage "decline to answer" responses, i.e. example with no data.)
# 4. prompts/reconstruct_v2_{ttype}.json - prompt to use for reconstructing the data from the responses 3.
# 5. data/{ttype}_ground_truth.csv - data to be used as ground truth for the factors. (see generate_data_for_evaluation.py)

# Output:
# 1. data_points.json - in per ttype subfolder.

import json
import pandas as pd
import google.generativeai as genai
import datetime
import os
import random
import time
import re
from tqdm import tqdm

path = "C:/Users/Amy/Desktop/Green_Git/twelve-gpt-educational/evaluation/"
N = 1  # Min number of (non-"None") labels per factor per data point

# read secrets.json
with open(".streamlit/secrets.json") as f:
    secrets = json.load(f)

GEMINI_API_KEYS = [
    secrets[x]
    for x in [
        "GEMINI_API_KEY_N",
        "GEMINI_API_KEY",
    ]
]

current_key = 0
genai.configure(api_key=GEMINI_API_KEYS[current_key])
GEMINI_CHAT_MODEL = "gemini-2.0-flash"
generationConfig = {
    "temperature": 1.0,
    # "topK": 1, # variable names?
    # "topP": 1,
    # "maxOutputTokens": 2048,
}


ttypes = [
    "country",
    "person",
    "player",
]  # entity types, different applications
dt = "2025-04-29"  # datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # "2025-04-05"  #

prompt_files = []
prompts = []
reconstruct_prompt_files = []
reconstruct_prompts = []
folder_names = []
entity_names_list = []
entity_texts_list = []
ground_truth_dfs = []
for ttype in ttypes:
    prompt_file = f"prompt_v1_{ttype}.json"
    # read prompt from json
    with open(path + "prompts/" + prompt_file) as f:
        msgs = json.load(f)

    prompt_files.append(prompt_file)
    prompts.append(msgs)

    ####################

    # rec_prompt_file = f"reconstruct_v1_{ttype}.json"
    # # read prompt from json
    # with open(path + "prompts/" + rec_prompt_file) as f:
    #     msgs = json.load(f)
    # reconstruct_prompt_files.append(rec_prompt_file)
    # reconstruct_prompts.append(msgs)

    rec_prompt_file = f"reconstruct_v2_{ttype}.json"
    # read prompt from json
    with open(path + "prompts/" + rec_prompt_file) as f:
        msgs = json.load(f)
    reconstruct_prompt_files.append(rec_prompt_file)
    reconstruct_prompts.append(msgs)

    ####################

    folder_name = path + f"{dt}/{prompt_file.split('.')[0]}/"
    # if folder does not exist create it
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    folder_names.append(folder_name)

    ####################

    # read country specify text from csv
    entity_texts = pd.read_csv(path + "data/" + f"{ttype}_texts.csv")  # .head(3)
    entity_names = entity_texts[ttype].tolist()  # [:3]

    entity_names_list.append(entity_names)
    entity_texts_list.append(entity_texts)

    ####################

    df_ground_truth = pd.read_csv(path + f"data/{ttype}_ground_truth.csv")

    if ttype == "player":
        cols_y = [
            "non-penalty expected goals",
            "goals",
            "assists",
            "key passes",
            "smart passes",
            "final third passes",
            "final third receptions",
            "ground duels",
            "air duels",
        ]

        cols_x = [
            "npxG_adjusted_per90",
            "goals_adjusted_per90",
            "assists_adjusted_per90",
            "key_passes_adjusted_per90",
            "smart_passes_adjusted_per90",
            "final_third_passes_adjusted_per90",
            "final_third_receptions_adjusted_per90",
            "ground_duels_won_adjusted_per90",
            "air_duels_won_adjusted_per90",
        ]

        # map column name in df_ground_truth from cols_x to cols_y
        df_ground_truth = df_ground_truth.rename(columns=dict(zip(cols_x, cols_y)))

    # rename columns to lowercase
    df_ground_truth.columns = df_ground_truth.columns.str.lower()

    ground_truth_dfs.append(df_ground_truth)


# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 2,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    max_delay: float = 60 * 5,
    errors: tuple = (Exception,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                if delay > max_delay:

                    global current_key
                    current_key = current_key + 1
                    # mod len(GEMINI_API_KEYS) to ensure current_key is within the range of GEMINI_API_KEYS
                    current_key = current_key % len(GEMINI_API_KEYS)
                    print("max delay reached switching to next key", current_key)

                print(f"Retrying in {delay} seconds...")
                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


def get_response(msgs, text):

    genai.configure(api_key=GEMINI_API_KEYS[current_key])
    model = genai.GenerativeModel(
        model_name=GEMINI_CHAT_MODEL,
        system_instruction=msgs["system_instruction"],
        generation_config=generationConfig,
    )
    msgs["content"]["parts"] = text
    chat = model.start_chat(history=msgs["history"])
    response = chat.send_message(content=msgs["content"])

    return response


@retry_with_exponential_backoff
def completions_with_backoff(**kwargs):
    return get_response(**kwargs)


# label_factor_dict = {}
# label_factor_dict["country"] = {
#     "factors": [
#         "traditional vs secular values",
#         "survival vs self-expression values",
#         "neutrality",
#         "fairness",
#         "skepticism",
#         "societal tranquility",
#     ],
#     "labels": [
#         "far above average",
#         "above average",
#         "average",
#         "below average",
#         "far below average",
#         "None",
#     ],
# }

# label_factor_dict["player"] = {
#     "factors": [
#         "non-penalty expected goals",
#         "goals",
#         "assists",
#         "key passes",
#         "smart passes",
#         "final third passes",
#         "final third receptions",
#         "ground duels",
#         "air duels",
#     ],
#     "labels": [
#         "outstanding",
#         "excellent",
#         "good",
#         "average",
#         "below average",
#         "poor",
#         "None",
#     ],
# }
# label_factor_dict["person"] = {
#     "factors": [
#         "extraversion",
#         "neuroticism",
#         "agreeableness",
#         "conscientiousness",
#         "openness",
#     ],
#     "labels": {
#         "extraversion": ["solitary and reserved", "outgoing and energetic", "None"],
#         "neuroticism": ["sensitive and nervous", "resilient and confident", "None"],
#         "agreeableness": [
#             "critical and rational",
#             "friendly and compassionate",
#             "None",
#         ],
#         "conscientiousness": [
#             "extravagant and careless",
#             "efficient and organized",
#             "None",
#         ],
#         "openness": ["inventive and curious", "consistent and cautious", "None"],
#     },
# }

label_factor_dict = {}
label_factor_dict["country"] = {
    "factors": [
        "traditional vs secular values",
        "survival vs self-expression values",
        "neutrality",
        "fairness",
        "skepticism",
        "societal tranquility",
    ],
    "labels": [
        "yes",
        "no",
        "omitted",
        "None",
    ],
}

label_factor_dict["player"] = {
    "factors": [
        "non-penalty expected goals",
        "goals",
        "assists",
        "key passes",
        "smart passes",
        "final third passes",
        "final third receptions",
        "ground duels",
        "air duels",
    ],
    "labels": [
        "yes",
        "no",
        "omitted",
        "None",
    ],
}
label_factor_dict["person"] = {
    "factors": [
        "extraversion",
        "neuroticism",
        "agreeableness",
        "conscientiousness",
        "openness",
    ],
    "labels": [
        "yes",
        "no",
        "omitted",
        "None",
    ],
}

for t in ttypes:
    if t not in label_factor_dict.keys():
        raise ValueError(
            f"{t} not in label_factor_dict, please add factors and labels for your application."
        )


def match_re(text):
    pattern = re.compile(r"{.*?}", re.DOTALL)
    match = pattern.findall(text)

    text = match[0] if match else None
    return text


def extract_metrics(text):
    text = text.replace("'", "")
    text = text.replace('"', "")
    text = text.replace("\n", "")
    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.replace("{", "")
    text = text.replace("}", "")
    text = text.split(",")
    metrics = {}

    try:
        for t in text:
            key, value = t.split(":")
            metrics[key.strip().lower()] = value.strip()

        return metrics
    except:
        return {}


def get_metrics(entity, text, labels, factors):
    text = match_re(text)
    if not text:
        metrics = {}
    else:
        metrics = extract_metrics(text)

    if isinstance(labels, dict):
        metrics = [(f, metrics.get(f, "None")) for f in factors]
        metrics = {f: m if m in labels[f] else "None" for f, m in metrics}
    else:
        metrics = [(f, metrics.get(f, "None")) for f in factors]
        metrics = {f: m if m in labels else "None" for f, m in metrics}

    return metrics


start = time.time()
to_do = 0
max_retries = 1  # max number of tries to generate a response

for (
    ttype,
    entity_names,
    entity_texts,
    msg,
    msg_rec,
    ground_truth_df,
    folder_name,
) in zip(
    ttypes,
    entity_names_list,
    entity_texts_list,
    prompts,
    reconstruct_prompts,
    ground_truth_dfs,
    folder_names,
):
    # if data points already exist, load them
    if os.path.exists(folder_name + "data_points.json"):
        with open(folder_name + "data_points.json") as f:
            data_points = json.load(f)
    else:
        data_points = {}

    for name in tqdm(entity_names):
        for t, tt in zip(
            ["text_empty", "text", "table"], ["control", "textual", "numerical"]
        ):

            count = -1
            factors = label_factor_dict[ttype]["factors"]
            keys = [k for k in data_points.keys() if name + "_" + tt in k]
            factor_counts = {
                factor: sum(
                    [
                        1 if data_points[k][factor + "_pred"] != "None" else 0
                        for k in keys
                    ]
                )
                for factor in factors
            }

            #####################
            to_do_list = [N - factor_counts[factor] for factor in factors]
            to_do += max(to_do_list)

            # if any([x < N for x in factor_counts.values()]):
            #     print(name, tt)
            #     print(factor_counts)

            continue
            #####################

            text = entity_texts[entity_texts[ttype] == name][t].values[0]
            fact = entity_texts[entity_texts[ttype] == name]["description_text"].values[
                0
            ]

            # while the factor count of any factor is less than 5 continue to generate responses
            while any([x < N for x in factor_counts.values()]) and count < max_retries:
                count += 1
                if "_".join([name, tt, str(count)]) in keys:
                    # print("_".join([name, tt, str(count)]), "already completed")
                    pass
                else:

                    #####################
                    response = completions_with_backoff(
                        msgs=msg,
                        text=text,
                    )

                    hyp = response.candidates[0].content.parts[0].text
                    response_text = f"Factual statistical description:\n{fact}\n\nDescriptive text:\n{hyp}\n"

                    response_rec = completions_with_backoff(
                        msgs=msg_rec,
                        text=response_text,
                    )
                    response_text_rec = response_rec.candidates[0].content.parts[0].text

                    metrics = get_metrics(
                        entity=name,
                        text=response_text_rec,
                        labels=label_factor_dict[ttype]["labels"],
                        factors=factors,
                    )
                    #####################
                    # response_text = ""
                    # response_text_rec = ""
                    # hyp = ""
                    # metrics = {
                    #     f: ground_truth_df[ground_truth_df[ttype] == name][f].values[0]
                    #     for f in factors
                    # }
                    #####################

                    # increase factor counts if the response is not None
                    for factor in factors:
                        if metrics[factor] != "None":
                            factor_counts[factor] += 1

                    data_points["_".join([name, tt, str(count)])] = {
                        "entity": name,
                        "type": tt,
                        "count": count,
                        "text": text,
                        "wordalisation": hyp,
                        "description": fact,
                        "response_dict": response_text_rec,
                    }
                    for factor in factors:
                        data_points["_".join([name, tt, str(count)])][
                            factor + "_true"
                        ] = ground_truth_df[ground_truth_df[ttype] == name][
                            factor
                        ].values[
                            0
                        ]
                        data_points["_".join([name, tt, str(count)])][
                            factor + "_pred"
                        ] = metrics[factor]

            # save data_points to json
            with open(folder_name + "data_points.json", "w") as f:
                json.dump(data_points, f)

            # if count >= max_retries:
            #     print("Max tries reached.", "_".join([name, tt, str(count)]))

end = time.time()
print(f"Time taken: {end-start} seconds")
print(to_do)
