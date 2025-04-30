# %%

import json
import pandas as pd
import openpyxl
import os


DFS = []
data = {}
for ttype in ["player", "person", "country"]:
    JSON_FILE_PATH = f"C:/Users/Amy/Desktop/Green_Git/twelve-gpt-educational/evaluation/2025-04-29/prompt_v1_{ttype}/data_points.json"  # <--- CHANGE THIS TO YOUR JSON FILE NAME
    XLSX_FILE_PATH = f"C:/Users/Amy/Desktop/Green_Git/twelve-gpt-educational/evaluation/spreadsheets/2025-04-05_{ttype}.xlsx"  # <--- Name for the new Google Sheet

    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        tmp = json.load(f)

    for k in tmp.keys():
        tmp[k]["type"] = ttype

    # print("len data:", len(tmp))
    # add tmp to data dict
    data.update(tmp)

    dfs = []
    for j in range(4):
        df = pd.read_excel(
            XLSX_FILE_PATH, engine="openpyxl", sheet_name=f"{ttype}_{j+1}"
        )
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df[["id", "factor", "evaluation"]]
    df["entity"] = ttype

    # print("len df:", len(df), len(df["id"].unique()))

    DFS.append(df)


df = pd.concat(DFS, ignore_index=True)
# merge DATA dicts
# print(len(df), len(df["id"].unique()))
print(f"data evaluated: {len(df['id'].unique()) / len(data):.2f}")
# print(df.columns)

# drop row with nan in evaluation column
df = df.dropna(subset=["evaluation"])
# drop rows where evaluation is "omitted"
df = df[df["evaluation"] != "omitted"]

# print(df["evaluation"].unique())
assert set(df["evaluation"].unique()) == set(["yes", "no"])

df["evaluation"] = df["evaluation"].map({"yes": 1, "no": 0})

# group by id and factor and aggregate evaluation column, if one or more "yes" set to "yes"
dfg = (
    df.groupby(["entity", "id", "factor"])
    .agg({"evaluation": lambda x: 1 if (x == 1).any() else 0})
    .reset_index()
)
# %%

# get unique factors grouped by entity
dfg["factor"] = dfg["factor"].astype(str)
dfg["entity"] = dfg["entity"].astype(str)
dfg["id"] = dfg["id"].astype(str)

factors_dict = {}
for ttype in ["player", "person", "country"]:
    factors = dfg[dfg["entity"] == ttype]["factor"].unique()
    factors_dict[ttype] = factors.tolist()

# %%

IDS = []
FACTORS = []
EVSL_LLM = []
ENTITY = []
for k in data.keys():

    for f in data[k].keys():
        if f.endswith("pred"):
            IDS.append(k)
            ENTITY.append(data[k]["type"])
            FACTORS.append(f.split("_")[0])
            EVSL_LLM.append(data[k][f])

dfd = pd.DataFrame(
    {"entity": ENTITY, "id": IDS, "factor": FACTORS, "evaluation_LLM": EVSL_LLM}
)

dfd["factor"] = dfd["factor"].astype(str)
dfd["id"] = dfd["id"].astype(str)
dfd["factor"] = dfd["factor"].astype(str)

dfd = dfd[dfd["evaluation_LLM"] != "omitted"]
dfd = dfd[dfd["evaluation_LLM"] != "None"]

# print(dfd["evaluation_LLM"].unique())
assert set(dfd["evaluation_LLM"].unique()) == set(["yes", "no"])


dfd["evaluation_LLM"] = dfd["evaluation_LLM"].map({"yes": 1, "no": 0})

# %%

# merge dfd and dfg on id and factor
dfm = pd.merge(dfd, dfg, on=["entity", "id", "factor"], how="outer")

# %%

missing = []
# print ids in data.keys() but not in dfm['id']
for k in data.keys():
    if k not in dfm["id"].unique():
        missing.append(k)

print("Resolve these issues:")
print(len(missing), "missing ids")
print("missing ids:", missing)
# print(len(data) - len(dfm["id"].unique()))

# %%

print("Full data")
dfm.head()
# %%

dfd_agg = (
    dfd.groupby(["entity", "id"])
    .agg({"evaluation_LLM": lambda x: 1 if (x == 1).any() else 0})
    .reset_index()
)
dfg_agg = (
    dfg.groupby(["entity", "id"])
    .agg({"evaluation": lambda x: 1 if (x == 1).any() else 0})
    .reset_index()
)

dfm_agg = pd.merge(dfd_agg, dfg_agg, on=["entity", "id"], how="inner")
dfm_agg_outer = pd.merge(dfd_agg, dfg_agg, on=["entity", "id"], how="outer")
# %%

# print(dfm_agg[dfm_agg["evaluation"]!=dfm_agg["evaluation_LLM"]]['id'].to_list())
print("Resolve these issues:")
issues = dfm[
    dfm["id"].isin(
        dfm_agg[dfm_agg["evaluation"] != dfm_agg["evaluation_LLM"]]["id"].to_list()
    )
]
issues[issues["evaluation"] != issues["evaluation_LLM"]]
# %%

print("Aggregated")
dfm_agg_outer.head()
# %%

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# confusion matrix from "evaluation" and "evaluation_LLM"
cm = confusion_matrix(
    dfm_agg["evaluation"], dfm_agg["evaluation_LLM"], labels=[0, 1], normalize="true"
)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["no", "yes"],
)
disp.plot(cmap=plt.cm.Blues, values_format=".2f")
# %%

dfd_agg["type"] = dfd_agg["id"].apply(lambda x: x.split("_")[-2])

# group by type and entity and compute the mean evaluation_LLM
tmp = dfd_agg.groupby(["type", "entity"]).agg({"evaluation_LLM": "mean"}).reset_index()

dfg_agg["type"] = dfg_agg["id"].apply(lambda x: x.split("_")[-2])

# group by type and entity and compute the mean evaluation_LLM
tmp_2 = dfg_agg.groupby(["type", "entity"]).agg({"evaluation": "mean"}).reset_index()

# merge tmp and tmp_2 on type and entity
h_rates = pd.merge(tmp, tmp_2, on=["type", "entity"], how="outer")

# %%

print("Hallucination rates")
h_rates
# %%
