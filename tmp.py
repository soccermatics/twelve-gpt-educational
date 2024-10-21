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
