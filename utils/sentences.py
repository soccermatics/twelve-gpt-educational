# Give the correct gnder words
import numpy as np
import pandas as pd
def pronouns(gender):
    if gender.lower() == "male":
        subject_p, object_p, possessive_p = "he", "him", "his"
    else:
        subject_p, object_p, possessive_p = "she", "her", "her"

    return subject_p, object_p, possessive_p

# Describe the level of a metric in words
def describe_level(value):
    thresholds = [1.5, 1, 0.5, -0.5, -1]
    words = ["outstanding", "excellent", "good", "average", "below average", "poor"]
    return describe(thresholds, words, value)

def describe(thresholds, words, value):
    """
    thresholds = lower bound of each word in descending order\n
    len(words) = len(thresholds) + 1
    """
    assert len(words) == len(thresholds) + 1, "Issue with thresholds and words"
    i = 0
    while i < len(thresholds) and value < thresholds[i]:
        i += 1

    return words[i]

# Format the metrics for display and descriptions
def format_metric(metric):
    return metric.replace("_", " ").replace(" adjusted per90", "").replace("npxG","non-penalty expected goals").capitalize()

def write_out_metric(metric):
    return metric.replace("_"," ").replace("adjusted","adjusted for possession").replace("per90","per 90").replace("npxG","non-penalty expected goals") + " minutes"



def describe_shot_contributions(shot_contributions):
    text = "The contributions of the features to the xG of the shot are as follows:\n"

    # Filter out non-numeric contributions (like 'shot_id')
    numeric_contributions = shot_contributions.select_dtypes(include=[np.number])

    for feature, contribution in numeric_contributions.items():
        # Convert each contribution to a scalar if it's a single value Series
        if isinstance(contribution, pd.Series):
            contribution = contribution.item()  # Extract scalar value from the Series
        
        text += f"{feature}: {contribution:.2f}\n"  # Format the contribution with 2 decimal places

    return text

