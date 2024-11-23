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
def describe_level(
    value,
    thresholds=[1.5, 1, 0.5, -0.5, -1],
    words=["outstanding", "excellent", "good", "average", "below average", "poor"],
):
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
    return (
        metric.replace("_", " ")
        .replace(" adjusted per90", "")
        .replace("npxG", "non-penalty expected goals")
        .capitalize()
    )


def write_out_metric(metric):
    return (
        metric.replace("_", " ")
        .replace("adjusted", "adjusted for possession")
        .replace("per90", "per 90")
        .replace("npxG", "non-penalty expected goals")
        + " minutes"
    )
    return metric.replace("_"," ").replace("adjusted","adjusted for possession").replace("per90","per 90").replace("npxG","non-penalty expected goals") + " minutes"

feature_name_mapping = {
    'vertical_distance_to_center_contribution': 'vertical distance to center',
    'euclidean_distance_to_goal_contribution': 'euclidean distance to goal',
    'nearby_opponents_in_3_meters_contribution': 'nearby opponents within 3 meters',
    'opponents_in_triangle_contribution': 'number of opponents in triangle formed by shot location and goalposts',
    'goalkeeper_distance_to_goal_contribution': 'distance to goal of the goalkeeper',
    'header_contribution': 'header',
    'distance_to_nearest_opponent_contribution': 'distance to nearest opponent',
    'angle_to_goalkeeper_contribution': 'angle to goalkeepr',
    'shot_with_left_foot_contribution': 'shot taken with left foot',
    'shot_after_throw_in_contribution': 'shot after throw in',
    'shot_after_corner_contribution': 'shot after corner',
    'shot_after_free_kick_contribution': 'shot after free kick',
    'shot_during_regular_play_contribution': 'shot during regular play'

}




def describe_shot_contributions(shot_contributions, feature_name_mapping=feature_name_mapping, thresholds=None):
    text = "The contributions of the features to the xG of the shot, sorted by their magnitude from largest to smallest, are as follows:\n"

    # Default thresholds if none are provided
    thresholds = thresholds or {
        'very_large': 0.75,
        'large': 0.50,
        'moderate': 0.25,
        'low': 0.00
    }

    # Initialize a list to store contributions that are not 'match_id', 'id', or 'xG'
    valid_contributions = {}

    # Loop through the columns to select valid ones
    for feature, contribution in shot_contributions.iloc[0].items():
        if feature not in ['match_id', 'id', 'xG']:  # Skip these columns
            valid_contributions[feature] = contribution

    # Convert to Series and sort by absolute values in descending order
    sorted_contributions = (
        pd.Series(valid_contributions)
        .apply(lambda x: abs(x))
        .sort_values(ascending=False)
    )

    # Loop through the sorted contributions and categorize them based on thresholds
    for feature, contribution in sorted_contributions.items():
        # Get the original sign of the contribution
        original_contribution = valid_contributions[feature]

        # Use the feature_name_mapping dictionary to get the display name for the feature
        feature_display_name = feature_name_mapping.get(feature, feature)

        # Determine the contribution level
        if abs(contribution) > thresholds['very_large']:
            level = 'very large'
        elif abs(contribution) > thresholds['large']:
            level = 'large'
        elif abs(contribution) > thresholds['moderate']:
            level = 'moderate'
        else:
            level = 'low'

        # Distinguish between positive and negative contributions
        if original_contribution > 0:
            explanation = f"{feature_display_name} has a {level} positive contribution, which increased the xG of the shot."
        elif original_contribution < 0:
            explanation = f"{feature_display_name} has a {level} negative contribution, which reduced the xG of the shot."
        else:
            explanation = f"{feature_display_name} had no contribution to the xG of the shot."

        # Add to the text
        text += f"{explanation}\n"
    
    return text
