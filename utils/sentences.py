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



def describe_xg(xG):

    if xG < 0.028723: # 25% percentile
        description = "This was a slim chance of scoring."
    elif xG < 0.056474: # 50% percentile
        description = "This was a low chance of scoring."
    elif xG < 0.096197: # 75% percentile
        description = "This was a decent chance."
    elif xG < 0.3: # very high
        description = "This was a high-quality chance, with a good probability of scoring."
    else:
        description = "This was an excellent chance."
    
    return description


# In sentences.py or wherever you manage your sentences module

def describe_shot_features(features):
    descriptions = []

    # Binary features description
    #if features['header'] == 1:
        #descriptions.append("The shot was a header.")
    if features['shot_with_left_foot'] == 1:
            descriptions.append("The shot was with the left foot.")    
    else:
        descriptions.append("The shot was with the right foot.")

    if features['shot_during_regular_play'] == 1:
        descriptions.append("The shot was taken during open play.")
    else:
        if features['shot_after_throw_in'] == 1:
            descriptions.append("The shot was taken after a throw-in.")
        elif features['shot_after_corner'] == 1:
            descriptions.append("The shot was taken after a corner.")
        elif features['shot_after_free_kick'] == 1:
            descriptions.append("The shot was taken after a free-kick.")
        else:    
            descriptions.append("The shot was taken from a set-piece.")    

    # Continuous features description
    if features['vertical_distance_to_center'] < 2.805:
        descriptions.append("It was taken from very close to the center of the pitch.")
    elif features['vertical_distance_to_center'] < 9.647:
        descriptions.append("It was taken reasonably centrally.")
    else:
        descriptions.append("It was taken quite a long way from the centre of the pitch.")

    if features['euclidean_distance_to_goal'] < 10.278:
        descriptions.append("It was taken from a close range, near the goal.")
    elif features['euclidean_distance_to_goal'] < 21.116:
        descriptions.append("It was taken from a moderate distance from the goal.")
    else:
        descriptions.append("It was taken from long range, far from the goal.")

    if features['nearby_opponents_in_3_meters'] < 1:
        descriptions.append("It was taken with little or no pressure from opponents.")
    elif features['nearby_opponents_in_3_meters'] < 2:
        descriptions.append("It was taken with moderate pressure, with one opponent within 3 meters.")
    else:
        descriptions.append("It was taken under heavy pressure, with several opponents within 3 meters.")

    if features['opponents_in_triangle'] < 1:
        descriptions.append("it was taken with no oppositions between the shooter and the goals.")
    elif features['opponents_in_triangle'] < 2:
        descriptions.append("There were some opposition players blocking the path, but there was spac for a well-placed shot.")
    else:
        descriptions.append("There we multiple opponents blocking the path.")

    if features['goalkeeper_distance_to_goal'] < 1.649:
        descriptions.append("The goalkeeper was very close to the goal.")
    elif features['goalkeeper_distance_to_goal'] < 3.217:
        descriptions.append("The goalkeeper was at a moderate distance from the goal.")
    else:
        descriptions.append("The goalkeeper was positioned far from the goal.")

    if features['distance_to_nearest_opponent'] < 1.119:
        descriptions.append("The shot was taken with strong pressure from a very close opponent.")
    elif features['distance_to_nearest_opponent'] < 1.779:
        descriptions.append("The shot was taken with moderate pressure from an opponent nearby.")
    else:
        descriptions.append("The shot was taken with no immediate pressure from any close opponent, with the nearest opponent far away.")

    if features['angle_to_goalkeeper'] < -23.36:
        descriptions.append("The shot was taken from a broad angle towards goalkeeper being on left, making it difficult to score.")
    elif features['angle_to_goalkeeper'] < 22.72:
        descriptions.append("The shot was taken from a relatively good angle, allowing for a decent chance.")
    else:
        descriptions.append("The shot was taken from a broad angle towards the goalkeeper being on right.")

    return descriptions

def describe_shot_single_feature(feature_name, feature_value):
    # Describe binary features
    if feature_name == 'header':
        return "the shot was a header." if feature_value == 1 else "the shot was not a header."
    if feature_name == 'shot_with_left_foot':
        return "the shot was with the left foot." if feature_value == 1 else "the shot was with the right foot."
    if feature_name == 'shot_during_regular_play':
        return "the shot was taken during regular play." if feature_value == 1 else "the shot was taken from a set-piece."
    if feature_name == 'shot_after_throw_in':
        return "the shot was taken after a throw-in." if feature_value == 1 else "the shot was not taken after a throw-in."
    if feature_name == 'shot_after_corner':
        return "the shot was taken after a corner." if feature_value == 1 else "the shot was not taken after a corner."
    if feature_name == 'shot_after_free_kick':
        return "the shot was taken after a free-kick." if feature_value == 1 else "the shot was not taken after a free-kick."

    # Describe continuous features
    if feature_name == 'vertical_distance_to_center':
        if feature_value < 2.805:
            return "the shot was taken closer to the center of the pitch (less vertical distance)."
        elif feature_value < 9.647:
            return "the shot was taken from an intermediate vertical distance."
        else:
            return "the shot was taken far from the center, closer to the touchline."
    if feature_name == 'euclidean_distance_to_goal':
        if feature_value < 10.278:
            return "the shot was taken from a close range, near the goal."
        elif feature_value < 21.116:
            return "the shot was taken from a moderate distance to the goal."
        else:
            return "the shot was taken from a long range, far from the goal."
    if feature_name == 'nearby_opponents_in_3_meters':
        if feature_value < 1:
            return "the shot was taken with little to no pressure from opponents within 3 meters."
        elif feature_value < 2:
            return "the shot was taken with moderate pressure, with some opponents nearby within 3 meters."
        else:
            return "the shot was taken under heavy pressure, with several opponents close by within 3 meters."
    if feature_name == 'opponents_in_triangle':
        if feature_value < 1:
            return "the shot was taken with minimal opposition in the shooting triangle."
        elif feature_value < 2:
            return "the shot was taken with some opposition blocking the path."
        else:
            return "the shot was heavily contested, with multiple opponents blocking the path."
    if feature_name == 'goalkeeper_distance_to_goal':
        if feature_value < 1.649:
            return "the goalkeeper was very close to the goal."
        elif feature_value < 3.217:
            return "the goalkeeper was at a moderate distance from the goal."
        else:
            return "the goalkeeper was positioned far from the goal."
    if feature_name == 'distance_to_nearest_opponent':
        if feature_value < 1.119:
            return "the shot was taken with strong pressure from a very close opponent."
        elif feature_value < 1.779:
            return "the shot was taken with moderate pressure from an opponent nearby."
        else:
            return "the shot was taken with no immediate pressure, as the nearest opponent was far away."
    if feature_name == 'angle_to_goalkeeper':
        if feature_value < -23.36:
            return "the shot was taken from a broad angle to the left of the goalkeeper, making it difficult to score."
        elif feature_value < 22.72:
            return "the shot was taken from a good angle, providing a decent chance to score."
        else:
            return "the shot was taken from a broad angle to the right of the goalkeeper."
    
    # Default case if the feature is unrecognized
    return f"No description available for {feature_name}."




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
def describe_shot_contributions(shot_contributions, shot_features, feature_name_mapping=feature_name_mapping):
    text = "The contributions of the features to the xG of the shot, sorted by their magnitude from largest to smallest, are as follows:\n"
    
    # Extract the contributions from the shot_contributions DataFrame
    contributions = shot_contributions.iloc[0].drop(['match_id', 'id', 'xG'])  # Drop irrelevant columns
    
    # Sort the contributions by their absolute value (magnitude) in descending order
    sorted_contributions = contributions.abs().sort_values(ascending=False)
    
    # Get the top 4 contributions
    top_contributions = sorted_contributions.head(4)
    
    # Loop through the top contributions to generate descriptions
    for idx, (feature, contribution) in enumerate(top_contributions.items()):

        # Get the original sign of the contribution
        original_contribution = contributions[feature]
        
        # Remove "_contribution" suffix to match feature names in shot_features
        feature_name = feature.replace('_contribution', '')
        
        # Use feature_name_mapping to get the display name for the feature (if available)
        feature_display_name = feature_name_mapping.get(feature, feature)
        
        # Get the feature value from shot_features
        feature_value = shot_features[feature_name]
        
        # Get the feature description
        feature_value_description = describe_shot_single_feature(feature_name, feature_value)
        
        # Add the feature's contribution to the xG description
        if original_contribution > 0:
            impact = 'maximum positive contribution'
            impact_text = "increased the xG of the shot."
        elif original_contribution < 0:
            impact = 'maximum negative contribution'
            impact_text = "reduced the xG of the shot."
        else:
            impact = 'no contribution'
            impact_text = "had no impact on the xG of the shot."

        # Use appropriate phrasing for the first feature and subsequent features
        if idx == 0:
            text += f"\nThe most impactful feature is {feature_display_name}, which had the {impact} because {feature_value_description}. This feature {impact_text}"
        else:
            text += f"\nAnother impactful feature is {feature_display_name}, which had the {impact} because {feature_value_description} This feature {impact_text}"
        

    return text




def describe_shot_contributions1(shot_contributions, feature_name_mapping=feature_name_mapping, thresholds=None):
    
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
