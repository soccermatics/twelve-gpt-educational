# Give the correct gender words
def pronouns(gender):
    if gender.lower() == "male":
        subject_p, object_p, possessive_p = "he", "him", "his"
    else:
        subject_p, object_p, possessive_p = "she", "her", "her"

    return subject_p, object_p, possessive_p

def article(word):
    if word.strip()[0].lower() in ["a", "e", "i", "o", "u"]:
        return "An"
    else:
        return "A"
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


# look up formated metric name for display and descriptions
def lookup_metric(metric, parameters):
    # for column name look up Explanation in the parameters
    explanation = parameters.loc[parameters['Parameter'] == metric, 'Explanation'].values[0]
    explanation.strip().capitalize()
    return explanation

def write_out_metric(metric):
    return (
        metric.replace("_", " ")
        .replace("adjusted", "adjusted for possession")
        .replace("per90", "per 90")
        .replace("npxG", "non-penalty expected goals")
        + " minutes"
    )
def describe_contributions(value,  thresholds = [10, 5, 2, -2,-5,-10], words = ["implies a seriously increased risk", "implies an increased risk", "implies a small increase in risk", "does not significantly effect the risk", "implies slightly smaller risk", "implies a decreased risk", "implies a greatly decreased risk"]):

    return describe(thresholds, words, value)

def format_numbers(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    else:
        return round(value, 2)
    