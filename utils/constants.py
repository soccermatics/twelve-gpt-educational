"""
Constants.
"""
from collections import OrderedDict

#from settings_club import ClubSettings
import streamlit as st

# For test of pdf, when running test_report.py
# if 'user_info' not in st.session_state:
#     user_info = {
#              "app_metadata": {
#                  "club": "default"
#              }
#          }
#     settings = ClubSettings(user_info)

# else:
#     settings = ClubSettings(st.session_state.user_info)


# def get_quality_dict():
#     qualities = settings.get_scout_qualities()    
#     for quality in qualities.keys():
#         metrics = qualities[quality]["metrics"]
#         qualities[quality]["metrics"] = [
#             metric + " per 90"
#             if all(keyword not in metric for keyword in ["%", " per ", "Minutes", " eff", "_id", " - ", "(m^2)", "(m)", "position",
#                                                          "defensive area", "after defensive action"])
#             else metric
#             for metric in metrics
#         ]

#     qualities = OrderedDict(sorted(qualities.items(), key=lambda t: t[0]))
#     return qualities

def player_positions_detailed():
    return {
        'CB': 'Central Defender',
        'LB': 'Left Back',
        'RB': 'Right Back',
        'LWB': 'Left Back',
        'RWB': 'Right Back',
        'CM':'Central Midfielder',
        'DM': 'Defensive Midfielder',
        'AM': 'Attacking Midfielder',
        'RAM':"Right Winger",
        'LAM': 'Left Winger',
        'LW': 'Left Winger',
        'RW': 'Right Winger',
        'CF': 'Striker',
    }

# def get_match_quality_dict():
#     quality_dict = settings.get_match_qualities()    
#     new_dict = {}
#     for quality, value in quality_dict.items():
#         new_dict[quality] = {}
#         new_dict[quality]["metrics"] = value["metrics"]
#         sum_weights = sum([abs(w) for w in value["weights"]])
#         new_dict[quality]["weights"] = [w / sum_weights for w in value["weights"]]
#     #new_dict = OrderedDict(sorted(new_dict.items(), key=lambda t: t[0], reverse=False))
#     return new_dict

# def get_extra_quality_dict():
#     quality_dict = settings.get_extra_qualities()
#     new_dict = {}
#     for quality, value in quality_dict.items():
#         new_dict[quality] = {}
#         new_dict[quality]["metrics"] = value["metrics"]
#         sum_weights = sum([abs(w) for w in value["weights"]])
#         new_dict[quality]["weights"] = [w / sum_weights for w in value["weights"]]
#     #new_dict = OrderedDict(sorted(new_dict.items(), key=lambda t: t[0], reverse=False))
#     return new_dict

# def get_season_quality_dict():
#     quality_dict = settings.get_season_qualities()
#     new_dict = {}
#     for quality, value in quality_dict.items():
#         new_dict[quality] = {}
#         new_dict[quality]["metrics"] = value["metrics"]
#         sum_weights = sum([abs(w) for w in value["weights"]])
#         new_dict[quality]["weights"] = [w / sum_weights for w in value["weights"]]
#     #new_dict = OrderedDict(sorted(new_dict.items(), key=lambda t: t[0], reverse=False))
#     return new_dict


# def get_unique_metrics(match=False, season = False):
#     if match:
#         quality_dict = get_match_quality_dict()
#         #quality_dict.update(get_extra_quality_dict())
#     elif season:
#         quality_dict = get_season_quality_dict()
#     else:
#         quality_dict = get_quality_dict()
#     metrics = list(
#         set(
#             [
#                 metric
#                 for category in quality_dict.values()
#                 for metric in category["metrics"]
#             ]
#         )
#     )
#     return metrics


def get_pitch_zones():
    horizontal_zones = [
        "own penalty area", "defensive central zone", "defensive right wing",
        "defensive left wing", "defensive right half-space", "defensive left half-space",
        "opposition penalty area", "attacking central zone", "attacking right wing",
        "attacking left wing", "attacking right half-space", "attacking left half-space",
    ]
    vertical_zones = ["outside of the box", "left side of the box", "right side of the box", "six yard box", "golden zone"]

    own_half = [zone for zone in horizontal_zones if "defensive" in zone or "own" in zone]
    att_half = [zone for zone in horizontal_zones if "attacking" in zone or "opposition" in zone]

    res = {
        "Under pressure retention per 90": horizontal_zones,
        "xGBuildup per 90": horizontal_zones,
        "xGCreated per 90": att_half,
        "Attacking aerials won per 90": att_half,
        #"xG per box touch": vertical_zones,
        "xG per 90": vertical_zones,
        "Carries (xT) per 90": horizontal_zones,
        "Defensive intensity per 90": horizontal_zones,
        "xG per 90": vertical_zones,
        "Shot conversion %": vertical_zones,
        "Dribbles (xT) per 90": att_half,
        "Defensive actions per 90": horizontal_zones,
        "Playmaking passes per 90": horizontal_zones,
        "Ball recoveries per 90": horizontal_zones,
        "Defensive aerials won per 90": own_half,
        "Passes (xT) per 90": horizontal_zones,
        "Defensive duels won %": horizontal_zones,
        "Pressure resistance %": horizontal_zones,
    }
    return res

#PLAYER_QUALITY_DICT = get_quality_dict()

PLAYER_NEGATIVE_METRICS = [
    "Losses per 90", "High turnovers per 90", "High turnovers per low reception",
    "Opposition xT into defensive area", "Opposition pass success % into defensive area",
    "Opposition xG after defensive action", "Opposition xG from defensive area",
    "Opposition xT from defensive area", "Opposition progressive passes from defensive area %"
]

"""
Have a fixed and a logical order of qualities:
1. Involvement
2. Defensive qualities
3. Heading (start with defensive heading to follow defensive qualities)
4. Attacking qualities starting closer to own goal and ending wiht goal scoring.
    - Passing -> running -> dribbling -> creating -> scoring
"""
#POSITION_QUALITIES = settings.get_scout_position_qualities()

#The qualities are plotted reversed in radars and distribution plots, so reverse the order
#POSITION_QUALITIES = {k: list(reversed(v)) for k, v in POSITION_QUALITIES.items()}

#PLAYER_METRICS = get_unique_metrics()

#QUALITY_PITCH_METRICS = settings.get_scout_qualities_pitch_metrics()

#METRIC_PITCH_ZONES = get_pitch_zones()

PITCH_ZONES_BBOX = {
    'own penalty area': {
        'x_lower_bound': 0, 'x_upper_bound': 16,
        'y_lower_bound': 19, 'y_upper_bound': 81,
    },
    'opposition penalty area': {
        'x_lower_bound': 84, 'x_upper_bound': 100,
        'y_lower_bound': 19, 'y_upper_bound': 81,
    },
    'defensive right half-space': {
        'x_lower_bound': 16, 'x_upper_bound': 50,
        'y_lower_bound': 19, 'y_upper_bound': 37,
    },
    'defensive left half-space': {
        'x_lower_bound': 16, 'x_upper_bound': 50,
        'y_lower_bound': 63, 'y_upper_bound': 81,
    },
    'defensive central zone': {
        'x_lower_bound': 16, 'x_upper_bound': 50,
        'y_lower_bound': 37, 'y_upper_bound': 63,
    },
    'defensive right wing': {
        'x_lower_bound': 0, 'x_upper_bound': 50,
        'y_lower_bound': 0, 'y_upper_bound': 19,
    },
    'defensive left wing': {
        'x_lower_bound': 0, 'x_upper_bound': 50,
        'y_lower_bound': 81, 'y_upper_bound': 100,
    },
    'attacking right half-space': {
        'x_lower_bound': 50, 'x_upper_bound': 84,
        'y_lower_bound': 19, 'y_upper_bound': 37,
    },
    'attacking left half-space': {
        'x_lower_bound': 50, 'x_upper_bound': 84,
        'y_lower_bound': 63, 'y_upper_bound': 81,
    },
    'attacking central zone': {
        'x_lower_bound': 50, 'x_upper_bound': 84,
        'y_lower_bound': 37, 'y_upper_bound': 63,
    },
    'attacking right wing': {
        'x_lower_bound': 50, 'x_upper_bound': 100,
        'y_lower_bound': 0, 'y_upper_bound': 19,
    },
    'attacking left wing': {
        'x_lower_bound': 50, 'x_upper_bound': 100,
        'y_lower_bound': 81, 'y_upper_bound': 100,
    },
}

SHOT_ZONES_BBOX= {
    'six yard box': {
        'x_lower_bound': 94, 'x_upper_bound': 100,
        'y_lower_bound': 38, 'y_upper_bound': 62,
    },
    'golden zone': {
        'x_lower_bound': 84, 'x_upper_bound': 94,
        'y_lower_bound': 38, 'y_upper_bound': 62,
    },
    'left side of the box': {
        'x_lower_bound': 84, 'x_upper_bound': 100,
        'y_lower_bound': 19, 'y_upper_bound': 38,
    },
    'right side of the box': {
        'x_lower_bound': 84, 'x_upper_bound': 100,
        'y_lower_bound': 62, 'y_upper_bound': 81,
    },
    'outside of the box': {
        'x_lower_bound': 50, 'x_upper_bound': 84,
        'y_lower_bound': 0, 'y_upper_bound': 100
    },
}

THIRDS_ZONES_BBOX = {
    "own third": {
        "x_lower_bound": 0, "x_upper_bound": 33,
        "y_lower_bound": 0, "y_upper_bound": 100
    },
    "middle third": {
        "x_lower_bound": 33, "x_upper_bound": 67,
        "y_lower_bound": 0, "y_upper_bound": 100
    },
    "final third": {
        "x_lower_bound": 67, "x_upper_bound": 100,
        "y_lower_bound": 0, "y_upper_bound": 100
    }
}

DEF_ENTRY_ZONES_BBOX = {
    "right corner zone": {
        "x_lower_bound": 0, "x_upper_bound": 16,
        "y_lower_bound": 0, "y_upper_bound": 19
    },
    "left corner zone": {
        "x_lower_bound": 0, "x_upper_bound": 16,
        "y_lower_bound": 81, "y_upper_bound": 100
    },
    "right wing": {
        "x_lower_bound": 16, "x_upper_bound": 33,
        "y_lower_bound": 0, "y_upper_bound": 19
    },
    "left wing": {
        "x_lower_bound": 16, "x_upper_bound": 33,
        "y_lower_bound": 81, "y_upper_bound": 100
    },
    "left side of the box": {
        "x_lower_bound": 0, "x_upper_bound": 16,
        "y_lower_bound": 62, "y_upper_bound": 81
    },
    "right side of the box": {
        "x_lower_bound": 0, "x_upper_bound": 16,
        "y_lower_bound": 19, "y_upper_bound": 38
    },
    "left half-space": {
        "x_lower_bound": 16, "x_upper_bound": 33,
        "y_lower_bound": 62, "y_upper_bound": 81
    },
    "right half-space": {
        "x_lower_bound": 16, "x_upper_bound": 33,
        "y_lower_bound": 19, "y_upper_bound": 38
    },
    "golden zone": {
        "x_lower_bound": 0, "x_upper_bound": 16,
        "y_lower_bound": 38, "y_upper_bound": 62
    },
    "central zone": {
        "x_lower_bound": 16, "x_upper_bound": 33,
        "y_lower_bound": 38, "y_upper_bound": 62
    },
}

OFF_ENTRY_ZONES_BBOX_OPP = {
    zone: {
        "x_lower_bound": 100 - bbox["x_upper_bound"], "x_upper_bound": 100 - bbox["x_lower_bound"],
        "y_lower_bound": bbox["y_lower_bound"], "y_upper_bound": bbox["y_upper_bound"],
    } for zone, bbox in DEF_ENTRY_ZONES_BBOX.items()
}


OFF_ENTRY_ZONES_BBOX = {
    zone: {
        "x_lower_bound": 100 - bbox["x_upper_bound"], "x_upper_bound": 100 - bbox["x_lower_bound"],
        "y_lower_bound": bbox["y_lower_bound"], "y_upper_bound": bbox["y_upper_bound"],
    } for zone, bbox in DEF_ENTRY_ZONES_BBOX.items()
}


#MATCH_QUALITY_DICT = get_match_quality_dict()
#EXTRA_QUALITIES = get_extra_quality_dict()

#MATCH_METRICS = get_unique_metrics(match=True)

MATCH_NEGATIVE_METRICS = [
    "Opp. Goals", "Opp. xG","Opp. np xG", "PPDA",
    "Opp. Possessions to final third %",
    "Opp. Final third to box %", "Opp. xT",
    "Turnovers", "Opp. Field tilt %",
    "Time to defensive action (s)",
    "Time to recovery (s)",
    "Opp. Final third entries within 10s after recovery",
    "Opp. Box entries within 10s after recovery",
    "Opp. xG within 10s after recovery",
    "Opp. xT within 10s after recovery",
    "Yellow cards","Red cards","Opp. Penalties","Opp. Corners",
    "Opp. Final third throw-ins",
    "Long ball %", "Opp. Pass tempo"
]

#SEASON_QUALITY_DICT = get_season_quality_dict()

#SEASON_METRICS = get_unique_metrics(season=True)

GPT_TOKEN_LIMIT = 20000

#Get constants from settings
# PAGES_PERMISSIONS = settings.get_pages_permissions()
# SCOUT_PLAYER_TABS_PERMISSIONS = settings.get_scout_player_tabs_permissions()
# SCOUT_CHAT_PERMISSIONS = settings.get_scout_chat_permissions()
# SCOUT_PDF_PERMISSIONS = settings.get_scout_pdf_permissions()
# MATCH_PDF_PERMISSIONS = settings.get_match_pdf_permissions()
# SEASON_PDF_PERMISSIONS = settings.get_season_pdf_permissions()

# SCOUT_PROVIDER = settings.get_scout_provider()
# SCOUT_TRACKING = settings.get_scout_tracking()
# SCOUT_TRACKING_PROVIDER = settings.get_scout_tracking_provider()
# MATCH_PROVIDER = settings.get_match_provider()
# MATCH_TRACKING = settings.get_match_tracking()
# MATCH_TRACKING_PROVIDER = settings.get_match_tracking_provider()

# DARK_MAIN_COLOR = settings.get_dark_main_color()
# MEDIUM_MAIN_COLOR = settings.get_medium_main_color()
# BRIGHT_MAIN_COLOR = settings.get_bright_main_color()

# DESCRIBE_BASE = settings.get_describe_path()
# GPT_EXAMPLES_BASE = settings.get_gpt_example_path()

# MATCH_SUMMARY_QUALITIES = settings.get_match_summary_qualities()
# MATCH_QUALITY_VISUALS = settings.get_match_quality_visuals()

# GENERATE_DATA_FOR_DAYS = settings.get_amount_of_days_for_data()
# SEASON_SUMMARY_QUALITIES = settings.get_season_summary_qualities()
# SEASON_QUALITY_VISUALS = settings.get_season_quality_visuals()

# MATCH_REPORT_PAGES = settings.get_match_report_pages()
# MATCH_REPORT_STANDOUT_PAGE = settings.get_match_report_standout_page()
# MATCH_REPORT_INTRO_PAGE = settings.get_match_report_intro_page()

# FIND_PLAYERS_DEFAULT_ARGS = settings.get_find_player_default_args()