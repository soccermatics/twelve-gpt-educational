from io import BytesIO


def split_names(player_names):
    # Iterate over each name in the player_names list
    # and return a modified list of names
    return [
        # If the name consists of only one word
        # or the second-to-last word does not have a length of 2 characters,
        # then the last word is the last name
        (
            name.split()[-1]
            if len(name.split()) == 1 or len(name.split()[-2]) != 2
            # Otherwise, join the second-to-last and last words with a space in between
            # and consider it as the last name
            else " ".join(name.split()[-2:])
        )
        for name in player_names
    ]


def add_per_90(attributes):
    return [
        (
            c + " per 90"
            if "%" not in c
            and "per" not in c
            and "adj" not in c
            and "eff" not in c
            and " - " not in c
            else c
        )
        for c in attributes
    ]


def normalize_text(s, sep_token=" \n "):
    s = " ".join(s.split())
    s = s.replace(". ,", ",")
    s = s.replace(" ,", ",")
    s = s.replace("..", ".")
    s = s.replace(". .", ".")
    s = s.replace("\n", "")
    s = s.strip()
    return s


def insert_newline(s, n_length=15):
    if len(s) <= n_length:
        return s
    else:
        last_space_before_15 = s.rfind(" ", 0, n_length)
        if last_space_before_15 == -1:  # No space found within the first 15 characters
            return s  # Return original string
        else:
            # Split the string at the space and insert a newline
            return s[:last_space_before_15] + "\n" + s[last_space_before_15 + 1 :]


# Function to convert RGBA to HEX
def rgba_to_hex(rgba):
    r, g, b, a = rgba
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def convert_df_to_csv(df, n=1000, ignore=[]):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    # cols = df.columns
    return df.head(n).to_csv(index=None).encode("utf-8")


def get_img_bytes(fig, custom=False, format="png", dpi=200):
    tmpfile = BytesIO()

    if custom:
        fig.savefig(
            tmpfile,
            format=format,
            dpi=dpi,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            pad_inches=0.35,
        )
    else:
        fig.savefig(
            tmpfile,
            format=format,
            dpi=dpi,
            facecolor=fig.get_facecolor(),
            transparent=False,
        )  # , frameon=False)  # , transparent=False, bbox_inches='tight', pad_inches=0.35)

    tmpfile.seek(0)

    return tmpfile


import matplotlib.colors as c


def hex_color_transparency(hex, alpha):
    return c.to_hex(c.to_rgba(hex, alpha), True)


import copy


def select_player(container, players, gender, position):

    # Make a copy of Players object
    player = copy.deepcopy(players)

    # Filter players by position and select a player with sidebar selectors
    with container:

        # Filter for player name
        player.select_and_filter(
            column_name="player_name",
            label="Player",
        )

        # Return data point

        player = player.to_data_point(gender, position)

    return player


def select_country(container, countries):

    # Make a copy of Players object
    country = copy.deepcopy(countries)
    # look up the full country name from the ISO 3166-1 alpha-3 country code 
    country.df["country"] = country.df["country"].map(
     
{
  "AFG": "Afghanistan",
  "ALB": "Albania",
  "DZA": "Algeria",
  "AND": "Andorra",
  "AGO": "Angola",
  "ATG": "Antigua and Barbuda",
  "ARG": "Argentina",
  "ARM": "Armenia",
  "AUS": "Australia",
  "AUT": "Austria",
  "AZE": "Azerbaijan",
  "BHS": "Bahamas",
  "BHR": "Bahrain",
  "BGD": "Bangladesh",
  "BRB": "Barbados",
  "BLR": "Belarus",
  "BEL": "Belgium",
  "BLZ": "Belize",
  "BEN": "Benin",
  "BTN": "Bhutan",
  "BOL": "Bolivia",
  "BIH": "Bosnia and Herzegovina",
  "BWA": "Botswana",
  "BRA": "Brazil",
  "BRN": "Brunei",
  "BGR": "Bulgaria",
  "BFA": "Burkina Faso",
  "BDI": "Burundi",
  "CPV": "Cabo Verde",
  "KHM": "Cambodia",
  "CMR": "Cameroon",
  "CAN": "Canada",
  "CAF": "Central African Republic",
  "TCD": "Chad",
  "CHL": "Chile",
  "CHN": "China",
  "COL": "Colombia",
  "COM": "Comoros",
  "COG": "Congo",
  "CRI": "Costa Rica",
  "HRV": "Croatia",
  "CUB": "Cuba",
  "CYP": "Cyprus",
  "CZE": "Czechia",
  "DNK": "Denmark",
  "DJI": "Djibouti",
  "DMA": "Dominica",
  "DOM": "Dominican Republic",
  "ECU": "Ecuador",
  "EGY": "Egypt",
  "SLV": "El Salvador",
  "GNQ": "Equatorial Guinea",
  "ERI": "Eritrea",
  "EST": "Estonia",
  "SWZ": "Eswatini",
  "ETH": "Ethiopia",
  "FJI": "Fiji",
  "FIN": "Finland",
  "FRA": "France",
  "GAB": "Gabon",
  "GMB": "Gambia",
  "GEO": "Georgia",
  "DEU": "Germany",
  "GHA": "Ghana",
  "GRC": "Greece",
  "GRD": "Grenada",
  "GTM": "Guatemala",
  "GIN": "Guinea",
  "GNB": "Guinea-Bissau",
  "GUY": "Guyana",
  "HTI": "Haiti",
  "HND": "Honduras",
  "HUN": "Hungary",
  "ISL": "Iceland",
  "IND": "India",
  "IDN": "Indonesia",
  "IRN": "Iran",
  "IRQ": "Iraq",
  "IRL": "Ireland",
  "ISR": "Israel",
  "ITA": "Italy",
  "JAM": "Jamaica",
  "JPN": "Japan",
  "JOR": "Jordan",
  "KAZ": "Kazakhstan",
  "KEN": "Kenya",
  "KIR": "Kiribati",
  "PRK": "Korea, North",
  "KOR": "Korea, South",
  "KWT": "Kuwait",
  "KGZ": "Kyrgyzstan",
  "LAO": "Laos",
  "LVA": "Latvia",
  "LBN": "Lebanon",
  "LSO": "Lesotho",
  "LBR": "Liberia",
  "LBY": "Libya",
  "LIE": "Liechtenstein",
  "LTU": "Lithuania",
  "LUX": "Luxembourg",
  "MDG": "Madagascar",
  "MWI": "Malawi",
  "MYS": "Malaysia",
  "MDV": "Maldives",
  "MLI": "Mali",
  "MLT": "Malta",
  "MHL": "Marshall Islands",
  "MRT": "Mauritania",
  "MUS": "Mauritius",
  "MEX": "Mexico",
  "FSM": "Micronesia",
  "MDA": "Moldova",
  "MCO": "Monaco",
  "MNG": "Mongolia",
  "MNE": "Montenegro",
  "MAR": "Morocco",
  "MOZ": "Mozambique",
  "MMR": "Myanmar",
  "NAM": "Namibia",
  "NRU": "Nauru",
  "NPL": "Nepal",
  "NLD": "Netherlands",
  "NZL": "New Zealand",
  "NIC": "Nicaragua",
  "NER": "Niger",
  "NGA": "Nigeria",
  "MKD": "North Macedonia",
  "NOR": "Norway",
  "OMN": "Oman",
  "PAK": "Pakistan",
  "PLW": "Palau",
  "PAN": "Panama",
  "PNG": "Papua New Guinea",
  "PRY": "Paraguay",
  "PER": "Peru",
  "PHL": "Philippines",
  "POL": "Poland",
  "PRT": "Portugal",
  "QAT": "Qatar",
  "ROU": "Romania",
  "RUS": "Russia",
  "RWA": "Rwanda",
  "KNA": "Saint Kitts and Nevis",
  "LCA": "Saint Lucia",
  "VCT": "Saint Vincent and the Grenadines",
  "WSM": "Samoa",
  "SMR": "San Marino",
  "STP": "Sao Tome and Principe",
  "SAU": "Saudi Arabia",
  "SEN": "Senegal",
  "SRB": "Serbia",
  "SYC": "Seychelles",
  "SLE": "Sierra Leone",
  "SGP": "Singapore",
  "SVK": "Slovakia",
  "SVN": "Slovenia",
  "SLB": "Solomon Islands",
  "SOM": "Somalia",
  "ZAF": "South Africa",
  "SSD": "South Sudan",
  "ESP": "Spain",
  "LKA": "Sri Lanka",
  "SDN": "Sudan",
  "SUR": "Suriname",
  "SWE": "Sweden",
  "CHE": "Switzerland",
  "SYR": "Syria",
  "TWN": "Taiwan",
  "TJK": "Tajikistan",
  "TZA": "Tanzania",
  "THA": "Thailand",
  "TLS": "Timor-Leste",
  "TGO": "Togo",
  "TON": "Tonga",
  "TTO": "Trinidad and Tobago",
  "TUN": "Tunisia",
  "TUR": "Turkey",
  "TKM": "Turkmenistan",
  "TUV": "Tuvalu",
  "UGA": "Uganda",
  "UKR": "Ukraine",
  "ARE": "United Arab Emirates",
  "GBR": "United Kingdom",
  "USA": "United States",
  "URY": "Uruguay",
  "UZB": "Uzbekistan",
  "VUT": "Vanuatu",
  "VEN": "Venezuela",
  "VNM": "Vietnam",
  "YEM": "Yemen",
  "ZMB": "Zambia",
  "ZWE": "Zimbabwe"
}
    )




    


    # Filter players by position and select a player with sidebar selectors
    with container:

        # Filter for player name
        country.select_and_filter(
            column_name="country",
            label="Country",
        )

        # Return data point

        country = country.to_data_point()

    return country


def create_chat(to_hash, chat_class, *args, **kwargs):
    chat_hash_state = hash(to_hash)
    chat = chat_class(chat_hash_state, *args, **kwargs)
    return chat
