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

    rnd = int(country.select_random())
    # Filter country by position and select a player with sidebar selectors
    with container:

        # Filter for player name
        country.select_and_filter(
            column_name="country",
            label="Country",
            default_index=rnd,  # randomly select a country for default
        )

        # Return data point

        country = country.to_data_point()

    return country


def create_chat(to_hash, chat_class, *args, **kwargs):
    chat_hash_state = hash(to_hash)
    chat = chat_class(chat_hash_state, *args, **kwargs)
    return chat
