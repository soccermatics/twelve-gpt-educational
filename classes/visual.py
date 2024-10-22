import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


from utils.sentences import format_metric

from classes.data_point import Player, Person, Country
from classes.data_source import PlayerStats, PersonStat, CountryStats
from typing import Union


def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


def rgb_to_color(rgb_color: tuple, opacity=1):
    return f"rgba{(*rgb_color, opacity)}"


def tick_text_color(color, text, alpha=1.0):
    # color: hexadecimal
    # alpha: transparency value between 0 and 1 (default is 1.0, fully opaque)
    s = (
        "<span style='color:rgba("
        + str(int(color[1:3], 16))
        + ","
        + str(int(color[3:5], 16))
        + ","
        + str(int(color[5:], 16))
        + ","
        + str(alpha)
        + ")'>"
        + str(text)
        + "</span>"
    )
    return s




class Visual:
    # Can't use streamlit options due to report generation
    dark_green = hex_to_rgb(
        "#002c1c"
    )  # hex_to_rgb(st.get_option("theme.secondaryBackgroundColor"))
    medium_green = hex_to_rgb("#003821")
    bright_green = hex_to_rgb(
        "#00A938"
    )  # hex_to_rgb(st.get_option("theme.primaryColor"))
    bright_orange = hex_to_rgb("#ff4b00")
    bright_yellow = hex_to_rgb("#ffcc00")
    bright_blue = hex_to_rgb("#0095FF")
    white = hex_to_rgb("#ffffff")  # hex_to_rgb(st.get_option("theme.backgroundColor"))
    gray = hex_to_rgb("#808080")
    black = hex_to_rgb("#000000")
    light_gray = hex_to_rgb("#d3d3d3")
    table_green = hex_to_rgb('#009940')
    table_red = hex_to_rgb('#FF4B00')

    def __init__(self):
        
        self.font_size_multiplier = 1.0
        self.fig = go.Figure()
        self._setup_styles()

    def show(self):
        st.plotly_chart(self.fig, config={"displayModeBar": False}, height=500, use_container_width=True)


    def _setup_styles(self):
        side_margin = 60
        top_margin = 75
        pad = 16
        self.fig.update_layout(
            autosize=True,
            height=500,
            margin=dict(l=side_margin, r=side_margin, b=70, t=top_margin, pad=pad),
            paper_bgcolor=rgb_to_color(self.dark_green),
            plot_bgcolor=rgb_to_color(self.dark_green),
            legend=dict(
                orientation="h",
                font={
                    "color": rgb_to_color(self.white),
                    "family": "Gilroy-Light",
                    "size": 11 * self.font_size_multiplier,
                },
                itemclick=False,
                itemdoubleclick=False,
                x=0.5,
                xanchor="center",
                y=-0.2,
                yanchor="bottom",
                valign="middle",  # Align the text to the middle of the legend
            ),
            xaxis=dict(
                tickfont={
                    "color": rgb_to_color(self.white, 0.5),
                    "family": "Gilroy-Light",
                    "size": 12 * self.font_size_multiplier,
                },
            ),
        )

    def add_title(self, title, subtitle):
        self.title = title
        self.subtitle = subtitle
        self.fig.update_layout(
            title={
                "text": f"<span style='font-size: {15*self.font_size_multiplier}px'>{title}</span><br>{subtitle}",
                "font": {
                    "family": "Gilroy-Medium",
                    "color": rgb_to_color(self.white),
                    "size": 12 * self.font_size_multiplier,
                },
                "x": 0.05,
                "xanchor": "left",
                "y": 0.93,
                "yanchor": "top",
            },
        )

    def add_low_center_annotation(self, text):
        self.fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.07,
            text=text,
            showarrow=False,
            font={
                "color": rgb_to_color(self.white, 0.5),
                "family": "Gilroy-Light",
                "size": 12 * self.font_size_multiplier,
            },
        )
    def show(self):
        st.plotly_chart(self.fig, config={"displayModeBar": False}, height=500, use_container_width=True)
    
    def close(self):
        pass

class DistributionPlot(Visual):
    def __init__(self, columns, labels=None, *args, **kwargs):
        self.empty = True
        self.columns = columns
        self.marker_color = (
            c for c in [Visual.white, Visual.bright_yellow, Visual.bright_blue]
        )
        self.marker_shape = (s for s in ["square", "hexagon", "diamond"])
        super().__init__(*args, **kwargs)
        if labels is not None:
            self._setup_axes(labels)
        else:
            self._setup_axes()

    def _setup_axes(self, labels=["Worse", "Average", "Better"]):
        self.fig.update_xaxes(
            range=[-4, 4],
            fixedrange=True,
            tickmode="array",
            tickvals=[-3, 0, 3],
            ticktext=labels,
        )
        self.fig.update_yaxes(
            showticklabels=False,
            fixedrange=True,
            gridcolor=rgb_to_color(self.medium_green),
            zerolinecolor=rgb_to_color(self.medium_green),
        )

    def add_group_data(self, df_plot, plots, names, legend, hover="", hover_string=""):
        showlegend = True

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string

            metric_name = format_metric(col)

            temp_df = pd.DataFrame(df_plot[col + hover])
            temp_df["name"] = metric_name

            self.fig.add_trace(go.Scatter(
                    x=df_plot[col + plots],
                    y=np.ones(len(df_plot)) * i,
                    mode="markers",
                    marker={
                        "color": rgb_to_color(self.bright_green, opacity=0.2),
                        "size": 10,
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=names,
                    customdata=df_plot[col + hover],
                    name=legend,
                    showlegend=showlegend,
                )
            )
            showlegend = False

    def add_data_point(
        self, ser_plot, plots, name, hover="", hover_string="", text=None
    ):
        if text is None:
            text = [name]
        elif isinstance(text, str):
            text = [text]
        legend = True
        color = next(self.marker_color)
        marker = next(self.marker_shape)

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string

            metric_name = format_metric(col)

            self.fig.add_trace(
                go.Scatter(
                    x=[ser_plot[col + plots]],
                    y=[i],
                    mode="markers",
                    marker={
                        "color": rgb_to_color(color, opacity=0.5),
                        "size": 10,
                        "symbol": marker,
                        "line_width": 1.5,
                        "line_color": rgb_to_color(color),
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=text,
                    customdata=[ser_plot[col + hover]],
                    name=name,
                    showlegend=legend,
                )
            )
            legend = False

            self.fig.add_annotation(
                x=0,
                y=i + 0.4,
                text=self.annotation_text.format(
                    metric_name=metric_name, data=ser_plot[col]
                ),
                showarrow=False,
                font={
                    "color": rgb_to_color(self.white),
                    "family": "Gilroy-Light",
                    "size": 12 * self.font_size_multiplier,
                },
            )

    # def add_player(self, player: Player, n_group,metrics):

    #     # Make list of all metrics with _Z and _Rank added at end
    #     metrics_Z = [metric + "_Z" for metric in metrics]
    #     metrics_Ranks = [metric + "_Ranks" for metric in metrics]

    #     self.add_data_point(
    #         ser_plot=player.ser_metrics,
    #         plots = '_Z',
    #         name=player.name,
    #         hover='_Ranks',
    #         hover_string="Rank: %{customdata}/" + str(n_group)
    #     )

    def add_player(self, player: Union[Player, Country], n_group, metrics):

        # # Make list of all metrics with _Z and _Rank added at end
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]

        # Determine the appropriate attributes for player or country
        if isinstance(player, Player):
            ser_plot = player.ser_metrics
            name = player.name
        elif isinstance(player, Country):  # Adjust this based on your class structure
            ser_plot = (
                player.ser_metrics
            )  # Assuming countries have a similar metric structure
            name = player.name
        else:
            raise TypeError("Invalid player type: expected Player or Country")

        self.add_data_point(
            ser_plot=ser_plot,
            plots="_Z",
            name=name,
            hover="_Ranks",
            hover_string="Rank: %{customdata}/" + str(n_group),
        )

    # def add_players(self, players: PlayerStats, metrics):

    #     # Make list of all metrics with _Z and _Rank added at end
    #     metrics_Z = [metric + "_Z" for metric in metrics]
    #     metrics_Ranks = [metric + "_Ranks" for metric in metrics]

    #     self.add_group_data(
    #         df_plot=players.df,
    #         plots="_Z",
    #         names=players.df["player_name"],
    #         hover="_Ranks",
    #         hover_string="Rank: %{customdata}/" + str(len(players.df)),
    #         legend=f"Other players  ",  # space at end is important
    #     )

    def add_players(self, players: Union[PlayerStats, CountryStats], metrics):

        # Make list of all metrics with _Z and _Rank added at end
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]

        if isinstance(players, PlayerStats):
            self.add_group_data(
                df_plot=players.df,
                plots="_Z",
                names=players.df["player_name"],
                hover="_Ranks",
                hover_string="Rank: %{customdata}/" + str(len(players.df)),
                legend=f"Other players  ",  # space at end is important
            )
        elif isinstance(players, CountryStats):
            self.add_group_data(
                df_plot=players.df,
                plots="_Z",
                names=players.df["country"],
                hover="_Ranks",
                hover_string="Rank: %{customdata}/" + str(len(players.df)),
                legend=f"Other countries  ",  # space at end is important
            )
        else:
            raise TypeError("Invalid player type: expected Player or Country")

    # def add_title_from_player(self, player: Player):
    #     self.player = player

    #     title = f"Evaluation of {player.name}?"
    #     subtitle = f"Based on {player.minutes_played} minutes played"

    #     self.add_title(title, subtitle)

    def add_title_from_player(self, player: Union[Player, Country]):
        self.player = player

        title = f"Evaluation of {player.name}?"
        if isinstance(player, Player):
            subtitle = f"Based on {player.minutes_played} minutes played"
        elif isinstance(player, Country):
            subtitle = f"Based on questions answered in the World Values Survey"
        else:
            raise TypeError("Invalid player type: expected Player or Country")

        self.add_title(title, subtitle)
#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------



class DistributionPlotPersonality(Visual):
    def __init__(self, columns, *args, **kwargs):
        self.empty = True
        self.columns = columns
        self.marker_color = (c for c in [Visual.white, Visual.bright_yellow, Visual.bright_blue])
        self.marker_shape = (s for s in ["square", "hexagon", "diamond"])
        super().__init__(*args, **kwargs)
        self._setup_axes()

    def _setup_axes(self):
        self.fig.update_xaxes(range=[-4, 4], fixedrange=True, tickmode="array", tickvals=[-3, 0, 3], ticktext=["Worse", "Average", "Better"])
        self.fig.update_yaxes(showticklabels=False, fixedrange=True, gridcolor=rgb_to_color(self.medium_green), zerolinecolor=rgb_to_color(self.medium_green))

    def add_group_data(self, df_plot, plots, names, legend, hover='', hover_string=""):
        showlegend = True

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string

            metric_name = format_metric(col)

            temp_df = pd.DataFrame(df_plot[col+hover])
            temp_df['name'] = metric_name
            
            self.fig.add_trace(go.Scatter(
                    x=df_plot[col+plots], y=np.ones(len(df_plot))*i,
                    mode="markers",
                    marker={
                        "color": rgb_to_color(self.bright_green, opacity=0.2), "size": 10,
                    },
                    hovertemplate='%{text}<br>'+temp_hover_string+'<extra></extra>',
                    text=names,
                    customdata=round(df_plot[col+hover]),
                    name=legend,
                    showlegend=showlegend,
                )
            )
            showlegend = False

    def add_data_point(self, ser_plot, plots, name, hover='', hover_string="", text=None):
        if text is None:
            text = [name]
        elif isinstance(text, str):
            text = [text]
        legend = True
        color = next(self.marker_color)
        marker = next(self.marker_shape)

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string
         

            metric_name = format_metric(col)

            
            self.fig.add_trace(
                go.Scatter(
                    x=[ser_plot[col+plots]], y=[i], mode="markers",
                    marker={"color": rgb_to_color(color, opacity=0.5), "size": 10, "symbol": marker, "line_width": 1.5, "line_color": rgb_to_color(color)},
                    hovertemplate='%{text}<br>'+temp_hover_string+'<extra></extra>',
                    text=text,
                    customdata=[round(ser_plot[col+hover])],
                    name=name,
                    showlegend=legend
                )
            )
            legend = False

            self.fig.add_annotation(
                x=0, y=i + 0.4, text=f"<span style=''>{metric_name}: {int(ser_plot[col]):.0f}</span>", showarrow=False,
                font={"color": rgb_to_color(self.white), "family": "Gilroy-Light",
                        "size": 12 * self.font_size_multiplier},
            )

    def add_person(self, person: Person, n_group, metrics):
        # Make list of all metrics with _Z and _Rank added at end 
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]
        
        self.add_data_point(
            ser_plot = person.ser_metrics, 
            plots = '_Z',
            name=person.name, 
            hover='_Ranks',
            hover_string="Rank: %{customdata}/" + str(n_group)
        )
    
    def add_persons(self, persons: PersonStat, metrics):

        # Make list of all metrics with _Z and _Rank added at end 
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]
        
        self.add_group_data(
                df_plot= persons.df,
                plots = '_Z',
                names=persons.df["name"],
                hover='_Ranks',
                hover_string="Rank: %{customdata}/" + str(len(persons.df)),
                legend=f"Other persons  ",
            )
    
    def add_title_from_person(self, person: Person):            
        self.person = person
        title = f"Evaluation of {person.name}"
        subtitle = f"Based on Big Five scores"
        self.add_title(title, subtitle)



'''class ViolinPlot(Visual):
    def violin(data, point_data):
        # Create a figure object
        fig = go.Figure()

        # Labels for the columnshover
        labels = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']

        # Loop through each label to add a violin plot trace
        for label in labels:
            fig.add_trace(go.Violin(
                x=df_plot[label],  # Use x for the data
                name=label,      # Label each violin plot correctly
                box_visible=True,
                meanline_visible=True,
                line_color='black',  # Color of the violin outline
                fillcolor='rgba(0,100,200,0.3)',  # Color of the violin fill
                opacity=0.6,
                orientation='h'  # Set orientation to horizontal
            )
        )
        for label, value in point_data.items():
            fig.add_trace(
                go.Scatter(x=[value], y=[label], mode='markers', marker=dict(color='red', size=8, symbol='cross'), name=f'{label} Candidate Point'))

        # Update layout for better visualization
        fig.update_layout(
            title='Distribution of Personality Traits',
            xaxis_title='Score',  
            yaxis_title='Trait',
            xaxis=dict(range=[0, 40]),
            violinmode='overlay', 
            showlegend=True)

        # Display the plot in Streamlit
        st.plotly_chart(fig)


    def radarPlot(Visual):
        # Data import
        data_r = data_p.to_list()  
        labels = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
        df = pd.DataFrame({'data': data_r,'label': labels})
    
        # Create the radar plot
        fig = px.line_polar(df, r='data', theta='label', line_close=True, markers=True)
        fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0, 40])),showlegend=True, title= 'Candidate profile')
        fig.update_traces(fill='toself', marker=dict(size=5))
        # Display the plot in Streamlit
        st.plotly_chart(fig)'''

