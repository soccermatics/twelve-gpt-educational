import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


from utils.sentences import format_metric

from classes.data_point import Player
from classes.data_source import PlayerStats
import utils.constants as const


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
    s = "<span style='color:rgba(" + str(int(color[1:3], 16)) + "," + \
        str(int(color[3:5], 16)) + "," + \
        str(int(color[5:], 16)) + "," + str(alpha) + ")'>" + str(text) + "</span>"
    return s
class Visual():
    # Can't use streamlit options due to report generation
    dark_green = hex_to_rgb("#002c1c") # hex_to_rgb(st.get_option("theme.secondaryBackgroundColor"))
    medium_green = hex_to_rgb("#003821")
    bright_green = hex_to_rgb("#00A938") # hex_to_rgb(st.get_option("theme.primaryColor"))
    bright_orange = hex_to_rgb("#ff4b00")
    bright_yellow = hex_to_rgb("#ffcc00")
    bright_blue = hex_to_rgb("#0095FF")
    white = hex_to_rgb("#ffffff") # hex_to_rgb(st.get_option("theme.backgroundColor"))
    gray = hex_to_rgb("#808080")
    black = hex_to_rgb("#000000")
    light_gray = hex_to_rgb("#d3d3d3")
    table_green = hex_to_rgb('#009940')
    table_red = hex_to_rgb('#FF4B00')
    def __init__(self, pdf=False):
        self.pdf = pdf
        if pdf:
            self.font_size_multiplier = 1.4
        else:
            self.font_size_multiplier = 1.
        self.fig = go.Figure()
        self._setup_styles()

    def show(self):
        st.plotly_chart(self.fig, config={"displayModeBar": False}, height=500, use_container_width=True)

    def close(self):
        pass

    def _setup_styles(self):
        side_margin = 60
        top_margin =  75
        pad = 16
        self.fig.update_layout(
            autosize=True,
            height=500,
            margin=dict(
                l=side_margin, r=side_margin, b=70, t=top_margin, pad=pad
            ),
            paper_bgcolor=rgb_to_color(self.dark_green),
            plot_bgcolor=rgb_to_color(self.dark_green),
            legend=dict(
                orientation="h",
                font={"color": rgb_to_color(self.white), "family": "Gilroy-Light", "size":11*self.font_size_multiplier},
                itemclick=False,
                itemdoubleclick=False,
                x=0.5, xanchor="center", y=-0.2, yanchor="bottom",
                valign="middle", #Align the text to the middle of the legend
            ),
            xaxis=dict(
                tickfont={"color": rgb_to_color(self.white,0.5), "family": "Gilroy-Light", "size": 12*self.font_size_multiplier},
            )
        )

    def add_title(self, title, subtitle):
        self.title = title
        self.subtitle = subtitle
        self.fig.update_layout(
            title={
                "text": f"<span style='font-size: {15*self.font_size_multiplier}px'>{title}</span><br>{subtitle}",
                "font": {"family": "Gilroy-Medium", "color": rgb_to_color(self.white), "size": 12*self.font_size_multiplier},
                "x": 0.05, "xanchor": "left", "y": 0.93, "yanchor": "top"
            },
        )
    
    def add_low_center_annotation(self, text):
        self.fig.add_annotation(
            xref = 'paper', yref='paper',
            x=0.5, y=-0.07, text=text, showarrow=False,
            font={"color": rgb_to_color(self.white,0.5), "family": "Gilroy-Light", "size": 12*self.font_size_multiplier},
        )

    

class DistributionPlot(Visual):
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
            
            self.fig.add_trace(
                go.Scatter(
                    x=df_plot[col+plots], y=np.ones(len(df_plot))*i,
                    mode="markers",
                    marker={
                        "color": rgb_to_color(self.bright_green, opacity=0.2), "size": 10,
                    },
                    hovertemplate='%{text}<br>'+temp_hover_string+'<extra></extra>',
                    text=names,
                    customdata=df_plot[col+hover],
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
                    customdata=[ser_plot[col+hover]],
                    name=name,
                    showlegend=legend
                )
            )
            legend = False

            self.fig.add_annotation(
                x=0, y=i + 0.4, text=f"<span style=''>{metric_name}: {ser_plot[col]:.2f} per 90</span>", showarrow=False,
                font={"color": rgb_to_color(self.white), "family": "Gilroy-Light",
                        "size": 12 * self.font_size_multiplier},
            )


    def add_player(self, player: Player, n_group,metrics):
        
        # Make list of all metrics with _Z and _Rank added at end 
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]
        
        self.add_data_point(
            ser_plot=player.ser_metrics,
            plots = '_Z',
            name=player.name,
            hover='_Ranks',
            hover_string="Rank: %{customdata}/" + str(n_group)
        )

    def add_players(self, players: PlayerStats, metrics):

        # Make list of all metrics with _Z and _Rank added at end 
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]
        
        self.add_group_data(
            df_plot=players.df,
            plots = '_Z',
            names=players.df["player_name"],
            hover='_Ranks',
            hover_string="Rank: %{customdata}/" + str(len(players.df)),
            legend=f"Other players  ", #space at end is important
        )

    def add_title_from_player(self, player: Player):            
        self.player = player
  
        title = f"Evaluation of {player.name}?"
        subtitle = f"Based on {player.minutes_played} minutes played"

        self.add_title(title, subtitle)







class PitchVisual(Visual):
    def __init__(self, metric, pdf = False, *args, **kwargs):
        self.metric = metric
        super().__init__(*args, **kwargs)
        self._add_pitch()
        self.pdf = pdf

    @staticmethod
    def ellipse_arc(x_center=0, y_center=0, a=1, b=1, start_angle=0, end_angle=2 * np.pi, N=100):
        t = np.linspace(start_angle, end_angle, N)
        x = x_center + a * np.cos(t)
        y = y_center + b * np.sin(t)
        path = f'M {x[0]}, {y[0]}'
        for k in range(1, len(t)):
            path += f'L{x[k]}, {y[k]}'
        return path

    def _add_pitch(self):
        self.fig.update_layout(
            hoverdistance=100,
            xaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=True,
                zeroline=False,
                # Range slightly larger to avoid half of line being hidden by edge of plot
                range=[-0.2, 105],
                scaleanchor="y",
                scaleratio=1.544,
                constrain="domain",
                tickvals=[25, 75],
                ticktext=["Defensive", "Offensive"],
                fixedrange=True,
            ),
            yaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=False,
                zeroline=False,
                # Range slightly larger to avoid half of line being hidden by edge of plot
                range=[-0.2, 68],
                constrain="domain",
                fixedrange=True,
            ),
        )
        shapes = self._get_shapes()
        for shape in shapes:
            shape.update(dict(line={"color": "white", "width": 2}, xref="x", yref="y", ))
            self.fig.add_shape(**shape)

    def _get_shapes(self):
        # Plotly doesn't support arcs svg paths, so we need to create them manually
        shapes = [
            # Center circle
            dict(
                type="circle", x0=41.28, y0=36.54, x1=58.71, y1=63.46,
            ),
            # Own penalty area
            dict(
                type="rect", x0=0, y0=19, x1=16, y1=81,
            ),
            # Opponent penalty area
            dict(
                type="rect", x0=84, y0=19, x1=100, y1=81,
            ),
            dict(
                type="rect", x0=0, y0=0, x1=100, y1=100,
            ),
            # Halfway line
            dict(
                type="line", x0=50, y0=0, x1=50, y1=100,
            ),
            # Own goal area
            dict(
                type="rect", x0=0, y0=38, x1=6, y1=62,
            ),
            # Opponent goal area
            dict(
                type="rect", x0=94, y0=38, x1=100, y1=62,
            ),
            # Own penalty spot
            dict(
                type="circle", x0=11.2, y0=49.5, x1=11.8, y1=50.5, fillcolor="white"
            ),
            # Opponent penalty spot
            dict(
                type="circle", x0=89.2, y0=49.5, x1=89.8, y1=50.5, fillcolor="white"
            ),
            # Penalty arc
            # Not sure why we need to multiply the radii by 1.35, but it seems to work
            dict(
                type="path", path=self.ellipse_arc(11, 50, 6.2 * 1.35, 9.5 * 1.35, -0.3 * np.pi, 0.3 * np.pi),
            ),
            dict(
                type="path", path=self.ellipse_arc(89, 50, 6.2 * 1.35, 9.5 * 1.35, 0.7 * np.pi, 1.3 * np.pi),
            ),
            # Corner arcs
            # Can't plot a part of a cirlce
            # dict(
            #     type="circle", x0=-6.2*0.3, y0=-9.5*0.3, x1=6.2*0.3, y1=9.5*0.3,
            # ),
            dict(
                type="path", path=self.ellipse_arc(0, 0, 6.2 * 0.3, 9.5 * 0.3, 0, np.pi / 2),
            ),
            dict(
                type="path", path=self.ellipse_arc(100, 0, 6.2 * 0.3, 9.5 * 0.3, np.pi / 2, np.pi),
            ),
            dict(
                type="path", path=self.ellipse_arc(100, 100, 6.2 * 0.3, 9.5 * 0.3, np.pi, 3 / 2 * np.pi),
            ),
            dict(
                type="path", path=self.ellipse_arc(0, 100, 6.2 * 0.3, 9.5 * 0.3, 3 / 2 * np.pi, 2 * np.pi),
            ),
            # Goals
            dict(
                type="rect", x0=-3, y0=44, x1=0, y1=56,
            ),
            dict(
                type="rect", x0=100, y0=44, x1=103, y1=56,
            ),
        ]
        return shapes

    def add_group_data(self, *args, **kwargs):
        pass

    def iter_zones(self, zone_dict=const.PITCH_ZONES_BBOX):
        for key, value in zone_dict.items():
            x = [
                value["x_lower_bound"],
                value["x_upper_bound"],
                value["x_upper_bound"],
                value["x_lower_bound"],
                value["x_lower_bound"],
            ]
            y = [
                value["y_lower_bound"],
                value["y_lower_bound"],
                value["y_upper_bound"],
                value["y_upper_bound"],
                value["y_lower_bound"],
            ]
            yield key, x, y

    def add_data_point(self, ser_plot, name, ser_hover, hover_string):
        for key, x, y in self.iter_zones():
            if key in ser_plot.index and ser_plot[key] == ser_plot[key]:
                self.fig.add_trace(
                    go.Scatter(
                        x=x, y=y,
                        mode="lines",
                        line={"color": rgb_to_color(self.bright_green), "width": 1, },
                        fill="toself",
                        fillcolor=rgb_to_color(self.bright_green, opacity=float(ser_plot[key] / 100), ),
                        showlegend=False,
                        name=name,
                        hoverinfo="skip"
                    )
                )
                self.fig.add_trace(
                    go.Scatter(
                        x=[x[0] / 2 + x[1] / 2],
                        y=[y[0] / 2 + y[2] / 2],
                        mode="text",
                        hovertemplate=name + '<br>Zone: ' + key.capitalize() + "<br>" + hover_string + '<extra></extra>',
                        text=describe_level(ser_hover[key][0]).capitalize(),
                        textposition="middle center",
                        textfont={"color": rgb_to_color(self.white), "family": "Gilroy-Light",
                                  "size": 10 * self.font_size_multiplier},
                        customdata=[ser_hover[key]],
                        showlegend=False,
                    )
                )
            else:
                self.fig.add_trace(
                    go.Scatter(
                        x=x, y=y,
                        mode="lines",
                        line={"width": 0, },
                        fill="toself",
                        fillcolor=rgb_to_color(self.gray, opacity=0.5),
                        showlegend=False,
                        hoverinfo="skip"
                    )
                )
                self.fig.add_trace(
                    go.Scatter(
                        x=[x[0] / 2 + x[1] / 2],
                        y=[y[0] / 2 + y[2] / 2],
                        mode="none",
                        hovertemplate='Not enough data<extra></extra>',
                        text=[name],
                        showlegend=False,
                    )
                )

    def add_player(self, player, n_group, quality):
        metric = const.QUALITY_PITCH_METRICS[quality]
        multi_level = {}
        for key, _, _ in self.iter_zones():
            ser = player.ser_zones["Z"][metric]
            if key in ser.index and ser[key] == ser[key]:
                multi_level[key] = np.stack([
                    player.ser_zones["Z"][metric][key],
                    player.ser_zones["Raw"][metric][key]
                ], axis=-1)

        self.add_data_point(
            ser_plot=player.ser_zones["Rank_pct"][metric],
            ser_hover=multi_level,
            name=player.name,
            hover_string="Z-Score: %{customdata[0]:.2f}<br>%{customdata[1]:.2f} " + self.metric.lower(),
        )

    def add_title_from_player(self, player: Player, other_player: Player = None, quality=None):
        short_metric_name = self.metric.replace(" per 90", "").replace(" %", "")
        short_metric_name = short_metric_name[0].lower() + short_metric_name[1:]
        title = f"How is {player.name} at {'<i>' if not self.pdf else ''}{short_metric_name}{'</i>' if not self.pdf else ''}?"
        subtitle = f"{player.competition.get_plot_subtitle()} | {player.minutes} minutes played"
        self.add_title(title, subtitle)
        self.add_low_center_annotation(f"Compared to other {player.detailed_position.lower()}s")

    # def add_title_from_match(self, match: Match, quality):
    #     if quality is None:
    #         title = const.MATCH_QUALITY_VISUALS["Summary"]["pitch_title"]
    #     else:
    #         title = const.MATCH_QUALITY_VISUALS[quality]["pitch_title"]
        
    #     title = title.replace("TEAM_NAME", match.team_name).replace("OPPONENT_NAME", match.opp_team_name)
    #     subtitle = f"{match.label} | {match.date.split(' ')[0]}"
    #     self.add_title(title, subtitle)

    # def add_extra_info(self, match: Match):
    #     # Function for adding metrics values above plot. Only used in club specific settings test.py.
    #     val = match.ser_metrics['Raw']['Possessions to final third %']
    #     val2 = match.ser_metrics['Raw']['Final third to box %']
    #     val3 = match.ser_metrics['Raw']['Box entries to shot %']

    #     self.fig.add_annotation(
    #         text=f"Possessions to final third: {val * 100:.0f} %<br>Final third to box: {val2 * 100:.0f} %<br>Box entries to shot: {val3 * 100:.0f} %",
    #         xref="paper", yref="paper",
    #         align="left",
    #         x=0, y=0.99, showarrow=False,
    #         font={"family": "Gilroy-Medium", "size": 12 * self.font_size_multiplier,
    #               "color": rgb_to_color(self.white)})

    # def add_entries(self, match: Match, opp):
    #     df_plot = match.df_entries_against if opp else match.df_entries
    #     df_plot['category'] = np.where(df_plot['type'] == 'pass', 'Pass', 'Carry')
        
    #     nans = np.empty((len(df_plot)))
    #     nans[:] = np.nan
    #     x = np.array([df_plot.start_x, df_plot.end_x, nans]).T.flatten()
    #     y = np.array([df_plot.start_y, df_plot.end_y, nans]).T.flatten()
    #     # Add 0 before seconds under 10
    #     customdata = np.stack(
    #         [df_plot.name, df_plot.minute,
    #          np.where(df_plot.second < 10, '0' + df_plot.second.astype(str), df_plot.second.astype(str)), df_plot.xG,
    #          df_plot.category], axis=-1
    #     )

    #     customdata = np.repeat(customdata, 3, axis=0)
    #     masks = [
    #         np.repeat(~df_plot.carry & ~df_plot.goal, 3),
    #         np.repeat(df_plot["carry"], 3),
    #         np.repeat(df_plot.goal & ~df_plot.carry, 3),
    #         np.repeat(df_plot.goal & df_plot.carry, 3)
    #     ]
    #     markers = [
    #                   {"color": rgb_to_color(self.white, opacity=0.5), "size": 5, "symbol": "arrow-up",
    #                    "angleref": "previous"}
    #               ] * 2 + [
    #                   {"color": rgb_to_color(self.bright_green, opacity=1), "size": 5, "symbol": "arrow-up",
    #                    "angleref": "previous"}
    #               ] * 2
    #     lines = [
    #         {"color": rgb_to_color(self.white, opacity=0.5), "width": 1},
    #         {"color": rgb_to_color(self.white, opacity=0.5), "width": 1, "dash": "dot"},
    #         {"color": rgb_to_color(self.bright_green), "width": 2},
    #         {"color": rgb_to_color(self.bright_green), "width": 2, "dash": "dot"}
    #     ]
    #     names = ["Pass", "Carry", "Goal", 'goal_carry']
    #     for mask, marker, line, name in zip(masks, markers, lines, names):
    #         self.fig.add_trace(
    #             go.Scatter(
    #                 x=x[mask], y=y[mask],
    #                 mode="lines+markers",
    #                 marker=marker,
    #                 line=line,
    #                 showlegend=False,
    #                 name=name,
    #                 customdata=customdata[mask],
    #                 hovertemplate='<b>%{customdata[4]}</b><br>%{customdata[0]}<br> %{customdata[1]}:%{customdata[2]}<br>%{customdata[3]:.2f} xG created<extra></extra>',
    #             )
    #         )
    #         if name != "goal_carry":
    #             # Customized legend
    #             self.fig.add_trace(
    #                 go.Scatter(
    #                     x=np.array([-10, -10]).T.flatten(),
    #                     y=np.array([-10, -10]).T.flatten(),
    #                     mode="lines+markers",
    #                     marker=marker,
    #                     line=line,
    #                     name=name,
    #                     hoverinfo="skip"
    #                 )
    #             )

    #     # Customized legend
    #     self.fig.update_layout(
    #         legend=dict(
    #             yanchor="bottom",
    #             entrywidth=50,
    #             y=-0.05,
    #         )
    #     )

    #     # Manually added in legend
    #     if self.pdf:
    #         x_high = 0.72
    #         x_low = 0.35
    #     else:
    #         x_high = 0.735
    #         x_low = 0.34

    #     self.fig.add_annotation(text="Higher xG",
    #                             xref="paper", yref="paper",
    #                             x=x_high, y=-0.1, showarrow=False,
    #                             font={"family": "Gilroy-Medium", "size": 12 * self.font_size_multiplier,
    #                                   "color": rgb_to_color(self.white)})

    #     self.fig.add_annotation(text="Lower xG",
    #                             xref="paper", yref="paper",
    #                             x=x_low, y=-0.1, showarrow=False,
    #                             font={"family": "Gilroy-Medium", "size": 12 * self.font_size_multiplier,
    #                                   "color": rgb_to_color(self.white)})

    #     self.fig.add_annotation(text="In comparison to other matches",
    #                             xref="paper", yref="paper",
    #                             x=0.5, y=-0.15, showarrow=False,
    #                             font={"family": "Gilroy-Medium", "size": 12 * self.font_size_multiplier,
    #                                   "color": '#80958D'})

    #     squares_parameters = [
    #         {"x0": 0.535, "y0": -0.09, "x1": 0.565, "y1": -0.06, "line_color": rgb_to_color(self.bright_green),
    #          "fill_color": rgb_to_color(self.bright_green)},
    #         {"x0": 0.485, "y0": -0.09, "x1": 0.515, "y1": -0.06, "line_color": "green", "fill_color": "green"},
    #         {"x0": 0.435, "y0": -0.09, "x1": 0.465, "y1": -0.06, "line_color": "darkgreen", "fill_color": "darkgreen"}
    #     ]

    #     for params in squares_parameters:
    #         square = go.layout.Shape(
    #             type="rect",
    #             xref="paper", yref="paper",
    #             x0=params["x0"], y0=params["y0"],
    #             x1=params["x1"], y1=params["y1"],
    #             line=dict(color=params["line_color"]),
    #             fillcolor=params["fill_color"]
    #         )
    #         self.fig.add_shape(square)

    # def add_entry_zones(self, match: Match, opp):
    #     zones = const.DEF_ENTRY_ZONES_BBOX if opp else const.OFF_ENTRY_ZONES_BBOX
    #     entries =match.ser_zone_entry_xg_against if opp else match.ser_zone_entry_xg
    #     for key, x, y in self.iter_zones(zone_dict=zones):
    #         opacity = min(entries[key] * 1.5,0.9) # Limit opacity to a maximum of 0.9
    #         self.fig.add_trace(
    #             go.Scatter(
    #                 x=x, y=y,
    #                 mode="lines",
    #                 line={"color": rgb_to_color(self.bright_green), "width": 1, },
    #                 fill="toself",
    #                 fillcolor=rgb_to_color(self.bright_green, opacity=opacity),
    #                 showlegend=False,
    #             )
    #         )
    #         # Add annotation at the center of the zone with the xT value
    #         self.fig.add_annotation(
    #             x=x[0]/2+x[1]/2, y=y[0]/2+y[2]/2,
    #             text=f"{entries[key]:.2f} xG",
    #             showarrow=False,
    #             font={"color": rgb_to_color(self.white), "family": "Gilroy-Medium",
    #                   "size": 12 * self.font_size_multiplier},
    #         )

    # def add_recoveries(self, match: Match):
    #     df_plot = match.df_recoveries.copy()

    #     x = df_plot.start_x
    #     y = df_plot.start_y
    #     marker_size = (df_plot.xT * 100) + 4
    #     marker_color = np.where(df_plot.goal_10s, rgb_to_color(self.bright_green), rgb_to_color(self.white))
    #     # Add 0 before seconds under 10
    #     customdata = np.stack(
    #         [df_plot.name, df_plot.minute,
    #          np.where(df_plot.second < 10, '0' + df_plot.second.astype(str), df_plot.second.astype(str)), df_plot.xT],
    #         axis=-1
    #     )
    #     self.fig.add_trace(
    #         go.Scatter(
    #             x=x, y=y,
    #             mode="markers",
    #             marker={"size": marker_size, "color": marker_color},
    #             showlegend=False,
    #             hovertemplate='<b>Recovery</b><br>%{customdata[0]}<br> %{customdata[1]}:%{customdata[2]}<br>%{customdata[3]:.3f} xT within 10s in the following possession <extra></extra>',
    #             customdata=customdata,
    #         )
    #     )

    #     # Legends
    #     self.fig.update_layout(
    #         legend=dict(
    #             yanchor="bottom",
    #             entrywidth=80,
    #             x=0.05,
    #             y=-0.1,
    #             orientation="v"
    #         )
    #     )
    #     self.fig.add_trace(
    #         go.Scatter(
    #             x=[-10], y=[-10],
    #             mode="markers",
    #             marker={"size": 10, "color": rgb_to_color(self.white)},
    #             name="Recovery",
    #         )
    #     )

    #     self.fig.add_trace(
    #         go.Scatter(
    #             x=[-10], y=[-10],
    #             mode="markers",
    #             marker={"size": 10, "color": rgb_to_color(self.bright_green)},
    #             name="Goal within 10s",
    #         )
    #     )

    #     # Manually added in legend

    #     if self.pdf:
    #         self.fig.add_annotation(text="Higher xT",
    #                                 xref="paper", yref="paper",
    #                                 x=0.755, y=-0.03, showarrow=False,
    #                                 font={"family": "Gilroy-Medium", "size": 12 * self.font_size_multiplier,
    #                                       "color": rgb_to_color(self.white)})
    #     else:
    #         self.fig.add_annotation(text="Higher xT",
    #                                 xref="paper", yref="paper",
    #                                 x=0.72, y=-0.03, showarrow=False,
    #                                 font={"family": "Gilroy-Medium", "size": 12 * self.font_size_multiplier,
    #                                       "color": rgb_to_color(self.white)})

    #     self.fig.add_annotation(text="Lower xT",
    #                             xref="paper", yref="paper",
    #                             x=0.35, y=-0.03, showarrow=False,
    #                             font={"family": "Gilroy-Medium", "size": 12 * self.font_size_multiplier,
    #                                   "color": rgb_to_color(self.white)})

    #     self.fig.add_annotation(text="In comparison to other matches",
    #                             xref="paper", yref="paper",
    #                             x=0.5, y=-0.085, showarrow=False,
    #                             font={"family": "Gilroy-Medium", "size": 12 * self.font_size_multiplier,
    #                                   "color": '#80958D'})

    #     squares_parameters = [
    #         {"x0": 0.535, "y0": -0.02, "x1": 0.565, "y1": 0.01, "line_color": rgb_to_color(self.bright_green),
    #          "fill_color": rgb_to_color(self.bright_green)},
    #         {"x0": 0.485, "y0": -0.02, "x1": 0.515, "y1": 0.01, "line_color": "green", "fill_color": "green"},
    #         {"x0": 0.435, "y0": -0.02, "x1": 0.465, "y1": 0.01, "line_color": "darkgreen", "fill_color": "darkgreen"}
    #     ]

    #     for params in squares_parameters:
    #         square = go.layout.Shape(
    #             type="rect",
    #             xref="paper", yref="paper",
    #             x0=params["x0"], y0=params["y0"],
    #             x1=params["x1"], y1=params["y1"],
    #             line=dict(color=params["line_color"]),
    #             fillcolor=params["fill_color"]
    #         )
    #         self.fig.add_shape(square)

    #     self.fig.add_annotation(text="xT within 10s",
    #                             xref="paper", yref="paper",
    #                             x=1.04, y=-0.085, showarrow=False,
    #                             font={"family": "Gilroy-Medium", "size": 12 * self.font_size_multiplier,
    #                                   "color": '#80958D'})

    #     self.fig.add_annotation(text="Low",
    #                             xref="paper", yref="paper",
    #                             x=0.88, y=-0.03, showarrow=False,
    #                             font={"family": "Gilroy-Medium", "size": 12 * self.font_size_multiplier,
    #                                   "color": rgb_to_color(self.white)})

    #     self.fig.add_annotation(text="High",
    #                             xref="paper", yref="paper",
    #                             x=1.10, y=-0.03, showarrow=False,
    #                             font={"family": "Gilroy-Medium", "size": 12 * self.font_size_multiplier,
    #                                   "color": rgb_to_color(self.white)})

    #     circles_parameters = [
    #         {"x_center": 0.9, "y_center": -0.0, "radius": 0.005, "line_color": "white",
    #          "fill_color": "white"},
    #         {"x_center": 0.94, "y_center": -0.0, "radius": 0.01, "line_color": "white", "fill_color": "white"},
    #         {"x_center": 0.99, "y_center": -0.0, "radius": 0.015, "line_color": "white",
    #          "fill_color": "white"}
    #     ]

    #     for params in circles_parameters:
    #         circle = go.layout.Shape(
    #             type="circle",
    #             xref="paper", yref="paper",
    #             x0=params["x_center"] - params["radius"], y0=params["y_center"] - params["radius"],
    #             x1=params["x_center"] + params["radius"], y1=params["y_center"] + params["radius"],
    #             line=dict(color=params["line_color"]),
    #             fillcolor=params["fill_color"]
    #         )
    #         self.fig.add_shape(circle)

    #     par = {
    #         "x_center": 0.01, "y_center": -0.1, "radius": 0.012, "line_color": 'green', "fill_color": 'green'
    #     }

    # def add_recovery_zones(self, match: Match):
    #     for key, x, y in self.iter_zones(zone_dict=const.THIRDS_ZONES_BBOX):
    #         opacity = match.ser_zone_recovery_xt[key] * 0.8
    #         self.fig.add_trace(
    #             go.Scatter(
    #                 x=x, y=y,
    #                 mode="lines",
    #                 line={"color": rgb_to_color(self.bright_green), "width": 1, },
    #                 fill="toself",
    #                 fillcolor=rgb_to_color(self.bright_green, opacity=opacity),
    #                 showlegend=False,
    #                 hoverinfo="skip"
    #             )
    #         )
    #         # Add annotation at the bottom of the zone with the xT value
    #         self.fig.add_annotation(
    #             x=x[0] / 2 + x[1] / 2, y=y[0] + 10,
    #             text=f"{match.ser_zone_recovery_xt[key]:.2f} xT",
    #             # text=f"<span style='background-color: rgba(255, 255, 255, 0.7); padding: 5px;'>{match.ser_zone_recovery_xt[key]:.2f} xT</span>",
    #             showarrow=False,
    #             font={"color": rgb_to_color(self.white), "family": "Gilroy-Medium",
    #                   "size": 16 * self.font_size_multiplier},
    #         )

    #         # Add rectangle behind text - Not done
    #         '''
    #         self.fig.add_shape(
    #             type="rect",
    #             x0=x[0] / 2 + x[1] / 2 - 7,
    #             x1=x[0] / 2 + x[1] / 2 + 7,
    #             y0=y[0] + 10 - 5,
    #             y1=y[0] + 10 + 5,
    #             fillcolor=rgb_to_color(self.bright_green, opacity=opacity),
    #             line=dict(
    #                 color='rgba(0,0,0,0)',  # Setting line color to fully transparent
    #             ))
    #         '''

    # def add_defensive_actions(self, match: Match):
    #     # Scatter defensive actions
    #     df_plot = match.df_defensive_actions
    #     x = df_plot.start_x
    #     y = df_plot.start_y
    #     # Add 0 before seconds under 10
    #     customdata = np.stack(
    #         [df_plot.minute,
    #          np.where(df_plot.second < 10, '0' + df_plot.second.astype(str), df_plot.second.astype(str)),
    #          df_plot['Time to defensive action']], axis=-1
    #     )
    #     self.fig.add_trace(
    #         go.Scatter(
    #             x=x, y=y,
    #             mode="markers",
    #             marker={"size": 5, "color": rgb_to_color(self.white)},
    #             showlegend=True,
    #             name='Turnover',
    #             hovertemplate='<b>Turnover</b><br>%{customdata[0]}:%{customdata[1]}<br>%{customdata[2]:.3f} sec to defensive action<extra></extra>',
    #             customdata=customdata,
    #         )
    #     )

    #     # Legends
    #     self.fig.update_layout(
    #         legend=dict(
    #             yanchor="bottom",
    #             entrywidth=70,
    #             y=-0.1,
    #             x=0.2
    #         )
    #     )

    #     if self.pdf is False:
    #         x_less = 0.982
    #         x_more = 0.58
    #         x_comp = 0.985

    #     else:
    #         x_less = 0.95
    #         x_more = 0.6
    #         x_comp = 0.92

    #     # Manually added in legend
    #     self.fig.add_annotation(text="Less time",
    #                             xref="paper", yref="paper",
    #                             x=x_less, y=-0.07, showarrow=False,
    #                             font={"family": "Gilroy-Medium", "size": 12 * self.font_size_multiplier,
    #                                   "color": rgb_to_color(self.white)})

    #     self.fig.add_annotation(text="More time",
    #                             xref="paper", yref="paper",
    #                             x=x_more, y=-0.07, showarrow=False,
    #                             font={"family": "Gilroy-Medium", "size": 12 * self.font_size_multiplier,
    #                                   "color": rgb_to_color(self.white)})

    #     self.fig.add_annotation(text="In comparison to other matches",
    #                             xref="paper", yref="paper",
    #                             x=x_comp, y=-0.11, showarrow=False,
    #                             font={"family": "Gilroy-Medium", "size": 12 * self.font_size_multiplier,
    #                                   "color": '#80958D'})

    #     squares_parameters = [
    #         {"x0": 0.78, "y0": -0.06, "x1": 0.81, "y1": -0.03, "line_color": rgb_to_color(self.bright_green),
    #          "fill_color": rgb_to_color(self.bright_green)},
    #         {"x0": 0.73, "y0": -0.06, "x1": 0.76, "y1": -0.03, "line_color": "green", "fill_color": "green"},
    #         {"x0": 0.68, "y0": -0.06, "x1": 0.71, "y1": -0.03, "line_color": "darkgreen", "fill_color": "darkgreen"}
    #     ]

    #     for params in squares_parameters:
    #         square = go.layout.Shape(
    #             type="rect",
    #             xref="paper", yref="paper",
    #             x0=params["x0"], y0=params["y0"],
    #             x1=params["x1"], y1=params["y1"],
    #             line=dict(color=params["line_color"]),
    #             fillcolor=params["fill_color"]
    #         )
    #         self.fig.add_shape(square)

    # def add_defensive_zones(self, match: Match):
    #     # Plot time to defensive action for each third
    #     for key, x, y in self.iter_zones(zone_dict=const.THIRDS_ZONES_BBOX):
    #         opacity = max(1.1 - match.ser_zone_defensive_time[key] * 0.1, 0)
    #         if opacity != opacity:  # NaN check
    #             opacity = 0
    #         self.fig.add_trace(
    #             go.Scatter(
    #                 x=x, y=y,
    #                 mode="lines",
    #                 line={"color": rgb_to_color(self.bright_green), "width": 1, },
    #                 fill="toself",
    #                 fillcolor=rgb_to_color(self.bright_green, opacity=opacity),
    #                 showlegend=False,
    #                 hoverinfo="skip"
    #             )
    #         )
    #         # Add annotation at the bottom of the zone with the value
    #         self.fig.add_annotation(
    #             x=x[0] / 2 + x[1] / 2, y=y[0] + 10,
    #             text=f"{match.ser_zone_defensive_time[key]:.1f} Sec" if match.ser_zone_defensive_time[key] ==
    #                                                                     match.ser_zone_defensive_time[
    #                                                                         key] else "No data",
    #             showarrow=False,
    #             font={"color": rgb_to_color(self.white), "family": "Gilroy-Medium",
    #                   "size": 16 * self.font_size_multiplier},
    #         )

    # def add_shots(self, match: Match):
    #     shots_df = match.df_shots.copy()
    #     shots_df['category'] = shots_df['outcome']
    #     labels = {'off-target': 'Shot', 'on-target': 'Shot on target', 'goal': 'Goal'}
    #     shots_df['category'] = shots_df['category'].replace(labels)
    #     customdata = np.stack(
    #         [shots_df.name, shots_df.minute + 1, shots_df.xG, shots_df.category], axis=-1
    #     )
    #     # Affects opacity @Matthias
    #     shots_df['ms'] = shots_df['xG'] * 20 + 10

    #     masks = [shots_df['outcome'] == "off-target", shots_df['outcome'] == 'on-target', shots_df['goal']]
    #     markers = [
    #         {"symbol": "circle-open", "color": rgb_to_color(self.white, opacity=1),
    #          "line": {"color": rgb_to_color(self.white, opacity=1), "width": 2}},
    #         {"symbol": "circle", "color": rgb_to_color(self.white, opacity=0.75),
    #          "line": {"color": rgb_to_color(self.white, opacity=1), "width": 2}},
    #         {"symbol": "circle", "color": rgb_to_color(self.bright_green),
    #          "line": {"color": rgb_to_color(self.white), "width": 2}}]

    #     names = ["Shot", "On target", "Goal"]
    #     filtered_data = [shots_df[mask] for mask in masks]
    #     temp_customdata = [customdata[mask] for mask in masks]

    #     for data, marker, name, custom in zip(filtered_data, markers, names, temp_customdata):
    #         self.fig.add_trace(
    #             go.Scatter(
    #                 x=100 - data['start_y'], y=data['start_x'],
    #                 mode="markers",
    #                 marker=marker,
    #                 marker_size=data['ms'],
    #                 showlegend=False,
    #                 name=name,
    #                 customdata=custom,
    #                 hovertemplate='<b>%{customdata[3]}</b><br>%{customdata[0]}<br>Minute: %{customdata[1]}<br>%{customdata[2]:.2f} xG<extra></extra>',

    #             )
    #         )

    #         self.fig.add_trace(
    #             go.Scatter(
    #                 x=[-100],
    #                 y=[0],
    #                 mode="markers",
    #                 marker=marker,
    #                 name=name,
    #                 showlegend=True,
    #                 marker_size=10
    #             )
    #         )

    #     self.fig.update_layout(
    #         legend=dict(
    #             y=-0.05,
    #         )
    #     )

    #     # Manually added in legend
    #     self.fig.add_annotation(text="Higher xG",
    #                             xref="paper", yref="paper",
    #                             x=0.735, y=-0.13, showarrow=False,
    #                             font={"family": "Gilroy-Medium", "size": 12 * self.font_size_multiplier,
    #                                   "color": rgb_to_color(self.white)})

    #     self.fig.add_annotation(text="Lower xG",
    #                             xref="paper", yref="paper",
    #                             x=0.355, y=-0.13, showarrow=False,
    #                             font={"family": "Gilroy-Medium", "size": 12 * self.font_size_multiplier,
    #                                   "color": rgb_to_color(self.white)})

    #     circles_parameters = [
    #         {"x_center": 0.550, "y_center": -0.105, "radius": 0.015, "line_color": "white",
    #          "fill_color": "white"},
    #         {"x_center": 0.5, "y_center": -0.105, "radius": 0.01, "line_color": "white", "fill_color": "white"},
    #         {"x_center": 0.451, "y_center": -0.105, "radius": 0.005, "line_color": "white",
    #          "fill_color": "white"}
    #     ]

    #     for params in circles_parameters:
    #         circle = go.layout.Shape(
    #             type="circle",
    #             xref="paper", yref="paper",
    #             x0=params["x_center"] - params["radius"], y0=params["y_center"] - params["radius"],
    #             x1=params["x_center"] + params["radius"], y1=params["y_center"] + params["radius"],
    #             line=dict(color=params["line_color"]),
    #             fillcolor=params["fill_color"]
    #         )
    #         self.fig.add_shape(circle)

    # def add_shot_zones(self, match: Match):
    #     for key, x, y in self.iter_zones():
    #         opacity = match.ser_zone_shot_xG[key] * 1.5
    #         self.fig.add_trace(
    #             go.Scatter(
    #                 x=x, y=y,
    #                 mode="lines",
    #                 line={"color": rgb_to_color(self.bright_green), "width": 1, },
    #                 fill="toself",
    #                 fillcolor=rgb_to_color(self.bright_green, opacity=opacity),
    #                 showlegend=False,
    #             )
    #         )
    #         # Add annotation at the center of the zone with the xG value
    #         self.fig.add_annotation(
    #             x=x[0] / 2 + x[1] / 2, y=y[0] / 2 + y[2] / 2,
    #             text=f"{match.ser_zone_shot_xG[key]:.2f} xG",
    #             showarrow=False,
    #             font={"color": rgb_to_color(self.white), "family": "Gilroy-Medium",
    #                   "size": 12 * self.font_size_multiplier},
    #         )

    # def add_match(self, match: Match, n_group, info_dict = {}):
    #     opp = info_dict.get("opponent", False)
    #     for method in info_dict.get("methods", []):
    #         if method == "add_entry_zones":
    #             self.add_entry_zones(match, opp=opp)
    #         elif method == "add_entries":
    #             self.add_entries(match, opp=opp)
    #         elif method == "add_recovery_zones":
    #             self.add_recovery_zones(match)
    #         elif method == "add_recoveries":
    #             self.add_recoveries(match)
    #         elif method == "add_defensive_zones":
    #             self.add_defensive_zones(match)
    #         elif method == "add_defensive_actions":
    #             self.add_defensive_actions(match)
    #         elif method == "add_shots":
    #             self.add_shots(match)
    #         elif method == "add_extra_info":
    #             self.add_extra_info(match)





class VerticalPitchVisual(PitchVisual):
    def _add_pitch(self):
        self.fig.update_layout(
            hoverdistance=100,
            xaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=False,
                zeroline=False,
                range=[-0.2, 100],
                constrain="domain",
                fixedrange=True,
            ),
            yaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=False,
                zeroline=False,
                range=[50, 100.2],
                scaleanchor="x",
                scaleratio=1.544,
                constrain="domain",
                fixedrange=True,
            ),
        )
        shapes = self._get_shapes()
        for shape in shapes:
            if shape["type"] != "path":
                shape.update(dict(line={"color": "white", "width": 2}, xref="x", yref="y", ))
                shape["x0"], shape["x1"], shape["y0"], shape["y1"] = shape["y0"], shape["y1"], shape["x0"], shape["x1"]
                self.fig.add_shape(**shape)

        # Add the arcs
        arcs = [
            dict(
                type="path", path=self.ellipse_arc(50, 89, 9.5 * 1.35, 6.2 * 1.35, -0.8 * np.pi, -0.2 * np.pi),
            ),
            dict(
                type="path", path=self.ellipse_arc(100, 100, 9.5 * 0.3, 6.2 * 0.3, np.pi, 3 / 2 * np.pi),
            ),
            dict(
                type="path", path=self.ellipse_arc(0, 100, 9.5 * 0.3, 6.2 * 0.3, 3 / 2 * np.pi, 2 * np.pi),
            )]

        for arc in arcs:
            arc.update(dict(line={"color": "white", "width": 2}, xref="x", yref="y", ))
            self.fig.add_shape(**arc)

    def iter_zones(self):
        for key, value in const.SHOT_ZONES_BBOX.items():
            x = [
                value["y_lower_bound"],
                value["y_upper_bound"],
                value["y_upper_bound"],
                value["y_lower_bound"],
                value["y_lower_bound"],
            ]
            y = [
                value["x_lower_bound"],
                value["x_lower_bound"],
                value["x_upper_bound"],
                value["x_upper_bound"],
                value["x_lower_bound"],
            ]
            yield key, x, y



class ShotVisual(VerticalPitchVisual):
    def __init__(self, *args, **kwargs):
        self.line_color = 'white'
        self.line_width = 3
        self.shot_color = rgb_to_color(self.bright_blue)
        self.failed_color = rgb_to_color(self.bright_yellow)
        self.bbox_color = rgb_to_color(self.bright_green)
        self.marker_size = 15
        self.basic_text_size = 10
        self.text_size = 20
        super().__init__(*args, **kwargs)

    def add_shots(self, shots):

        shots_df = shots.df_shots.copy()
        shot_contribution= shots.df_contributions.copy()    

        shots_df['category'] = shots_df['goal']
        labels = {False: 'Shot', True: 'Goal'}
        shots_df['category'] = shots_df['category'].replace(labels)
        arrays_to_stack = [
                            #shots_df.player_name, shots_df.minute+1, 
                            shots_df.xG
                            #, shots_df.category,
                            #shot_contribution.angle.to_numpy(),
                            #shot_contribution.head_shot.to_numpy(),
                            #shot_contribution.match_state.to_numpy(),
                            #shot_contribution.strong_foot.to_numpy(),
                            #shot_contribution.assist_smart_pass.to_numpy(),
                            #shot_contribution.assist_cross.to_numpy(),
                            #shot_contribution.possession_counterattack.to_numpy(),
                            #shot_contribution.clear_header.to_numpy(),
                            #shot_contribution.rebound.to_numpy(),
                            #shot_contribution.assist_key_pass.to_numpy(),
                            #shot_contribution.self_created_shot.to_numpy()
                        ]

        # Stack the arrays
        customdata = np.stack(arrays_to_stack, axis=-1)

  
        shots_df['ms'] = shots_df['xG'] * 20 + 10

        masks = [shots_df['goal'] == False, shots_df['goal']== True]
        markers = [
            {"symbol": "circle-open", "color": rgb_to_color(self.white, opacity=1),
             "line": {"color": rgb_to_color(self.white, opacity=1), "width": 2}},
            {"symbol": "circle", "color": rgb_to_color(self.bright_green),
             "line": {"color": rgb_to_color(self.white), "width": 2}}]

        names = ["Shot", "Goal"]
        filtered_data = [shots_df[mask] for mask in masks]
        temp_customdata = [customdata[mask] for mask in masks]

        for data, marker, name, custom in zip(filtered_data, markers, names, temp_customdata):
            if custom.size == 0:  # Skip if customdata is empty
                continue
            # hovertemplate = ('<b>%{customdata[3]}</b><br>%{customdata[0]}<br>Minute: %{customdata[1]}<br>'
            #                  '<b>xG:</b> %{customdata[2]:.3f}<br>')

            # feature_names = ['Angle', 'Header', 'Match State', 'Strong Foot', 'Smart Pass', 'Cross',
            #                  'Counterattack', 'Clear Header', 'Rebound', 'Key Pass', 'Self Created Shot']
            # for i, feature_name in enumerate(feature_names, start=4):
            #     hovertemplate += f'<b>{feature_name}:</b> %{{customdata[{i}]:.3f}}<br>' if custom[0][i] > 0.005 else ''

            #hovertemplate += '<extra></extra>'
            self.fig.add_trace(
                go.Scatter(
                    x=(68 - data['start_y'])*100/68, y=data['start_x']*100/105,
                    mode="markers",
                    #marker=marker,
                    marker=dict(size=10),
                    #marker_size=data['ms'],
                    showlegend=False,
                    name=name,
                    customdata=custom,
                    #hovertemplate=hovertemplate,
                )
            )
            self.fig.add_trace(
                go.Scatter(
                    x=[-100],
                    y=[0],
                    mode="markers",
                    marker=marker,
                    name=name,
                    showlegend=True,
                    marker_size=10
                )
            )

        self.fig.update_layout(
            legend=dict(
                y=-0.05,
            )
        )

    # def add_title_from_match(self, match):
    #     title = f"How were {match.team_name}'s shots against {match.opp_team_name}?"
    #     subtitle = f"{match.competition.get_plot_subtitle()} | {match.date.split(' ')[0]}"
    #     self.add_title(title, subtitle)
    
    # def add_title_from_match_player(self, match, player):
    #     title = f"How were {player}'s shots in a match of {match.team_name} against {match.opp_team_name}?"
    #     subtitle = f"{match.competition.get_plot_subtitle()} | {match.date.split(' ')[0]}"
    #     self.add_title(title, subtitle)


    def add_shot(self, shots, shot_id):
        # Filter for the specific shot using the shot_id
        shot_data = shots.df_shots[shots.df_shots['id'] == shot_id]
        if shot_data.empty:
            raise ValueError(f"Shot with ID {shot_id} not found.")

        # Extract the shot coordinates (start_x, start_y)
        shot_x = shot_data['start_x'].values[0]
        shot_y = shot_data['start_y'].values[0]
        goal_status = shot_data['goal'].values[0]  # True if goal, False if no goal
        xG_value = shot_data['xG'].values[0]  # Get the xG value
        # Add existing logic to plot shots here...
        shot_data['category'] = shot_data['goal']
        labels = {False: 'Shot', True: 'Goal'}
        shot_data['category'] = shot_data['category'].replace(labels)

        # Extract teammate coordinates (e.g., teammate_1_x, teammate_1_y, etc.)
        teammate_x_cols = [col for col in shot_data.columns if 'teammate' in col and '_x' in col]
        teammate_y_cols = [col for col in shot_data.columns if 'teammate' in col and '_y' in col]

        teammate_x = shot_data[teammate_x_cols].values[0]
        teammate_y = shot_data[teammate_y_cols].values[0]

        # Extract opponent coordinates (e.g., opponent_1_x, opponent_1_y, etc.)
        opponent_x_cols = [col for col in shot_data.columns if 'opponent' in col and '_x' in col]
        opponent_y_cols = [col for col in shot_data.columns if 'opponent' in col and '_y' in col]

        opponent_x = shot_data[opponent_x_cols].values[0]
        opponent_y = shot_data[opponent_y_cols].values[0]

        # Plot the shot location (start_x, start_y)
        self.fig.add_trace(
            go.Scatter(
                x=[(68 - shot_y) * 100 / 68],
                y=[shot_x * 100 / 105],
                mode="markers",
                marker=dict(size=12, color=self.shot_color, symbol="circle"),
                #text=[f"Goal: {goal_status}<br>xG: {xG_value:.2f}"],
                textposition='top center',
                name="Shot",
                showlegend=True
            )
        )

        # Plot teammates' locations
        self.fig.add_trace(
            go.Scatter(
                x=(68 - teammate_y) * 100 / 68,
                y=teammate_x * 100 / 105,
                mode="markers",
                marker=dict(size=10, color='blue', symbol="circle-open"),
                name="Teammates",
                showlegend=True
            )
        )

        # Plot opponents' locations
        self.fig.add_trace(
            go.Scatter(
                x=(68 - opponent_y) * 100 / 68,
                y=opponent_x * 100 / 105,
                mode="markers",
                marker=dict(size=10, color='red', symbol="x"),
                name="Opponents",
                showlegend=True
            )
        )

        self.fig.update_layout(
            title={
                'text': f"Shot Visualization | Outcome: {'Goal' if goal_status else 'No Goal'} | xG: {xG_value:.2f}",
                'font': {'color': 'white'}  # Set the title color to white
            },
            legend=dict(y=-0.05),
)
