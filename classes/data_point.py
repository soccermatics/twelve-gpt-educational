import pandas as pd
import numpy as np


# Mostly used for unpacking ser_info and methods relative to a single data point
# (Getting a title, adjusting metrics for transfer, etc...)
class DataPoint:
    def __init__(self, ser_info):
        pass


class Stat(DataPoint):
    pass


class Player(Stat):

    def __init__(
        self, id, name, minutes_played, gender, position, ser_metrics, relevant_metrics
    ):

        # Unpack ser_info
        self.id = id
        self.name = name
        self.minutes_played = minutes_played
        self.gender = gender
        self.position = position
        self.relevant_metrics = relevant_metrics
        # Save metrics as a Series
        self.ser_metrics = ser_metrics


class Country(Stat):

    def __init__(self, id, name, ser_metrics, relevant_metrics, drill_down_metrics):

        # Unpack ser_info
        self.id = id
        self.name = name

        self.relevant_metrics = relevant_metrics
        # Save metrics as a Series
        self.ser_metrics = ser_metrics

        self.drill_down_metrics = drill_down_metrics


class Person(Stat):

    def __init__(self, id, name, ser_metrics):

        # Unpack ser_info
        self.id = id
        self.name = name
        self.ser_metrics = ser_metrics

class Individual(Stat):

    def __init__(self,id,ser_metrics):

        # Unpack ser_info
        self.id=id
        self.ser_metrics = ser_metrics  
        