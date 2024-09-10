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

    def __init__(self,id,name, minutes_played,gender,position,ser_metrics,relevant_metrics):

        # Unpack ser_info
        self.id=id
        self.name = name
        self.minutes_played = minutes_played
        self.gender = gender
        self.position = position
        

        self.relevant_metrics = relevant_metrics
        # Save metrics as a Series
        self.ser_metrics = ser_metrics

class PersonalityStats(Stat):
    data_point_class = data_point.Person
    
    def __init__(self):
        super().__init__()

    def get_raw_data(self):
        df = pd.read_csv("data/events/dataset.csv",encoding='unicode_escape')
        return df

    def to_data_point(self,name,extraversion,neurotiscism,agreeableness,conscientiousness,openness) -> data_point.Person:
        
        id = self.df.index
        name = self.name
        extraversion = self.extraversion
        neurotiscism = self.neurotiscism
        agreeableness = self.agreeableness
        conscientiousness = self.conscientiousness
        openness = self.openness

        
        return self.data_point_class(id=id,name=name, extraversion=extraversion,neurotiscism=neurotiscism,agreeableness=agreeableness,conscientiousness=conscientiousness,openness=openness)
