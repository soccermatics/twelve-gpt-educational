from classes.data_source import CountryStats
from classes.data_point import Country


countries = CountryStats()

metrics = [m for m in countries.df.columns if m not in ["country"]]

countries.calculate_statistics(metrics=metrics)


print()
