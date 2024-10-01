# %%
import pandas as pd

annotation_text = "<span style=''>{metric_name}: {data:.2f} per 90</span>"

# series with col "goals" and some dummy value
ser_plot = pd.Series({"goals": 0.5})
col = "goals"

print(annotation_text.format(metric_name="Goals", data=ser_plot[col]))
print()

# %%
