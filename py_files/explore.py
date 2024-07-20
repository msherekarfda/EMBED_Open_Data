# This file has random scripts to explore data in a .csv file
import pandas as pd

df = pd.read_csv('master_list_for_split.csv', low_memory=False)
print(df.head())
print(df.columns)