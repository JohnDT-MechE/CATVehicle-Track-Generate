import pandas as pd
import numpy as np

#read in the csv file
filename = "data.csv"

df = pd.read_csv(filename)

shape = df.shape

#get a dictionary mapping each event to a unique id
unique_dict = {}
for index, event_type in enumerate({c: df[c].unique() for c in df}['event']):
    unique_dict[event_type] = index


#loop through the rows in the data
for row_index in range(shape[0]):
    time = df.iloc[row_index, 0]
    event_type = unique_dict[df.iloc[row_index, 1]]

    print(time)
    print(event_type)