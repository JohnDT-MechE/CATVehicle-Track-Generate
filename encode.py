import pandas as pd
import numpy as np


def encode(filename, flipped=False):
    """
    This function takes in the filename of a CSV (needs to have headers) with columns for time and event type

    Reads into a dataframe
    Iterates through the rows, and then
    creates data from this by taking the times and classes
    then converts to a binary string
    """
    df = pd.read_csv(filename)

    shape = df.shape

    #get a dictionary mapping each event to a unique id
    unique_dict = {}
    for index, event_type in enumerate({c: df[c].unique() for c in df}['event']):
        unique_dict[event_type] = index

    gray_code = {}
    for i in range(0, 1<<5):
        gray=i^(i>>1)
        gray_code[i] = "{0:0{1}b}".format(gray,5)

    normal_map = {0: '00', 1: '01', 2:'10', 3:'11'}
    reverse_map = {0: '10', 1: '11', 2:'00', 3:'01'}


    code = ''
    #loop through the rows in the data
    for row_index in range(shape[0]):
        time = df.iloc[row_index, 0]
        event_type = unique_dict[df.iloc[row_index, 1]]

        code += gray_code[((int)(time/2) % 32)]
        code += normal_map[event_type] if not flipped else reverse_map[event_type]

    return code

#only run this code if we are running the file on its own, otherwise just let whatever code called encode
#handle the input and output to the function
if __name__ == "__main__":
    file = "data.csv"
    print(encode(file, flipped=False))