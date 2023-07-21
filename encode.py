import pandas as pd
import numpy as np

def encode_event(t, e, reverse=False):
    """
    Encodes a single event event, adding it to data
    """
    gray_code = {}
    for i in range(0, 1<<10):
        gray=i^(i>>1)
        gray_code[i] = "{0:0{1}b}".format(gray,5)

    normal_map = {'in left': '00', 'in right': '01', 'out left':'10', 'out right':'11'}
    reverse_map = {'out right': '00', 'out left': '01', 'in right':'10', 'in left':'11'}

    return gray_code[((int)(t/2) % 32)] + (normal_map[e] if not reverse else reverse_map[e])

def block_encoding(filename, start_time, block_size, num_blocks, time_gap=0, reverse=False):
    """
    Ecodes data from a csv into a list of encoded 'blocks', each containing all the events that occurred within a given time
    """
    df = pd.read_csv(filename)

    blocks = ['' for _ in range(num_blocks)]
    
    for _, entry in df.iterrows():
        t = entry.iloc[0]
        e = entry.iloc[1]

        block = int(t - start_time)//(block_size+time_gap)

        if block >= 0 and block < num_blocks and (t - start_time - block*(block_size+time_gap)) <= block_size:
            blocks[block] += (encode_event(t, e, reverse))

    return blocks

def filter_timestamps(file1, file2, cutoff):
    """
    Takes in two data sources, filters them to isolate events that occurred at a similar time (to remove erroneous events
    that aren't identified by both prespectives)

    Cutoff defines how far apart events can be to be considered 'at the same time'
    """
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    matched_events = []

    gray_code = {}
    for i in range(0, 1<<10):
        gray=i^(i>>1)
        gray_code[i] = "{0:0{1}b}".format(gray,5)

    normal_map = {'in left': '00', 'in right': '01', 'out left':'10', 'out right':'11'}
    reverse_map = {'out right': '00', 'out left': '01', 'in right':'10', 'in left':'11'}

    total_diff = 0
    events = 0

    data_string1 = ''
    data_string2 = ''

    #iterate over all the times in file 1
    for _, entry1 in df1.iterrows():
        t1 = entry1.iloc[0]
        e1 = entry1.iloc[1]
        for _, entry2 in df2.iterrows():
            t2 = entry2.iloc[0]
            e2 = entry2.iloc[1]
            #check if the times are within the cutoff
            if abs(t1-t2) < cutoff:
                #check we haven't already matched something to this event
                if t2 not in matched_events:
                    matched_events.append(t2)
                    status = 'good' if reverse_map[e2] == normal_map[e1] else 'bad'
                    total_diff += abs(t1 - t2)
                    events += 1
                    print('Match: ' + e1 + ' ' + e2 +'  \t\t' + str(t1) + ' ' + str(t2) + '\tDiff: ' + str(int((t1 - t2)*100)/100) +  '\t' + status)


                    data_string1 += gray_code[((int)(t1) % 32)] + normal_map[e1]
                    data_string2 += gray_code[((int)(t2) % 32)] + reverse_map[e2]

    print("Number of good events: " + str(events))
    print("Percentage of good events:\t file1: " + str(events/df1.shape[0]) + '; file2: ' + str(events/df2.shape[0]))
    print("Mean Difference Between 'good' events: " + str(total_diff/events))

    print('\n Filtered Data strings:')
    print(data_string1)
    print(data_string2)
    
    print((len(data_string1) - sum([1 if data_string1[index] != data_string2[index] else 0 for index in range(len(data_string1))])) / len(data_string1))

            

#only run this code if we are running the file on its own, otherwise just let whatever code called encode
#handle the input and output to the function
if __name__ == "__main__":
    file1 = "data_rear_ultrawide.csv"
    file2 = "data_front_ultrawide.csv"
    #print(encode(file, flipped=False))

    filter_timestamps(file1, file2, 3)

    data1 = block_encoding(file1, 1689368390, block_size=10, num_blocks=10, time_gap = 0)
    data2 = block_encoding(file2, 1689368390, block_size=10, num_blocks=10, time_gap = 0, reverse = True)

    total_accuracy = 0
    num_blocks_counted = 0

    for i in range(len(data1)):
        block1 = data1[i]
        block2 = data2[i]
        length = min(len(block1), len(block2))
        if length != 0:
            accuracy_percent = (length - sum([1 if block1[index] != block2[index] else 0 for index in range(length)])) / length
            total_accuracy += accuracy_percent
            num_blocks_counted += 1
        else:
            accuracy_percent = 'N/A'

        print('Block #: ' + str(i) + '\n\t Block one: ' + block1 + '\n\t Block two: ' + block2 + '\n\t Accuracy: ' + str(accuracy_percent))
    print('\n\nAverage Accuracy: ' + str(total_accuracy/num_blocks_counted))