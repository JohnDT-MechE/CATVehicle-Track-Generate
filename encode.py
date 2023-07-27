import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def encode_event(t, e, reverse=False, time_resolution=4, bits_to_drop=1):
    """
    Encodes a single event event, adding it to data
    """
    gray_code = {}
    for i in range(0, 1<<10):
        gray=i^(i>>1)
        gray_code[i] = "{0:0{1}b}".format(gray,5)

    normal_map = {'in left': '0001', 'in right': '0010', 'out left':'1000', 'out right':'0100'}
    reverse_map = {'out right': '0001', 'out left': '0010', 'in right':'1000', 'in left':'0100'}

    return gray_code[int(t/(1<<bits_to_drop)) % (1<<time_resolution)] + (normal_map[e] if not reverse else reverse_map[e])

def encode_zone(filename, start_time, length):
    """
    Takes in a file with sequential data about how many vehicles are in a zone at any given time
    Takes the average number of vehicles over the time period
    """
    gray_code = {}
    for i in range(0, 1<<10):
        gray=i^(i>>1)
        gray_code[i] = "{0:0{1}b}".format(gray,5)

    df = pd.read_csv(filename)

    num_frames = 0
    num_vehicles = 0

    for _, entry in df.iterrows():
        t = entry.iloc[0]
        num = entry.iloc[-1]

        if t > start_time and t < start_time + length:
            num_vehicles += int(num)
            num_frames += 1

    average_vehicles = num_vehicles/num_frames

    return gray_code[(average_vehicles*10)//1 % 16]

def block_encoding(filename, start_time, block_size, num_blocks, time_gap=0, reverse=False, tres=4, bits_to_drop=1, zone_name=None):
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
            blocks[block] += (encode_event(t, e, reverse, time_resolution=tres, bits_to_drop=bits_to_drop))
    if zone_name is not None:
        for index in range(len(blocks)):
            blocks[index] += encode_zone(zone_name, start_time + index*block_size, block_size)

    print(blocks)
    return blocks

def filter_timestamps(file1, file2, cutoff, f):
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

    normal_map = {'in left': '0001', 'in right': '0100', 'out left':'0010', 'out right':'1000'}
    reverse_map = {'out right': '0001', 'out left': '0100', 'in right':'0010', 'in left':'1000'}

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


                    data_string1 += gray_code[((int)(t1/2) % 16)] + normal_map[e1]
                    data_string2 += gray_code[((int)(t2/2) % 16)] + reverse_map[e2]

    f.write("\n\nNumber of good events: " + str(events))
    f.write("\nPercentage of good events:\t file1: " + str(events/df1.shape[0]) + '; file2: ' + str(events/df2.shape[0]))
    f.write("\nMean Difference Between 'good' events: " + str(total_diff/events))

    f.write('\n Filtered Data strings:')
    f.write('\n' + data_string1)
    f.write('\n' + data_string2)
    
    f.write('\n' + str((len(data_string1) - sum([1 if data_string1[index] != data_string2[index] else 0 for index in range(len(data_string1))])) / len(data_string1)))


def validate_block(data1, data2):
    """
    Takes in two lists encoded in blocks, and compares the average accuracy
    """

    total_accuracy = 0
    num_blocks_counted = 0
    
    for i in range(len(data1)):
        block1 = data1[i]
        block2 = data2[i]
        length = min(len(block1), len(block2))
        max_len = max(len(block1), len(block2))
        if len(block1) < len(block2):
            short = block1
            long = block2
        else:
            short = block2
            long = block1
        if len(short) != 0:
            short = short + '0'*(max_len-length)
            accuracy_percent = (max_len - sum([1 if short[index] != long[index] else 0 for index in range(max_len)])) / max_len
            total_accuracy += accuracy_percent
            num_blocks_counted += 1
        else:
            accuracy_percent = 'N/A'

        #output = '\nBlock #: ' + str(i) + '\n\t Block one: ' + block1 + '\n\t Block two: ' + block2 + '\n\t Accuracy: ' + str(accuracy_percent) + '\n'

    return total_accuracy/num_blocks_counted

def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.4f} + {:.4f}x'.format(a, b))

    return a, b


#only run this code if we are running the file on its own, otherwise just let whatever code called encode
#handle the input and output to the function
def graphs_old():

    rear_left = "data-files/data_adversary_rear_left.csv"
    rear_right = "data-files/data_adversary_rear_right.csv"
    front_long = "data-files/data_front_ultrawide_long.csv"
    front_normal = "data-files/data_front_ultrawide.csv"
    rear_normal = "data-files/data_rear_ultrawide.csv"

    trl = 1689369626
    trr = 1689369483
    tnorm = 1689368390

    #time for rear left: 1689369626, 120 seconds long
    #time for normal ultrawide: 1689368390, 238 seconds long
    #time for rear right: 1689369483, 117 seconds long
    length = 120

    data_normal = []
    data_right = []
    data_left = []
    X = []

    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, layout="constrained")
    fig2, ((bx)) = plt.subplots(1, 1, layout="constrained")

    ax = {5: ax1, 15: ax2, 25: ax3, 35: ax4}

    for i in [5, 15, 25, 35]:
        data_normal = []
        data_right = []
        data_left = []
        X = []
        for res in range(1,8):
            n = length//i
            
            rear_left_block = block_encoding(rear_left, 1689369626, block_size=i, num_blocks=n, time_gap = 0, reverse=True, tres=res, bits_to_drop=1, zone_name='data-zone/data_adversary_rear_left_zone.csv')
            rear_right_block = block_encoding(rear_right, 1689369483, block_size=i, num_blocks=n, time_gap = 0, reverse=True, tres=res, bits_to_drop=1, zone_name='data-zone/data_adversary_rear_right_zone.csv')
            front_left_block = block_encoding(front_long, 1689369626, block_size=i, num_blocks=n, time_gap = 0, tres=res, bits_to_drop=1, zone_name='data-zone/data_front_ultrawide_long_zone.csv')
            front_right_block = block_encoding(front_long, 1689369483, block_size=i, num_blocks=n, time_gap = 0, tres=res, bits_to_drop=1, zone_name='data-zone/data_front_ultrawide_long_zone.csv')
            front_block = block_encoding(front_normal, 1689368390, block_size=i, num_blocks=n, time_gap = 0, tres=res, bits_to_drop=1, zone_name='data-zone/data_front_ultrawide_zone.csv')
            rear_block = block_encoding(rear_normal, 1689368390, block_size=i, num_blocks=n, time_gap = 0, reverse=True, tres=res, bits_to_drop=1, zone_name='data-zone/data_rear_ultrawide_zone.csv')

            data_normal.append(validate_block(front_block, rear_block))
            data_right.append(validate_block(rear_right_block, front_right_block))
            data_left.append(validate_block(rear_left_block, front_left_block))
            X.append(res)

        #print(data_normal)
        #print(data_right)
        #print(data_left)
        #print(X)

        ax[i].set_ylim([0.65, 0.9])
        ax[i].set_xlim([0.5, 7.5])
        

        ax[i].scatter(X, data_normal)
        ax[i].scatter(X, data_right)
        ax[i].scatter(X, data_left)

        ax[i].legend(['Normal', 'Adversary Right','Adversary Left'], prop={'size': 6})

        #plot the lines of best fit
        a,b = best_fit(X, data_normal)
        yfit = [a + b * xi for xi in X]
        ax[i].plot(X, yfit)
        a,b = best_fit(X, data_right)
        yfit = [a + b * xi for xi in X]
        ax[i].plot(X, yfit)
        a,b = best_fit(X, data_left)
        yfit = [a + b * xi for xi in X]
        ax[i].plot(X, yfit)

        #show the image
        ax[i].set_xlabel("Time Resolution (bits)")
        ax[i].set_ylabel("Percent Similarity")
        ax[i].set_title(f"Block Size of {i}")

    fig1.suptitle("Similarity Percentage with respect to Time Resolution at Various Block Sizes")
    fig1.savefig('figures/Time-and-Block-Size.png', dpi = 300, bbox_inches='tight')

    data_normal = []
    data_right = []
    data_left = []
    X = []
    for i in range(5, 35):
        n = length//i
        
        rear_left_block = block_encoding(rear_left, 1689369626, block_size=i, num_blocks=n, time_gap = 0, reverse=True, zone_name='data-zone/data_adversary_rear_left_zone.csv')
        rear_right_block = block_encoding(rear_right, 1689369483, block_size=i, num_blocks=n, time_gap = 0, reverse=True, zone_name='data-zone/data_adversary_rear_right_zone.csv')
        front_left_block = block_encoding(front_long, 1689369626, block_size=i, num_blocks=n, time_gap = 0, zone_name='data-zone/data_front_ultrawide_long_zone.csv')
        front_right_block = block_encoding(front_long, 1689369483, block_size=i, num_blocks=n, time_gap = 0, zone_name='data-zone/data_front_ultrawide_long_zone.csv')
        front_block = block_encoding(front_normal, 1689368390, block_size=i, num_blocks=n, time_gap = 0, zone_name='data-zone/data_front_ultrawide_zone.csv')
        rear_block = block_encoding(rear_normal, 1689368390, block_size=i, num_blocks=n, time_gap = 0, reverse=True, zone_name='data-zone/data_rear_ultrawide_zone.csv')

        data_normal.append(validate_block(front_block, rear_block))
        data_right.append(validate_block(rear_right_block, front_right_block))
        data_left.append(validate_block(rear_left_block, front_left_block))
        X.append(i)

    #print(data_normal)
    #print(data_right)
    #print(data_left)
    #print(X)

    bx.set_ylim([0.6, 0.9])
    bx.set_xlim([4, 36])
    

    bx.scatter(X, data_normal)
    bx.scatter(X, data_right)
    bx.scatter(X, data_left)

    bx.legend(['Normal', 'Adversary Right','Adversary Left'])

    #plot the lines of best fit
    a,b = best_fit(X, data_normal)
    yfit = [a + b * xi for xi in X]
    bx.plot(X, yfit)
    a,b = best_fit(X, data_right)
    yfit = [a + b * xi for xi in X]
    bx.plot(X, yfit)
    a,b = best_fit(X, data_left)
    yfit = [a + b * xi for xi in X]
    bx.plot(X, yfit)

    #show the image
    bx.set_xlabel("Block Length (seconds)")
    bx.set_ylabel("Percent Similarity")

    fig2.suptitle("Similarity Percentage with respect to Block Length")
    fig2.savefig('figures/Similarity-Block-Length.png', dpi = 300, bbox_inches='tight')
    
    
    
    plt.show()

def graphs_normal_time_resolution_block_size():

    front_2 = "data-files/1690392480_platoon2_front.csv"
    rear_2 = "data-files/1690392483_platoon2_rear.csv"
    front_3 = "data-files/1690393032_platoon3_front.csv"
    rear_3 = "data-files/1690393045_platoon3_rear.csv"

    #front_2_zone = "data-zone/1690392480-platoon2-front_zone.csv"
    #rear_2_zone = "data-zone/1690392483_platoon2_rear_zone.csv"
    #front_3_zone = "data-zone/1690393032_platoon3_front_zone.csv"
    #rear_3_zone = "data-zone/1690393045_platoon3_rear_zone.csv"

    front_2_zone = None
    rear_2_zone = None
    front_3_zone = None
    rear_3_zone = None


    t2 = 1690392510
    t3 = 1690393045
    length = 200

    data = []
    X = []

    fig, ((ax)) = plt.subplots(1, 1, layout="constrained")


    for i in range(5, 36, 10):
        data = []
        X = []
        for res in range(2,8):
            n = length//i
            
            #front_2_block = block_encoding(front_2, t2, block_size=i, num_blocks=n, time_gap=0, reverse=True, tres=res, bits_to_drop=1, zone_name=front_2_zone)
            #rear_2_block = block_encoding(rear_2, t2, block_size=i, num_blocks=n, time_gap=0, tres=res, bits_to_drop=1, zone_name=rear_2_zone)
            front_3_block = block_encoding(front_3, t3, block_size=i, num_blocks=n, time_gap=0, tres=res, bits_to_drop=1, zone_name=front_3_zone)
            rear_3_block = block_encoding(rear_3, t3, block_size=i, num_blocks=n, time_gap=0, reverse=True, tres=res, bits_to_drop=1, zone_name=rear_3_zone)
            
            #data.append(validate_block(front_2_block, rear_2_block))  
            #X.append(res)
            data.append(validate_block(front_3_block, rear_3_block))  
            X.append(res)

            
        ax.scatter(X, data, label=f"Block Size: {i}")
        a, b = best_fit(X, data)
        yfit = [a + b * xi for xi in X]
        ax.plot(X, yfit)

    ax.legend()

        
    ax.set_xlabel("Time Resolution (bits)")
    ax.set_ylabel("Percent Similarity")

    fig.suptitle("Percent Similarities with respect to Time Resolution at Various Block Sizes")
    fig.savefig('figures/Similarity-Block-Length-Platoon-Platoon-3.png', dpi = 300, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    #graphs_old()
    #graphs_normal_time_resolution_block_size()
    #front_normal = "data-files/data_front_ultrawide.csv"
    #rear_normal = "data-files/data_rear_ultrawide.csv"
    #front_zone = 'data-zone/data_front_ultrawide_zone.csv'
    #rear_zone = 'data-zone/data_rear_ultrawide_zone.csv'

    #front_zone = None
    #rear_zone = None

    #front = block_encoding(front_normal, 1689368390, block_size=20, num_blocks=10, time_gap = 0, reverse=True, zone_name = front_zone)
    #rear = block_encoding(rear_normal, 1689368390, block_size=20, num_blocks=10, time_gap = 0, zone_name = rear_zone)

    #print(validate_block(front, rear))

    front_2 = "data-files/1690392480_platoon2_front.csv"
    rear_2 = "data-files/1690392483_platoon2_rear.csv"
    front_3 = "data-files/1690393032_platoon3_front.csv"
    rear_3 = "data-files/1690393045_platoon3_rear.csv"

    with open('filtered_platoon_2.txt', 'w') as f:
        filter_timestamps(front_2, rear_2, 3, f)
    with open('filtered_platoon_3.txt', 'w') as f:
        filter_timestamps(front_3, rear_3, 3, f)