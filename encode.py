import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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

def graphs_old():

    rear_left = "data_adversary_rear_left"
    rear_right = "data_adversary_rear_right"
    front_long = "data_front_ultrawide_long"
    front_normal = "data_front_ultrawide"
    rear_normal = "data_rear_ultrawide"

    platoon_2_front = "1690392480_platoon2_front"
    platoon_2_rear = "1690392483_platoon2_rear"
    platoon_3_front = "1690393032_platoon3_front"
    platoon_3_rear = "1690393045_platoon3_rear"

    adleft_rear_1 = "1690393465_adleft_rear_1"
    adleft_front_1 = "1690393466_adleft_front_1"
    adright_rear_1 = "1690394944_adright_rear_1"
    adright_front_1 = "1690394942_adright_front_1"

    trl = 1689369626
    trr = 1689369483
    tnorm = 1689368390

    pairs = [(rear_left, front_long, 1689369626, 'l', 120), (rear_right, front_long, 1689369483, 'r', 117),
             (front_normal, rear_normal, 1689368390, 'n', 238), (platoon_2_front, platoon_2_rear, 1690392500, 'n', 240),
             (platoon_3_front, platoon_3_rear, 1690393075, 'n', 275), (adleft_rear_1, adleft_front_1, 1690393470, 'l', 210),
             (adright_rear_1, adright_front_1, 1690394950, 'r', 280)]

    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, layout="constrained")
    #fig2, ((bx)) = plt.subplots(1, 1, layout="constrained")

    ax = {5: ax1, 10: ax2, 15: ax3, 25: ax4}


    for i in [5, 10, 15, 25]:
        data_normal = []
        X_normal = []
        data_right = []
        X_right = []
        data_left = []
        X_left = []
        data = {'l': data_left, 'r': data_right, 'n': data_normal}
        X = {'l': X_left, 'r': X_right, 'n': X_normal}

        for res in range(1,8):
            for pair in pairs:
                data_1 = 'data-files/' + pair[0] + '.csv'
                data_2 = 'data-files/' + pair[1] + '.csv'
                zone_1 = 'data-zone/' + pair[0] + '_zone.csv'
                zone_2 = 'data-zone/' + pair[1] + '_zone.csv'

                length = pair[-1]
                t = pair[2]
            
                n = length//i
            
                block_1 = block_encoding(data_1, t, block_size=i, num_blocks=n, time_gap=0, reverse=True,
                                         tres=res, bits_to_drop=1, zone_name=zone_1)
                block_2 = block_encoding(data_2, t, block_size=i, num_blocks=n, time_gap=0,
                                         tres=res, bits_to_drop=1, zone_name=zone_2)
            
                data[pair[3]].append(validate_block(block_1, block_2))
                X[pair[3]].append(res)
                
        ax[i].set_ylim([0.55, 0.9])
        ax[i].set_xlim([0.5, 7.5])
        

        ax[i].scatter(X_normal, data_normal)
        ax[i].scatter(X_right, data_right)
        ax[i].scatter(X_left, data_left)

        ax[i].legend(['Normal', 'Adversary Right','Adversary Left'], prop={'size': 6})

        #plot the lines of best fit
        a,b = best_fit(X_normal, data_normal)
        yfit = [a + b * xi for xi in X_normal]
        ax[i].plot(X_normal, yfit)
        a,b = best_fit(X_right, data_right)
        yfit = [a + b * xi for xi in X_right]
        ax[i].plot(X_right, yfit)
        a,b = best_fit(X_left, data_left)
        yfit = [a + b * xi for xi in X_left]
        ax[i].plot(X_left, yfit)

        #show the image
        ax[i].set_xlabel("Number of Time Bits Used")
        ax[i].set_ylabel("Percent Similarity")
        ax[i].set_title(f"Block Size of {i}")

    
    fig1.suptitle("Zone Similarity versus Time Resolution and Block Size")
    fig1.savefig('figures/Time-Block-Only-Zone.png', dpi = 300, bbox_inches='tight')
    plt.show()

def graph_block_size():

    
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig1, ((bx)) = plt.subplots(1, 1, layout="constrained")


    rear_left = "data_adversary_rear_left"
    rear_right = "data_adversary_rear_right"
    front_long = "data_front_ultrawide_long"
    front_normal = "data_front_ultrawide"
    rear_normal = "data_rear_ultrawide"

    platoon_2_front = "1690392480_platoon2_front"
    platoon_2_rear = "1690392483_platoon2_rear"
    platoon_3_front = "1690393032_platoon3_front"
    platoon_3_rear = "1690393045_platoon3_rear"

    adleft_rear_1 = "1690393465_adleft_rear_1"
    adleft_front_1 = "1690393466_adleft_front_1"
    adright_rear_1 = "1690394944_adright_rear_1"
    adright_front_1 = "1690394942_adright_front_1"

    pairs = [(rear_left, front_long, 1689369626, 'l', 120), (rear_right, front_long, 1689369483, 'r', 117),
             (front_normal, rear_normal, 1689368390, 'n', 238), (platoon_2_front, platoon_2_rear, 1690392500, 'n', 240),
             (platoon_3_front, platoon_3_rear, 1690393075, 'n', 275), (adleft_rear_1, adleft_front_1, 1690393470, 'l', 210),
             (adright_rear_1, adright_front_1, 1690394950, 'r', 280)]
    
    data_normal = []
    data_right = []
    data_left = []
    data = {'l': data_left, 'r': data_right, 'n': data_normal}
    X_normal = []
    X_right = []
    X_left = []
    X = {'l': X_left, 'r': X_right, 'n': X_normal}
    
    for i in range(5, 36, 2):
        
        temp_data_norm = [0]
        num_norm = [0]
        temp_data_right = [0]
        num_right = [0]
        temp_data_left = [0]
        num_left = [0]
        data = {'l': temp_data_left, 'r': temp_data_right, 'n': temp_data_norm}
        X = {'l': num_left, 'r': num_right, 'n': num_norm}
        
        for pair in pairs:
            data_1 = 'data-files/' + pair[0] + '.csv'
            data_2 = 'data-files/' + pair[1] + '.csv'
            zone_1 = 'data-zone/' + pair[0] + '_zone.csv'
            zone_2 = 'data-zone/' + pair[1] + '_zone.csv'

            length = pair[-1]
            t = pair[2]
        
            n = length//i
        
            block_1 = block_encoding(data_1, t, block_size=i, num_blocks=n, time_gap=0, reverse=True,
                                        tres=4, bits_to_drop=1, zone_name=zone_1)
            block_2 = block_encoding(data_2, t, block_size=i, num_blocks=n, time_gap=0,
                                        tres=4, bits_to_drop=1, zone_name=zone_2)
        
            #data[pair[3]].append(validate_block(block_1, block_2))
            #X[pair[3]].append(i)
            data[pair[3]][0] += validate_block(block_1, block_2)
            X[pair[3]][0] += 1

        
        
        data_normal.append(temp_data_norm[0]/num_norm[0])
        X_normal.append(i)
        data_left.append(temp_data_left[0]/num_left[0])
        X_left.append(i)
        data_right.append(temp_data_right[0]/num_right[0])
        X_right.append(i)
        
    

    bx.scatter(X_normal, data_normal)
    bx.scatter(X_right, data_right)
    bx.scatter(X_left, data_left)

    bx.legend(['Normal', 'Adversary Right','Adversary Left'])
    #bx.legend(['Normal', 'Adversary'])
    
    #plot the lines of best fit
    a,b = best_fit(X_normal, data_normal)
    yfit = [a + b * xi for xi in X_normal]
    bx.plot(X_normal, yfit)
    a,b = best_fit(X_right, data_right)
    yfit = [a + b * xi for xi in X_right]
    bx.plot(X_right, yfit)
    a,b = best_fit(X_left, data_left)
    yfit = [a + b * xi for xi in X_left]
    bx.plot(X_left, yfit)
    

    
    bx.set_xlabel("Block Length (seconds)")
    bx.set_ylabel("Percent Similarity")

    fig1.savefig('figures/Block-Length-Both-Scatter-Averaged.png', dpi = 300, bbox_inches='tight')
    
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
    graph_block_size()
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

    #front_2 = "data-files/1690392480_platoon2_front.csv"
    #rear_2 = "data-files/1690392483_platoon2_rear.csv"
    #front_3 = "data-files/1690393032_platoon3_front.csv"
    #rear_3 = "data-files/1690393045_platoon3_rear.csv"

    #with open('filtered_platoon_2.txt', 'w') as f:
    #    filter_timestamps(front_2, rear_2, 3, f)
    #with open('filtered_platoon_3.txt', 'w') as f:
    #    filter_timestamps(front_3, rear_3, 3, f)