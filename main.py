# Tracking and Counting
# Being used by CAT Vehicle Group 2
# Altered by Adhith, John, and Max
# Last updated 17 July 2023

import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

#custom classes
from tracker import*
from counter import Counter


#idk why I'm actually importing this, we could just use a placeholder. Would feel weird though
import time

import tqdm

model=YOLO('yolov8l.pt') # Change model if needed



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Describe name of video being used
cap=cv2.VideoCapture('long_range_b.mp4')
# REAR FOV
#cap=cv2.VideoCapture('realistic_FOV_T_60_edited.mp4')
# FRONT FOV
#cap=cv2.VideoCapture('realistic_FOV_J_30_edited.mp4')

#get the resolution of the video capture - because this is trimmed later on, I got lazy and hard coded it 
size = (1020, 500)
vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

   
# Below VideoWriter object will create a frame of above defined
# The output is stored in 'filename.avi' file.
# you have to add this to your .gitignore file (add the line below)
# output.*
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4',fourcc, 10.0, size)



#read the classes yolov8 identifies
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 


#establish a counter variable for the number of frames that have passed in the video
count=0
#create a new tracker opbject - this keeps track of which objects are actively crossing the lines
tracker=Tracker()

#create two new counters, one for the left and one for the right
cl = Counter(uy1 = 323, uy2 = 333, ux1=184, ux2=410,
            ly1 = 333, ly2 = 343, lx1=10, lx2=370)

cr = Counter(uy1 = 333, uy2 = 323, ux1=435, ux2=814,
            ly1 = 343, ly2 = 333, lx1=443, lx2=1007)

#-------------------------------------------------------------------------------------------------
## START
## For long_range_b.mp4
coord_y1=323 # Y-Coordinates for upper Line
coord_y2=333 # Y-Coordinates for lower Line
x1L=184 # Left-Side of X-Coordinates of upper Line
x1R=814 # Right-Side of X-Coordinates of upper Line
x2L=10 # Left-Side of X-Coordinates of lower Line
x2R=1007 # Right-Side of X-Coordinates of lower Line


# For Left Side
x1R_cutoff=410
x2R_cutoff=370

# For Right Side
x1L_cutoff=435
x2L_cutoff=443

offset1=4 # Offset for upper Line
offset2=6 # Offset for lower Line
offset3=4 # Offset for X-Axis
## END
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
## START
## REAR FOV
## For realistic_FOV_T_60_edited.mp4
#coord_y1=317 # Y-Coordinates for upper Line
#coord_y2=332 # Y-Coordinates for lower Line
#x1L=135 # Left-Side of X-Coordinates of upper Line
#x1R=926 # Right-Side of X-Coordinates of upper Line
#x2L=14 # Left-Side of X-Coordinates of lower Line
#x2R=1018 # Right-Side of X-Coordinates of lower Line


# For Left Side
#x1R_cutoff=487
#x2R_cutoff=462

# For Right Side
#x1L_cutoff=561
#x2L_cutoff=590

#offset1=5 # Offset for upper Line
#offset2=5 # Offset for lower Line
#offset3=4 # Offset for X-Axis
## END
#-------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------
# Have not messed with these parameters yet
## START
## FRONT FOV
## For realistic_FOV_J_30_edited.mp4
#coord_y1=317 # Y-Coordinates for upper Line
#coord_y2=332 # Y-Coordinates for lower Line
#x1L=135 # Left-Side of X-Coordinates of upper Line
#x1R=926 # Right-Side of X-Coordinates of upper Line
#x2L=14 # Left-Side of X-Coordinates of lower Line
#x2R=1018 # Right-Side of X-Coordinates of lower Line


# For Left Side
#x1R_cutoff=487
#x2R_cutoff=462

# For Right Side
#x1L_cutoff=561
#x2L_cutoff=590

#offset1=5 # Offset for upper Line
#offset2=5 # Offset for lower Line
#offset3=4 # Offset for X-Axis
## END
#-------------------------------------------------------------------------------------------------
# General Code
vh_in_left = {} # Holds IDs of cars going into frame on Left for tracking
vh_out_left = {} # Holds IDs of cars going out of frame on Left for tracking

vh_in_right = {} # Holds IDs of cars going into frame on Right for tracking
vh_out_right = {} # Holds IDs of cars going out of frame on Right for tracking

counter_in_left = [] # List of IDs of cars that have gone into frame on Left
counter_out_left = [] # List of IDs of cars that have come out of frame on Left

counter_in_right = [] # List of IDs of cars that have gone into frame on Right
counter_out_right = [] # List of IDs of cars that have gone out of frame on Right

#create a list to hold the events we will use to generate data
#end goal is to get this so we can store it in a csv file to use later to have more flexibility when we experiment with our data detection
data = []
#start time in GMT unix time
start_time = time.time()

#infinitely loop through the video
for _ in tqdm.tqdm(range(vid_length)):    
    ret,frame = cap.read()

    #this code exists to limit the number of frames the code actually looks at
    #counts the number of frames that have passed
    count += 1
    #if the number of frames isn't a multiple of three, skip to the next frame
    if count % 6 != 0:
        continue
    # For Tristan's recorded data it is at 60 FPS for all videos so we need to look at every 6 frames
    # For My recorded data, all but one is at 30 FPS so we need to look at every 3 frames and I can let you know which one is which
    # We can also pre-process the videos to make them 30 FPS each to ensure similar amounts of precision are being used
    # This is my bad, we were kinda in a rush so I wasn't double checking the FPS being recorded in
    # Regardless 30 FPS should be enough for the precision that we are looking for
    

    #resize the frame
    frame=cv2.resize(frame,(1020,500))
   

    #run YOLOv8 on the frame
    results=model.predict(frame, verbose=False)
    #print(results)
    #get the data from the classification
    a=results[0].boxes.data
    #create a dataframe of the results
    #why the f**k does it use a two letter variable I hate this
    px=pd.DataFrame(a).astype("float")
    #print(px)
    list=[]
             
    # Object Tracker

    #this loop selects all of the objects that are classes we care about, and adds the coordinates to the object tracking list
    for index,row in px.iterrows():
        
        #get the coordinates of the bounding box
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])

        #I don't know what this is for and one letter variables certainly don't help --___--
        d=int(row[5])
        #get the class of the object
        c=class_list[d]

        # Define what classes of objects to look for and record coordinates
        # Add more if needed (stick to streetside objects)
        relevant_classes = ['car', 'truck', 'bus', 'bicycle', 'motorcycle']

        if c in relevant_classes:
            list.append([x1,y1,x2,y2])
    
    # gets all of the bounding boxes we are tracking    
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        # gets the coordinates of the relevant bounding box
        x3,y3,x4,y4,id=bbox
        # Gets the midpoint of the x-axis of the bounding box
        center_x=int(x3+x4)//2
        # Uncomment line below for center point of bounding box
        #center_y=int(y3+y4)//2
        # Uncomment line below for center point of bottom y-axis of bounding box
        center_y=y4
        
        # LEFT SIDE
        # Counting vehicles going "inLeft" to frame
        #check if the object we are currently checking is within the offset of the upper line
        if cl.within_upper_line(center_x, center_y):
            vh_in_left[id] = center_y
        #check if the object was at one point within the offsets of the upper line
        if id in vh_in_left:
            #check if that object is now within the offset of the lower line
            if cl.within_lower_line(center_x, center_y):
                cv2.circle(frame,(center_x,center_y),4,(0,0,255),-1) # Draw circle
                cv2.putText(frame,str(id),(center_x,center_y),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2) # Give and Print ID
                if id not in counter_in_left:
                    counter_in_left.append(id)
                    #this is where we know a new event occured, because the counter was just incremented
                    #first we get the time the event occured. count/30 is the number of seconds since the video started
                    event_time = start_time + count/30
                    event_time = int(event_time*100) / 100
                    #now append that to data
                    data.append((event_time, 'in left'))

                    
        # Counting vehicles going "outLeft" of frame
        if cl.within_lower_line(center_x, center_y):
            vh_out_left[id] = center_y
        if id in vh_out_left:
            if cl.within_upper_line(center_x, center_y):
                cv2.circle(frame,(center_x,center_y),4,(0,0,255),-1) # Draw circle
                cv2.putText(frame,str(id),(center_x,center_y),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2) # Give and Print ID
                if id not in counter_out_left:
                    counter_out_left.append(id)
                    #we know a new event occured here, because this is where the counter is incremented
                    #first we get the time the event occured. count/30 is the number of seconds since the video started
                    event_time = start_time + count/30
                    event_time = int(event_time*100)/100
                    #now append that to data
                    data.append((event_time, 'out left'))
                    
                    
        # RIGHT SIDE
        # Counting vehicles going "inRight" to frame
        #check if the current vehicle is within the offset of the upper line
        if cr.within_upper_line(center_x, center_y):
            vh_in_right[id] = center_y
        #check if the id was at one point within the offset of the upper line
        if id in vh_in_right:
            #now check if the id is within the offset of the lower line
            if cr.within_lower_line(center_x, center_y):
                cv2.circle(frame,(center_x,center_y),4,(255,0,0),-1) # Draw circle
                cv2.putText(frame,str(id),(center_x,center_y),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2) # Give and Print ID
                if id not in counter_in_right:
                    counter_in_right.append(id)
                    #this is where we know a new event occured, because the counter was just incremented
                    #first we get the time the event occured. count/30 is the number of seconds since the video started
                    event_time = start_time + count/30
                    event_time = int(event_time*100) / 100
                    #now append that to data
                    data.append((event_time, 'in right'))
                    
        # Counting vehicles going "outRight" of frame
        if cr.within_lower_line(center_x, center_y):
            vh_out_right[id] = center_y
        if id in vh_out_right:
            if cr.within_upper_line(center_x, center_y):
                cv2.circle(frame,(center_x,center_y),4,(255,0,0),-1) # Draw circle
                cv2.putText(frame,str(id),(center_x,center_y),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2) # Give and Print ID
                if id not in counter_out_right:
                    counter_out_right.append(id)
                    #we know a new event occured here, because this is where the counter is incremented
                    #first we get the time the event occured. count/30 is the number of seconds since the video started
                    event_time = start_time + count/30
                    event_time = int(event_time*100)/100
                    #now append that to data
                    data.append((event_time, 'out right'))
        
        
    #For long_range_b.mp4
    #this part just annotates the frame
    cl.draw(frame=frame, label_upper='Upper Left', label_lower='Lower Left', color=(0,0,255), labels=False)
    cr.draw(frame=frame, label_upper='Upper Right', label_lower='Lower Right', color=(0,255,0), labels=False)
    
    # General Code
    #gets the number of cars in and out by counting the length of the arrays
    cin_Left = (len(counter_in_left)) # counter for in left
    cout_Left = (len(counter_out_left)) # counter for out left
    cin_Right = (len(counter_in_right)) # counter for in right
    cout_Right = (len(counter_out_right)) # counter for out right
    
    #displays the counts of cars in and out using openCV
    cv2.putText(frame,('inLeft: ')+str(cin_Left),(40,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
    cv2.putText(frame,('outLeft: ')+str(cout_Left),(40,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
    cv2.putText(frame,('inRight: ')+str(cin_Right),(840,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
    cv2.putText(frame,('outRight: ')+str(cout_Right),(840,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
    
    #shows the images and writes it to the video writer
    out.write(frame)

    cv2.imshow("Detecting Vehicles", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

#close the display window
cap.release()
out.release()
cv2.destroyAllWindows()

# print data to terminal
print(data)
# save the data to data.csv
#np.savetxt('data.csv', [row for row in data], delimiter=',', fmt='%s', header="time,event", comment="")
np.savetxt('data.csv', [row for row in data], delimiter=',', fmt='%s', header="time,event")