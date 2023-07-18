# Tracking and Counting
# Being used by CAT Vehicle Group 2
# Altered by Adhith, John, and Max
# Last updated 18 July 2023

import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import tqdm

#custom classes
from tracker import*
from counter import Counter, DataWriter

model=YOLO('yolov8s.pt') # Change model if needed

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Describe name of video being used
#cap=cv2.VideoCapture('long_range_b.mp4')
# REAR FOV
#cap=cv2.VideoCapture('realistic_FOV_T_60_edited.mp4')
# FRONT FOV
cap=cv2.VideoCapture('realistic_FOV_T_60_edited.mp4')

# resolution of the video capture this is also used to trim each frame later on
# I am not entirely sure why it uses this wacky resolution
# We should consider going back to a 16:9 aspect ratio because this appears to squish the footage,
# which could negatively impact detection performance
size = (1920, 1080)
vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
framerate = int(cap.get(cv2.CAP_PROP_FPS))

# Below VideoWriter object will create a frame of above defined
# The output is stored in 'filename.avi' file.
# you have to add this to your .gitignore file (add the line below)
# output.*
out = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

#read the classes yolov8 identifies
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

#establish a counter variable for the number of frames that have passed in the video
count=0
#create a new tracker opbject - this keeps track of which objects are actively crossing the lines
tracker=Tracker()

#create two new counters, one for the left and one for the right
#low key I'm not convinced this is any more readable than the way it was stored before
#but classes are cool, and it lets us do other cool things like diagonals and the code can be 
#cleaner, so I guess that's cool

#the u stands for upper, and point 1 should be on the left and point 2 on the right (though I don't think it actually matters)

#-------------------------------------------------------------------------------------------------
## START
## For long_range_b.mp4 (1020, 500)
#cl = Counter(uy1 = 323, uy2 = 333, ux1=184, ux2=410,
#            ly1 = 333, ly2 = 343, lx1=10, lx2=370)
#cr = Counter(uy1 = 333, uy2 = 323, ux1=435, ux2=814,
#            ly1 = 343, ly2 = 333, lx1=443, lx2=1007)
## END
#-------------------------------------------------------------------------------------------------
## START
## REAR FOV
## For realistic_FOV_T_60_edited.mp4

# (1020, 500)
#cl = Counter(uy1 = 317, uy2 = 327, ux1=135, ux2=487,
#            ly1 = 332, ly2 = 342, lx1=14, lx2=462, offx=4, offuy=5, offly=5)
#cr = Counter(uy1 = 333, uy2 = 323, ux1=561, ux2=926,
#            ly1 = 343, ly2 = 333, lx1=590, lx2=1018, offx=4, offuy=5, offly=5)

# (1920, 1080)
cl = Counter(uy1 = 685, uy2 = 706, ux1=254, ux2=917,
            ly1 = 717, ly2 = 738, lx1=26, lx2=869, offx=8, offuy=11, offly=11)
cr = Counter(uy1 = 719, uy2 = 698, ux1=1056, ux2=1743,
            ly1 = 741, ly2 = 719, lx1=1110, lx2=1916, offx=8, offuy=11, offly=11)

## END
#-------------------------------------------------------------------------------------------------
## START
## FRONT FOV
## For realistic_FOV_J_30_edited.mp4
#cl = Counter(uy1 = 500, uy2 = 600, ux1=100, ux2=800,
#            ly1 = 540, ly2 = 640, lx1=80, lx2=780, offx=4, offuy=5, offly=5)
#cr = Counter(uy1 = 600, uy2 = 500, ux1=1120, ux2=1820,
#            ly1 = 640, ly2 = 540, lx1=1130, lx2=1830, offx=4, offuy=5, offly=5)
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

#create a new instance of the datawriter class to record the data we gather
data_writer = DataWriter("data_ultrawide_rear.csv")

#start time in GMT unix time
start_time = time.time()

#loop through the video
for _ in tqdm.tqdm(range(vid_length)):    
    ret,frame = cap.read()

    #this code exists to limit the number of frames the code actually looks at
    #counts the number of frames that have passed
    count += 1
    #this limits the effective framerate of what we are looking at to 10, which seems to be sufficient
    #automatically gets the framerate of the video being used with opencv
    #TODO: actually implement the "automatic" part of this
    if count % (framerate/10) != 0:
        continue
    # For Tristan's recorded data it is at 60 FPS for all videos so we need to look at every 6 frames
    # For My recorded data, all but one is at 30 FPS so we need to look at every 3 frames and I can let you know which one is which
    # We can also pre-process the videos to make them 30 FPS each to ensure similar amounts of precision are being used


    #resize the frame
    frame=cv2.resize(frame,size)
   
    #run YOLOv8 on the frame
    results=model.predict(frame, verbose=False)
    #print(results)
    #get the data from the classification
    a=results[0].boxes.data
    #why the f**k does it use a two letter variable without at least an explanation I hate this
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
        object_class=class_list[d]

        # Define what classes of objects to look for and record coordinates
        # Add more if needed (stick to streetside objects)
        relevant_classes = ['car', 'truck', 'bus', 'bicycle', 'motorcycle']

        if object_class in relevant_classes:
            list.append([x1,y1,x2,y2])
    
    # gets all of the bounding boxes we are tracking    
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        # gets the coordinates of the relevant bounding box
        x3,y3,x4,y4,id=bbox
        # Gets the midpoint of the x-axis of the bounding box
        center_x=int(x3+x4)//2
        # Uncomment line below for center point of bounding box
        mid_center_y=int(y3+y4)//2
        # Uncomment line below for center point of bottom y-axis of bounding box
        lower_center_y=y4
        # Uncomment line below for center point of upper y-axis of bounding box
        upper_center_y=y3
        
        # LEFT SIDE
        # Counting vehicles going "inLeft" to frame
        #check if the object we are currently checking is within the offset of the upper line
        if cl.within_upper_line(center_x, lower_center_y):
            vh_in_left[id] = lower_center_y
        #check if the object was at one point within the offsets of the upper line
        if id in vh_in_left:
            #check if that object is now within the offset of the lower line
            if cl.within_lower_line(center_x, lower_center_y):
                cv2.circle(frame,(center_x,lower_center_y),4,(0,0,255),-1) # Draw circle
                cv2.putText(frame,str(id),(center_x,lower_center_y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2) # Give and Print ID
                if id not in counter_in_left:
                    counter_in_left.append(id)
                    #We know a new event occurred, so we now update the data writer with that information
                    data_writer.add_event('in left', start_time + count/framerate)

                    
        # Counting vehicles going "outLeft" of frame
        if cl.within_lower_line(center_x, mid_center_y):
            vh_out_left[id] = mid_center_y
        if id in vh_out_left:
            if cl.within_upper_line(center_x, mid_center_y):
                cv2.circle(frame,(center_x,mid_center_y),4,(0,0,255),-1) # Draw circle
                cv2.putText(frame,str(id),(center_x,mid_center_y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2) # Give and Print ID
                if id not in counter_out_left:
                    counter_out_left.append(id)
                    #We know a new event occurred, so we now update the data writer with that information
                    data_writer.add_event('out left', start_time + count/framerate)
                    
                    
        # RIGHT SIDE
        # Counting vehicles going "inRight" to frame
        #check if the current vehicle is within the offset of the upper line
        if cr.within_upper_line(center_x, lower_center_y):
            vh_in_right[id] = lower_center_y
        #check if the id was at one point within the offset of the upper line
        if id in vh_in_right:
            #now check if the id is within the offset of the lower line
            if cr.within_lower_line(center_x, lower_center_y):
                cv2.circle(frame,(center_x,lower_center_y),4,(255,0,0),-1) # Draw circle
                cv2.putText(frame,str(id),(center_x,lower_center_y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2) # Give and Print ID
                if id not in counter_in_right:
                    counter_in_right.append(id)
                    #We know a new event occurred, so we now update the data writer with that information
                    data_writer.add_event('in right', start_time + count/framerate)
                    
        # Counting vehicles going "outRight" of frame
        if cr.within_lower_line(center_x, mid_center_y):
            vh_out_right[id] = mid_center_y
        if id in vh_out_right:
            if cr.within_upper_line(center_x, mid_center_y):
                cv2.circle(frame,(center_x,mid_center_y),4,(255,0,0),-1) # Draw circle
                cv2.putText(frame,str(id),(center_x,mid_center_y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2) # Give and Print ID
                if id not in counter_out_right:
                    counter_out_right.append(id)
                    #We know a new event occurred, so we now update the data writer with that information
                    data_writer.add_event('out right', start_time + count/framerate)
        
        
    #this part annotates the lines on the frame
    cl.draw(frame=frame, label_upper='Upper Left', label_lower='Lower Left', color=(0,0,255))
    cr.draw(frame=frame, label_upper='Upper Right', label_lower='Lower Right', color=(255,0,0))
    
    # General Code
    #gets the number of cars in and out by counting the length of the arrays
    cin_Left = (len(counter_in_left)) # counter for in left
    cout_Left = (len(counter_out_left)) # counter for out left
    cin_Right = (len(counter_in_right)) # counter for in right
    cout_Right = (len(counter_out_right)) # counter for out right
    
    #displays the counts of cars in and out using openCV
    cv2.putText(frame,('inLeft: ')+str(cin_Left),(40,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    cv2.putText(frame,('outLeft: ')+str(cout_Left),(40,70),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    cv2.putText(frame,('inRight: ')+str(cin_Right),(840,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    cv2.putText(frame,('outRight: ')+str(cout_Right),(840,70),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    
    #shows the images and writes it to the video writer
    out.write(frame)

    cv2.imshow("Detecting Vehicles", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

#close the display window
cap.release()
out.release()
cv2.destroyAllWindows()

data_writer.store_data()