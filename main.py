# Tracking and Counting
# Being used by CAT Vehicle Group 2
# Altered by Adhith, John, and Max
# Last updated 18 July 2023

import cv2
import pandas as pd
from ultralytics import YOLO
import time
import tqdm
import json

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

#Code to parse the configurations from the configurations.json file
with open("configurations.json") as configuration:
    config = json.load(configuration)['ultrawide_front_1920_1080']

    size = (config['width'], config['height'])
    
    video_source = config['video_source']
    video_output = config['video_output']
    data_output = config['data_output']

    l_config = config['left_line']
    r_config = config['right_line']



    cl = Counter(uy1 = l_config['upper_1'][1], uy2 = l_config['upper_2'][1], ux1=l_config['upper_1'][0], ux2=l_config['upper_2'][0],
                ly1 = l_config['lower_1'][1], ly2 = l_config['lower_2'][1], lx1=l_config['lower_1'][0], lx2=l_config['lower_2'][0],
                offx=l_config['offx'], offuy=l_config['offuy'], offly=l_config['offly'])
    cr = Counter(uy1 = r_config['upper_1'][1], uy2 = r_config['upper_2'][1], ux1=r_config['upper_1'][0], ux2=r_config['upper_2'][0],
                ly1 = r_config['lower_1'][1], ly2 = r_config['lower_2'][1], lx1=r_config['lower_1'][0], lx2=r_config['lower_2'][0],
                offx=r_config['offx'], offuy=r_config['offuy'], offly=r_config['offly'])
    

cap=cv2.VideoCapture(video_source)
cap=cv2.VideoCapture('realistic_FOV_T_60_edited.mp4')

vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
framerate = int(cap.get(cv2.CAP_PROP_FPS))

#writes the output to the video_output file
out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

#read the classes yolov8 identifies
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

#establish a counter variable for the number of frames that have passed in the video
count=0
#create a new tracker opbject - this keeps track of which objects are actively crossing the lines
tracker=Tracker()

#This section holds IDs of cars going into and out of frame on left and right for tracking
vh_in_left = {}
vh_out_left = {}

vh_in_right = {}
vh_out_right = {}

#This section holds IDs of cars that have gone into and out of frame on left and right
#These are the counters that are updated when a car passes across the lines
counter_in_left = []
counter_out_left = []

counter_in_right = []
counter_out_right = []

#create a new instance of the datawriter class to record the data we gather
data_writer = DataWriter(data_output)

#start time in GMT unix time
start_time = time.time()

#loop through the video
for _ in tqdm.tqdm(range(vid_length)):    
    ret,frame = cap.read()

    #counts the number of frames that have passed
    count += 1
    #this limits the effective framerate of what we are looking at to 10
    if count % (framerate/10) != 0:
        continue
    #resize the frame according to the size specifid in the JSON configuration

    frame=cv2.resize(frame,size)
   
    #run YOLOv8 on the frame
    results=model.predict(frame, verbose=False)
    #get the data from the classification
    a=results[0].boxes.data

    px=pd.DataFrame(a).astype("float")

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
        # gets the various y-axis midpoints used in detection
        mid_center_y=int(y3+y4)//2
        lower_center_y=y4
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
                    data_writer.add_event('out right', start_time + count/framerate)
        
    #this part annotates the lines on the frame
    cl.draw(frame=frame, label_upper='Upper Left', label_lower='Lower Left', color=(0,0,255))
    cr.draw(frame=frame, label_upper='Upper Right', label_lower='Lower Right', color=(255,0,0))
    
    #gets the number of cars in and out by counting the length of the arrays
    cin_Left = (len(counter_in_left)) # counter for in left
    cout_Left = (len(counter_out_left)) # counter for out left
    cin_Right = (len(counter_in_right)) # counter for in right
    cout_Right = (len(counter_out_right)) # counter for out right
    
    #displays the counts of cars in and out using openCV
    cv2.putText(frame,('inLeft: ')+str(cin_Left),(40,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    cv2.putText(frame,('outLeft: ')+str(cout_Left),(40,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    cv2.putText(frame,('inRight: ')+str(cin_Right),(840,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    cv2.putText(frame,('outRight: ')+str(cout_Right),(840,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    
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