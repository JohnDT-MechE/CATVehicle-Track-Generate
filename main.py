# Tracking and Counting
# Being used by CAT Vehicle Group 2
# Altered by Adhith, John, and Max
# Last updated 24 July 2023

import cv2
import pandas as pd
from ultralytics import YOLO
import tqdm
import json

#custom classes
from tracker import*
from counter import Counter, DataWriter

# NEED TO ALTER CONFIGURATION OF "ZONE" COUNTER IN THIS DOCUMENT -- ONLY LOCATION THOUGH
# THIS IS THE ONLY CONFIGURATION THAT NEEDS TO BE CHANGED IN THIS DOCUMENT FOR VEHICLE PASSING COUNTER
configuration_name = 'ultrawide_front_1020_500'
model=YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

#Code to parse the configurations from the configurations.json file
with open("configurations.json") as configuration:
    # This reads in the specific configuration we are looking for. They are named by perspective and resolution
    config = json.load(configuration)[configuration_name]

    # The JSON configuration stores the size to clip the video footage to
    size = (config['width'], config['height'])

    try:
        start_time = config["timestamp"]
    except:
        start_time = 1689368367
    
    # The JSON configuration also stores the video source and video/data outputs
    video_source = config['video_source']
    video_output = config['video_output']
    data_output = config['data_output']

    try:
        data_output_zone = config['data_output_zone']
    except:
        data_output_zone = 'data.csv'

    # JSON configurations have two objects, one for the left line locations and offsets, and one for the right line
    l_config = config['left_line']
    r_config = config['right_line']

    # Each l_config or r_config has lists for each endpoint of the two lines, which give coordinates in [x, y] format
    cl = Counter(uy1 = l_config['upper_1'][1], uy2 = l_config['upper_2'][1], ux1=l_config['upper_1'][0], ux2=l_config['upper_2'][0],
                ly1 = l_config['lower_1'][1], ly2 = l_config['lower_2'][1], lx1=l_config['lower_1'][0], lx2=l_config['lower_2'][0],
                offx=l_config['offx'], offuy=l_config['offuy'], offly=l_config['offly'])
    cr = Counter(uy1 = r_config['upper_1'][1], uy2 = r_config['upper_2'][1], ux1=r_config['upper_1'][0], ux2=r_config['upper_2'][0],
                ly1 = r_config['lower_1'][1], ly2 = r_config['lower_2'][1], lx1=r_config['lower_1'][0], lx2=r_config['lower_2'][0],
                offx=r_config['offx'], offuy=r_config['offuy'], offly=r_config['offly'])
    

    # Variables for Zone Counter
    # CONFIG ZONE LINES HERE
    zone = config['zone']

    start_area_y = zone['y'][0]
    end_area_y = zone['y'][1]

    leftx_area1 = zone['x_1'][0]
    rightx_area1 = zone['x_1'][1]

    leftx_area2 = zone['x_2'][0]
    rightx_area2 = zone['x_2'][1]
    

cap=cv2.VideoCapture(video_source)

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

# This holds the IDs of cars that are within the shared "Zone"
vh_in_zone1 = {}
vh_in_zone2 = {}

#This section holds IDs of cars that have gone into and out of frame on left and right
#These are the counters that are updated when a car passes across the lines
counter_in_left = []
counter_out_left = []

counter_in_right = []
counter_out_right = []

# This is the counter that is updated when a car enters the shared "Zone"
counter_in_zone1 = []
counter_in_zone2 = []



#create a new instance of the datawriter class to record the data we gather
data_writer = DataWriter(data_output)
zone_writer = DataWriter(data_output_zone, header='time,left,right,total')


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
    
    # Clears the "Zone" Counter Lists every time a frame is looked at
    vh_in_zone1.clear()
    counter_in_zone1.clear()
    vh_in_zone2.clear()
    counter_in_zone2.clear()
             
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
        # gets the various y-axis midpoints of bounding boxes used in detection
        mid_center_y=int(y3+y4)//2
        lower_center_y=y4
        upper_center_y=y3
        lower_quarter_center_y=int((y3*0.25)+(y4*0.75))
        
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
        if cl.within_lower_line(center_x, lower_quarter_center_y):
            vh_out_left[id] = lower_quarter_center_y
        if id in vh_out_left:
            if cl.within_upper_line(center_x, lower_quarter_center_y):
                cv2.circle(frame,(center_x,lower_quarter_center_y),4,(0,0,255),-1) # Draw circle
                cv2.putText(frame,str(id),(center_x,lower_quarter_center_y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2) # Give and Print ID
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
        if cr.within_lower_line(center_x, lower_quarter_center_y):
            vh_out_right[id] = lower_quarter_center_y
        if id in vh_out_right:
            if cr.within_upper_line(center_x, lower_quarter_center_y):
                cv2.circle(frame,(center_x,lower_quarter_center_y),4,(255,0,0),-1) # Draw circle
                cv2.putText(frame,str(id),(center_x,lower_quarter_center_y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2) # Give and Print ID
                if id not in counter_out_right:
                    counter_out_right.append(id)
                    data_writer.add_event('out right', start_time + count/framerate)
                    
        # Counting Vehicles in Zone 1      
        if lower_center_y > (start_area_y) and lower_center_y < (end_area_y) and center_x > (leftx_area1) and center_x < (rightx_area1):
            vh_in_zone1[id] = lower_center_y
            cv2.circle(frame,(center_x,lower_center_y),4,(0,255,0),-1) # Draw circle
            cv2.putText(frame,str(id),(center_x,lower_center_y),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2) # Give and Print ID
        if id in vh_in_zone1:
            if id not in counter_in_zone1:
                counter_in_zone1.append(id)
                
        # Counting Vehicles in Zone 2      
        if lower_center_y > (start_area_y) and lower_center_y < (end_area_y) and center_x > (leftx_area2) and center_x < (rightx_area2):
            vh_in_zone2[id] = lower_center_y
            cv2.circle(frame,(center_x,lower_center_y),4,(0,255,0),-1) # Draw circle
            cv2.putText(frame,str(id),(center_x,lower_center_y),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2) # Give and Print ID
        if id in vh_in_zone2:
            if id not in counter_in_zone2:
                counter_in_zone2.append(id)
                       
                    
    #this part annotates the lines on the frame
    cl.draw(frame=frame, label_upper='Upper Left', label_lower='Lower Left', color=(0,0,255))
    cr.draw(frame=frame, label_upper='Upper Right', label_lower='Lower Right', color=(255,0,0))
    
    # Annotates the lines of the "Zone"
    cv2.line(frame,(leftx_area1,start_area_y),(rightx_area1,start_area_y),(0,255,0),1) 
    cv2.putText(frame,('Begin Zone 1'),(10,start_area_y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2) # Top Zone 1
    
    cv2.line(frame,(leftx_area1,end_area_y),(rightx_area1,end_area_y),(0,255,0),1) 
    cv2.putText(frame,('End Zone 1'),(10,end_area_y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2) # Bottom Zone 1
    
    cv2.line(frame,(leftx_area2,start_area_y),(rightx_area2,start_area_y),(0,255,0),1) 
    cv2.putText(frame,('Begin Zone 2'),(850,start_area_y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2) # Top Zone 2
    
    cv2.line(frame,(leftx_area2,end_area_y),(rightx_area2,end_area_y),(0,255,0),1) 
    cv2.putText(frame,('End Zone 2'),(850,end_area_y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2) # Bottom Zone 2
    
    # Processes length of "Zone" Counter and Prints it to screen
    czone1 = (len(counter_in_zone1))
    cv2.putText(frame,('In Zone 1: ')+str(czone1),(40,130),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(255,255,255),2)
    
    czone2 = (len(counter_in_zone2))
    cv2.putText(frame,('In Zone 2: ')+str(czone2),(840,130),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(255,255,255),2)
    
    czone_total = ((len(counter_in_zone1)) + (len(counter_in_zone2)))
    cv2.putText(frame,('In All Zones: ')+str(czone_total),(450,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(255,255,255),2)

    # Records Zone Data
    zone_writer.add_event(str(czone1) + ', ' + str(czone2) + ', ' + str(czone_total), start_time + count/framerate)
    
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
zone_writer.store_data()