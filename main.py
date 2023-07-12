# Tracking and Counting
# Being used by CAT Vehicle Group 2
# Altered by Adhith, John, and Max
# Last updated 12 July 2023

import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*

model=YOLO('yolov8s.pt') # Change model if needed



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Describe name of video being used
cap=cv2.VideoCapture('long_range_b.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0

tracker=Tracker()

# For long_range_b.mp4
coord_y1=323 # Y-Coordinates for upper Line
coord_y2=333 # Y-Coordinates for lower Line
offset1=4 # Offset for upper Line
offset2=6 # Offset for lower Line

# General Code
vh_in = {} # Holds IDs of cars going into frame for tracking
vh_out = {} # Holds IDs of cars going out of frame for tracking

counter_in = [] # List of IDs of cars that have gone into frame
counter_out = [] # List of IDs of cars that have come out of frame

#infinitely loop through the video
while True:    
    ret,frame = cap.read()
    #if the video is over, break out of the 'infinite' loop
    if not ret:
        break

    #this code exists to limit the number of frames the code actually looks at
    #counts the number of frames that have passed
    count += 1
    #if the number of frames isn't a multiple of three, skip to the next frame
    if count % 3 != 0:
        continue

    #resize the frame
    frame=cv2.resize(frame,(1020,500))
   

    #run YOLOv8 on the frame
    results=model.predict(frame)
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
            
    # Finds the midpoint of the bounding box        
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        # Use Bottom Midpoint later to experiment, but keep midpoints to test against it
        current_center_x=int(x3+x4)//2
        current_center_y=int(y3+y4)//2
        
        # Counting vehicles going "in" to frame
        if coord_y1 < (current_center_y+offset1) and coord_y1 > (current_center_y-offset1):
            vh_in[id] = current_center_y
        if id in vh_in:
            if coord_y2 < (current_center_y+offset2) and coord_y2 > (current_center_y-offset2):
                cv2.circle(frame,(current_center_x,current_center_y),4,(0,0,255),-1) # Draw circle
                cv2.putText(frame,str(id),(current_center_x,current_center_y),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2) # Give and Print ID
                if id not in counter_in:
                    counter_in.append(id)
                    
        # Counting vehicles going "out" of frame
        if coord_y2 < (current_center_y+offset2) and coord_y2 > (current_center_y-offset2):
            vh_out[id] = current_center_y
        if id in vh_out:
            if coord_y1 < (current_center_y+offset1) and coord_y1 > (current_center_y-offset1):
                cv2.circle(frame,(current_center_x,current_center_y),4,(0,0,255),-1) # Draw circle
                cv2.putText(frame,str(id),(current_center_x,current_center_y),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2) # Give and Print ID
                if id not in counter_out:
                    counter_out.append(id)
        
        
    # For long_range_b.mp4
    cv2.line(frame,(184,coord_y1),(814,coord_y1),(255,255,255),1) # X-Coordinates for upper Line
    cv2.putText(frame,('1Line'),(184,318),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,255,255),2) # Adds text above upper Line
    
    cv2.line(frame,(10,coord_y2),(1007,coord_y2),(255,255,255),1) # X-Coordinates for lower Line
    cv2.putText(frame,('2Line'),(1,331),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,255,255),2)
    
    # General Code
    #gets the number of cars in and out by counting the length of the arrays
    cin = (len(counter_in))
    cout = (len(counter_out))
    
    #displays the counts of cars in and out using openCV
    cv2.putText(frame,('In: ')+str(cin),(60,40),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,255,255),2)
    cv2.putText(frame,('Out: ')+str(cout),(60,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,255,255),2)
    
    #shows the images
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

#close the display window
cap.release()
cv2.destroyAllWindows()

