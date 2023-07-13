# Tracking and Counting
# Being used by CAT Vehicle Group 2
# Altered by Adhith, John, and Max
# Last updated 12 July 2023

import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
#idk why I'm actually importing this, we could just use a placeholder. Would feel weird though
import time

import tqdm

model=YOLO('yolov8s.pt') # Change model if needed



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Describe name of video being used
cap=cv2.VideoCapture('long_range_b.mp4')

#get the resolution of the video capture - because this is trimmed later on, I got lazy and hard coded it 
size = (1020, 500)
vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   
# Below VideoWriter object will create a frame of above defined
# The output is stored in 'filename.avi' file.
# you have to add this to your .gitignore file (add the line below)
# output.*
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 10.0, (1050,500))



#read the classes yolov8 identifies
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 


#establish a counter variable for the number of frames that have passed in the video
count=0
#create a new tracker opbject - idk wth this does, because there were no comments when I got here
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
    if count % 3 != 0:
        continue

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
    
    # Finds the midpoint of the bounding box        
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        # Use Bottom Midpoint later to experiment, but keep midpoints to test against it
        center_x=int(x3+x4)//2
        center_y=int(y3+y4)//2
        
        # Counting vehicles going "in" to frame
        if coord_y1 < (center_y+offset1) and coord_y1 > (center_y-offset1):
            vh_in[id] = center_y
        if id in vh_in:
            if coord_y2 < (center_y+offset2) and coord_y2 > (center_y-offset2):
                cv2.circle(frame,(center_x,center_y),4,(0,0,255),-1) # Draw circle
                cv2.putText(frame,str(id),(center_x,center_y),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2) # Give and Print ID
                if id not in counter_in:
                    counter_in.append(id)
                    #this is where we know a new event occured, because the counter was just incremented
                    #first we get the time the event occured. count/30 is the number of seconds since the video started
                    event_time = start_time + count/30
                    event_time = int(event_time*100) / 100
                    #now append that to data
                    data.append((event_time, 'in'))

                    
        # Counting vehicles going "out" of frame
        if coord_y2 < (center_y+offset2) and coord_y2 > (center_y-offset2):
            vh_out[id] = center_y
        if id in vh_out:
            if coord_y1 < (center_y+offset1) and coord_y1 > (center_y-offset1):
                cv2.circle(frame,(center_x,center_y),4,(0,0,255),-1) # Draw circle
                cv2.putText(frame,str(id),(center_x,center_y),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2) # Give and Print ID
                if id not in counter_out:
                    counter_out.append(id)
                    #we know a new event occured here, because this is where the counter is incremented
                    #first we get the time the event occured. count/30 is the number of seconds since the video started
                    event_time = start_time + count/30
                    event_time = int(event_time*100)/100
                    #now append that to data
                    data.append((event_time, 'out'))
        
        
    #For long_range_b.mp4
    #this part just annotates the frame
    cv2.line(frame,(184,coord_y1),(814,coord_y1),(255,255,255),1) # X-Coordinates for upper Line
    cv2.putText(frame,('1Line'),(184,318),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,255,255),2) # Adds text above upper Line
    
    cv2.line(frame,(10,coord_y2),(1007,coord_y2),(255,255,255),1) # X-Coordinates for lower Line
    cv2.putText(frame,('2Line'),(1,331),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,255,255),2)
    
    # General Code
    #gets the number of cars in and out by counting the length of the arrays
    cin = (len(counter_in))
    cout = (len(counter_out))
    
    #displays the counts of cars in and out using openCV
    cv2.putText(frame,('In: ')+str(cin),(60,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,255,255),2)
    cv2.putText(frame,('Out: ')+str(cout),(60,40),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,255,255),2)
    
    #shows the images and writes it to the video writer
    out.write(frame)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

#close the display window
cap.release()
out.release()
cv2.destroyAllWindows()

print(data)
#save the data
np.savetxt('data.csv', [row for row in data], delimiter=',', fmt='%s')