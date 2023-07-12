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
cy1=323 # Y-Coordinates for upper Line
cy2=333 # Y-Coordinates for lower Line
offset1=4
offset2=6

# General Code
vh_in = {}
vh_out = {}

counter_in = []
counter_out = []

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.boxes
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
             
    # Object Tracker
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        # Define what classes of objects to look for and record coordinates
        if 'car' in c:
            list.append([x1,y1,x2,y2])
        if 'truck' in c:
            list.append([x1,y1,x2,y2])
        if 'bus' in c:
            list.append([x1,y1,x2,y2])
        if 'bicycle' in c:
            list.append([x1,y1,x2,y2])
        if 'motorcycle' in c:
            list.append([x1,y1,x2,y2])
            
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        
    # Counting vehicles going "in" to frame
        if cy1 < (cy+offset1) and cy1 > (cy-offset1):
            vh_in[id] = cy
        if id in vh_in:
            if cy2 < (cy+offset2) and cy2 > (cy-offset2):
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1) # Draw circle
                cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2) # Give and Print ID
                if id not in counter_in:
                    counter_in.append(id)
                    
    # Counting vehicles going "out" of frame
        if cy2 < (cy+offset2) and cy2 > (cy-offset2):
            vh_out[id] = cy
        if id in vh_out:
            if cy1 < (cy+offset1) and cy1 > (cy-offset1):
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1) # Draw circle
                cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2) # Give and Print ID
                if id not in counter_out:
                    counter_out.append(id)
                    
        
        
        # cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
        # cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
           

# For long_range_b.mp4
    cv2.line(frame,(184,cy1),(814,cy1),(255,255,255),1) # X-Coordinates for upper Line
    cv2.putText(frame,('1Line'),(184,318),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,255,255),2) # Adds text above upper Line
    
    cv2.line(frame,(10,cy2),(1007,cy2),(255,255,255),1) # X-Coordinates for lower Line
    cv2.putText(frame,('2Line'),(1,331),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,255,255),2)
    
# General Code
    cin = (len(counter_in))
    cout = (len(counter_out))
    
    
    cv2.putText(frame,('In:')+str(cin),(60,40),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,255,255),2)
    cv2.putText(frame,('Out:')+str(cout),(60,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,255,255),2)
    
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

