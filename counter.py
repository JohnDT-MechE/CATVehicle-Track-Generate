import cv2
import numpy as np

class DataWriter:

    def __init__(self, filename=None):
        """
        Initialize with a csv file to store the data in
        """
        self.filename = filename
        self.data = []

    def add_event(self, event_type, time):
        """
        Gets event type (left_in, left_out, right_in, right_out) and unix time to the nearest 1000th
        Rounds time to the nearest hundredth of a second and appends a tuple with time and event_type to
        self.data
        """
        rounded_time = int(time*100) / 100
        self.data.append((rounded_time, event_type))

    def print(self):
        """
        prints out the data as a somewhat nicely formatted table to the terminal
        """
        #print a header
        print('time: \t\tEvent Type:')
        #iterate through the entries in self.data
        for event in self.data:
            #for each entry we iterate through, print out a formatted string with the time followed by the event type
            print(f'{event[0]}\t\t{event[1]}')

    def store_data(self, file=None):
        """
        Writes the data in self.data to the csv file specified in filename
        """
        
        np.savetxt(file if file is not None else self.filename, [row for row in self.data], delimiter=',', fmt='%s', header="time,event", comments="")

    

class Counter:

    def __init__(self, uy1, uy2, ux1, ux2, ly1, ly2, lx1, lx2, offx=4, offuy=4, offly=6):
        """
        Initializes a counter object, which stores the coordinates for two lines and contains functions to detect when
        coordinates are close to either line and to draw the lines out on a frame

        upper and lower lines are both defined by their two endpoints, (x1, y1) and (x2, y2)
        x offset is the same for both upper and lower lines, and y offsets are unique
        """
        self.upper_y1 = uy1
        self.upper_y2 = uy2
        self.upper_x1 = ux1
        self.upper_x2 = ux2

        self.lower_y1 = ly1
        self.lower_y2 = ly2
        self.lower_x1 = lx1
        self.lower_x2 = lx2

        self.offset_x = offx
        self.offset_upper_y = offuy
        self.offset_lower_y = offly

        #calculates the "slopes" of the two lines for checking if points are within lines in the future
        self.slope_upper = (self.upper_y2-self.upper_y1)/(self.upper_x2-self.upper_x1)
        self.slope_lower = (self.lower_y2-self.lower_y1)/(self.lower_x2-self.lower_x1)

    def within_upper_line(self, x, y):
        """
        This function checks if the coordinates of a point passed in as x and y are within the offsets of the upper line
        """
        #check if the point is within the x axis boundaries
        if (self.upper_x1-self.offset_x) < x and (self.upper_x2+self.offset_x) > x:
            #calculate the "expected" y value a.k.a. the y value of the line at the x coordinate passed in
            expected_y = (x-self.upper_x1)*self.slope_upper + self.upper_y1
            #check if the y coordinate is within the offset_upper_y of the "expected" y value
            if y < (expected_y+self.offset_upper_y) and y > (expected_y-self.offset_upper_y):
                return True
        return False
    
    def within_lower_line(self, x, y):
        """
        This function checks if the coordinates of a point passed in as x and y are within the offsets of the upper line
        """
        #check if the point is within the x axis boundaries
        if (self.lower_x1-self.offset_x) < x and (self.lower_x2+self.offset_x) > x:
            #calculate the "expected" y value a.k.a. the y value of the line at the x coordinate passed in
            expected_y = (x-self.lower_x1)*self.slope_lower + self.lower_y1
            #check if the y coordinate is within the offset_upper_y of the "expected" y value
            if y < (expected_y+self.offset_lower_y) and y > (expected_y-self.offset_lower_y):
                return True
        return False
        
    def draw(self, frame, label_upper='upper', label_lower='lower', color=(255,255,255), labels=True):
        """
        Draws the lines on a frame passed into the function
        """

        #draws upper line and label
        cv2.line(frame,(self.upper_x1,self.upper_y1),(self.upper_x2,self.upper_y2),color,2) # X-Coordinates for upper Line
        if labels:
            cv2.putText(frame,(label_upper),((self.upper_x1+self.upper_x2)//2,(self.upper_y1+self.upper_y2)//2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1) # Adds text above upper Line

        #draws lower line and label
        cv2.line(frame,(self.lower_x1,self.lower_y1),(self.lower_x2,self.lower_y2),color,2) # X-Coordinates for upper Line
        if labels:
            cv2.putText(frame,(label_lower),((self.lower_x1+self.lower_x2)//2,(self.lower_y1+self.lower_y2)//2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1) # Adds text above upper Line