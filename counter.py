import cv2

class Counter:

    def __init__(self, upper_y1, upper_y2, lower_y1, lower_y2, x1, x2):
        self.upper_y1 = upper_y1
        self.upper_y2 = upper_y2
        self.lower_y1 = lower_y1
        self.lower_y2 = lower_y2
        self.x1 = x1
        self.x2 = x2

    def draw(self, frame, label_upper='upper', label_lower='lower', color=(255,255,255)):
        """
        Draws the lines on a frame passed into the function
        """

        #draws upper line and label
        cv2.line(frame,(self.x1,self.upper_y1),(self.x2,self.upper_y2),color,1) # X-Coordinates for upper Line
        cv2.putText(frame,(label_upper),(self.x1,self.upper_y1),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,color,2) # Adds text above upper Line

        #draws lower line and label
        cv2.line(frame,(self.x1,self.lower_y1),(self.x2,self.lower_y2),color,1) # X-Coordinates for upper Line
        cv2.putText(frame,(label_lower),(self.x1,self.lower_y1),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,color,2) # Adds text above upper Line