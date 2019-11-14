import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
##import easygui

######### Reading Images #########

#Opening the image:

I = cv2.imread("../images/cropped.jpg")
cv2.imshow("Orig image", I)

#Converting to YUV colour space and extracting the V channel:

YUV = cv2.cvtColor(I, cv2.COLOR_BGR2YUV)
V = YUV[:,:,2] 
cv2.imshow("Out", V) 


#Creating threshold image
ret, thresh = cv2.threshold(V, 120, 255, 0)

shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
NewMask = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,shape)

NewMaskout = cv2.dilate(NewMask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20)))
cv2.imshow("Maskout", NewMaskout) 

contours, hierarchy = cv2.findContours(NewMaskout, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]

#Drawing a circle around the ROI
(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
cv2.circle(I,center,radius,(0,255,0),2)

font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(I,'Nucleus',(int(x)-int(radius),int(y)+int(radius)+23), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow("OUT", I) 

print (int(radius))

key = cv2.waitKey(0)
