# import the necessary packages:
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui


def main():
		I=getImage()
		Z=I
		B=getChannelet(I)
		Boundary=getBoundary(B)
		ROI=getROI(B,I)
		Contours=getcontours(Boundary,ROI,Z)
		Biggest=getbiggest(Z,Contours)



#task histogram
def  getImage():
		f = easygui.fileopenbox()
		I = cv2.imread(f)
		return I

# TODO Change this to include better filter
def getChannelet(I):
		# convert to YUV
		YUV = cv2.cvtColor(I, cv2.COLOR_BGR2YUV)
		Single = YUV[:,:,2]

		# Pass through a high pass filter
		k =np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=float)
		Filter =cv2.filter2D(I,ddepth=-1,kernel=k)

		# Grab gray scale 
		# TODO change it <SEAN> (discuss)
		G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
		
		# Get binary
		ret2,B = cv2.threshold(G,0,255,cv2.THRESH_OTSU)

		cv2.imshow("img", ret2)

		return B

# Get the ROI
def getROI(B,I):
		ROI = cv2.bitwise_and(I,I,mask=B)
		return ROI

# Create the Boundary of objects
def getBoundary(B):
		# Shape
		shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
		# create the kernal +size
		kernel = np.ones((2,2),np.uint8)
		# Create the boundary
		Boundary = cv2.morphologyEx(B,cv2.MORPH_GRADIENT,shape)
		# Get the regions of interest (Cells)
		return Boundary



# Create the contours and just bound a rectangle on a clean image of the first
def getcontours(Boundary,ROI,R):
		# Find the contours
		c, hierarchy = cv2.findContours(Boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		# Sort contours
		c = sorted(c, key=cv2.contourArea, reverse=True)
		# Draw contours
		Z = cv2.drawContours(ROI, c, -1, (0,255,0), 2)
  
		x,y,w,h=cv2.boundingRect(c[0])
		cv2.rectangle(R,(x,y),(x+w,y+h),(0,255,255),2)
		cv2.imshow("img",R)
		key = cv2.waitKey(0)


def show(I):
		cv2.imshow("img",I)
		key = cv2.waitKey(0)

main()
