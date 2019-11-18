# import the necessary packages:
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

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
		Filter =cv2.filter2D(Single,ddepth=-1,kernel=k)

		# Get binary
		ret2,B = cv2.threshold(Filter,0,255,cv2.THRESH_OTSU)
		return B

# Get the ROI
def getROI(B,I):
		ROI = cv2.bitwise_and(I,I,mask=B)
		return ROI

# Create the Boundary of objects
def getBoundary(B):
		# Shape
		shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
		# Create the boundary
		Boundary = cv2.morphologyEx(B,cv2.MORPH_GRADIENT,shape)
		# Get the regions of interest (Cells)
		return Boundary


# Create the contours and just bound a rectangle on a clean image of the first
def getcontours(Boundary,ROI,Rect):
		# Find the contours
		c, hierarchy = cv2.findContours(Boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		# Sort contours
		c = sorted(c, key=cv2.contourArea, reverse=True)
		# Draw contours
		Z = cv2.drawContours(ROI, c, -1, (0,255,0), 2)
		x,y,w,h=cv2.boundingRect(c[0])
		cv2.rectangle(Rect,(x,y),(x+w,y+h),(0,255,255),2)
		return Rect

def findXY(Boundary,I):
	c, hierarchy = cv2.findContours(Boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	c = sorted(c, key=cv2.contourArea, reverse=True)
	x, y, w, h = cv2.boundingRect(c[0])
	arrayXY = [x,y,w,h]
	return arrayXY

def crop(Boundary,I):
	c, hierarchy = cv2.findContours(Boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	c = sorted(c, key=cv2.contourArea, reverse=True)
	x, y, w, h = cv2.boundingRect(c[0])
	croppedimg = I[y:y + h, x:x + w]
	return croppedimg

def getThresh(I):
	YUV = cv2.cvtColor(I, cv2.COLOR_BGR2YUV)
	V = YUV[:, :, 2]
	ret, thresh = cv2.threshold(V, 120, 255, 0)
	return thresh

def findNucleus(thresh):
	shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
	NewMask = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,shape)
	NewMaskout = cv2.dilate(NewMask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20)))
	contours, hierarchy = cv2.findContours(NewMaskout, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	count = contours[0]
	return count

def DrawNucleus(count,arrayXYWH):
	disx = arrayXYWH[0]
	disy = arrayXYWH[1]
	#Drawing a circle around the ROI
	(x,y),radius = cv2.minEnclosingCircle(count)
	center = (int(x + disx),int(y + disy))
	radius = int(radius)
	cv2.circle(I,center,radius,(0,255,0),2)
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(I,'Nucleus',(int(x + disx)-int(radius),int(y + disy)+int(radius)+23), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
	return I


def show(I):
		cv2.imshow("img",I)
		key = cv2.waitKey(0)

I = getImage()
Z = I
B = getChannelet(I)
Boundary = getBoundary(B)
ROI = getROI(B,I)
Contours = getcontours(Boundary,ROI,Z)
arrayXYWH = findXY(Boundary,I)
Croppedimg = crop(Boundary,I)
threshold = getThresh(Croppedimg)
getCont = findNucleus(threshold)
drawNucleus = DrawNucleus(getCont, arrayXYWH)
show(drawNucleus)

key = cv2.waitKey(0)
