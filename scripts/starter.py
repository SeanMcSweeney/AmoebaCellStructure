# import the necessary packages:
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui
import imutils

#task histogram
def  getImage():
		#f = easygui.fileopenbox()
		I = cv2.imread("../images/amoeba.jpg")
		return I

# TODO Change this to include better filter
def getChannelet(I):
		# convert to YUV
		YUV = cv2.cvtColor(I, cv2.COLOR_BGR2YUV)
		Single = YUV[:,:,2]
		# G works on other images
		#G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
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
		return Boundary

def bindRectangle(I,C):
		c = sorted(C, key=cv2.contourArea, reverse=True)
		x,y,w,h=cv2.boundingRect(c[0])
		cv2.rectangle(I,(x,y),(x+w,y+h),(0,255,255),2)
		return I
  
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

# Create the contours and just bound a rectangle on a clean image of the first
def getcontours(Boundary):
	c, hierarchy = cv2.findContours(Boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	return c

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

def findCellMembrance(C,ROI,Current):
	C = sorted(C, key=cv2.contourArea, reverse=True)
	# Calculate the center of the first contour aka largest
	M = cv2.moments(C[0])
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	cv2.putText(Current, "Membrane", (cX + int(cX/2)-40, cY-80),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	# places next to membrane
	cv2.putText(Current, "Cytoplasm", (cX-30, cY-140),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),2)
	# Draw contours
	I = cv2.drawContours(Current, C[0], -1, (0,0,255), 2)
	return I

def DrawNucleus(count,arrayXYWH):
	disx = arrayXYWH[0]
	disy = arrayXYWH[1]
	#Drawing a circle around the ROI
	(x,y),radius = cv2.minEnclosingCircle(count)
	center = (int(x + disx),int(y + disy))
	radius = int(radius)
	cv2.circle(I,center,radius,(0,255,0),2)
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(I,'Nucleus',(int(x + disx)-int(radius),int(y + disy)+int(radius)+23), font, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
	return I

def crop2(Boundary,I):
	c, hierarchy = cv2.findContours(Boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	c = sorted(c, key=cv2.contourArea, reverse=True)
	x, y, w, h = cv2.boundingRect(c[0])
	croppedimg2 = I[y:y + h, x:x + w]
	return croppedimg2

def singleGray(I):
	# convert to YUV
	grey = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
	return grey

def OverlayImg(I,Crop,arrayXYWH):
	BGR = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
	x = arrayXYWH[0]
	y = arrayXYWH[1]
	s_img = Crop
	l_img = BGR
	x_offset = x
	y_offset = y
	l_img[y_offset:y_offset + s_img.shape[0], x_offset:x_offset + s_img.shape[1]] = s_img
	return l_img

def show(img):
		cv2.imshow("img",img)
		key = cv2.waitKey(0)

I = getImage()
Z = I
B = getChannelet(I)
Boundary = getBoundary(B)
ROI = getROI(B,I)
C = getcontours(Boundary)
I = bindRectangle(Z,C)
arrayXYWH = findXY(Boundary,I)
Croppedimg = crop(Boundary,I)
threshold = getThresh(Croppedimg)
getCont = findNucleus(threshold)
drawNucleus = DrawNucleus(getCont, arrayXYWH)
m = findCellMembrance(C,ROI,drawNucleus)
Croppedimg2 = crop2(Boundary,m)
grey = singleGray(getImage())
FinalImg = OverlayImg(grey,Croppedimg2,arrayXYWH)

show(FinalImg)
key = cv2.waitKey(0)
