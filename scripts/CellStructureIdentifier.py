"""
Cell Structure Identifier
Group assignment
Peter Swayne, Mark O'Byrne, Sean Mc Sweeney

Underlying Concepts
The underlying concept for our program is to identify the different parts of an Amoeba cell.
This includes its nucleus, membrane and cytoplasm which we wanted to label on an image.
The program needed to be simple, efficient and effective.
This would allow the users experience (ux) to be where we wanted it as it is very important in a program.

Performance
The program has been fully optimized to improve the ux.
We originally had small delays while using looping methods to detect different cell parts.
When we were optimising the code we replaced every loop that we could replace with OperCv methods.
OperCv methods have been fully optimised and are therefore far more efficient in these tasks.
The program usually runs in around two seconds which is great for image processing standards.

Algorithm 
1) Read in an image of an amoeba cell
2) get the binary threshold via otsu, after high pass filtering
3) get the boundary of the image
4) Determine the contours in the image
5) Sort the contours
6) Bind a rectangle around the largest 
7) Get the coordinates of the rectangle
8) Crop the image within the rectangle
9) get the threshold of the cropped image
10) Create a mark and dilate it, return largest contour
11) draw a circle around the nucleus and label it nucleus
12) draw the largest contour on the image, label it membrane
13) Crop the image within the rectangle inclusive
14) grayscale the original
15) Overlays the cropped version with labels over the grayscale
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui
import imutils


def getImage():
    f = easygui.fileopenbox()
    I = cv2.imread(f)
    return I


def getChannelet(I):
    YUV = cv2.cvtColor(I, cv2.COLOR_BGR2YUV)
    Single = YUV[:, :, 2]
    k = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float)
    Filter = cv2.filter2D(Single, ddepth=-1, kernel=k)
    ret2, B = cv2.threshold(Filter, 0, 255, cv2.THRESH_OTSU)
    return B


def getBoundary(B):
    shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    Boundary = cv2.morphologyEx(B, cv2.MORPH_GRADIENT, shape)
    return Boundary


def bindRectangle(I, C):
    c = sorted(C, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(c[0])
    cv2.rectangle(I, (x, y), (x + w, y + h), (0, 255, 255), 2)
    return I


def findXY(Boundary, I):
    c, hierarchy = cv2.findContours(Boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = sorted(c, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(c[0])
    arrayXY = [x, y, w, h]
    return arrayXY


def crop(Boundary, I):
    c, hierarchy = cv2.findContours(Boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = sorted(c, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(c[0])
    croppedimg = I[y:y + h, x:x + w]
    return croppedimg


def getcontours(Boundary):
    c, hierarchy = cv2.findContours(Boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return c


def getThresh(I):
    YUV = cv2.cvtColor(I, cv2.COLOR_BGR2YUV)
    V = YUV[:, :, 2]
    ret, thresh = cv2.threshold(V, 120, 255, 0)
    return thresh


def findNucleus(thresh):
    shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    NewMask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, shape)
    NewMaskout = cv2.dilate(NewMask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
    contours, hierarchy = cv2.findContours(NewMaskout, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count = contours[0]
    return count


def findCellMembrance(C, Current):
    C = sorted(C, key=cv2.contourArea, reverse=True)
    M = cv2.moments(C[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.putText(Current, "Membrane", (cX + int(cX / 2) - 40, cY - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(Current, "Cytoplasm", (cX - 30, cY - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    I = cv2.drawContours(Current, C[0], -1, (0, 0, 255), 2)
    return I


def DrawNucleus(count, arrayXYWH):
    disx = arrayXYWH[0]
    disy = arrayXYWH[1]
    (x, y), radius = cv2.minEnclosingCircle(count)
    center = (int(x + disx), int(y + disy))
    radius = int(radius)
    cv2.circle(I, center, radius, (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(I, 'Nucleus', (int(x + disx) - int(radius), int(y + disy) + int(radius) + 23), font, 1.2, (0, 255, 0),
                2, cv2.LINE_AA)
    return I


def crop2(Boundary, I):
    c, hierarchy = cv2.findContours(Boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = sorted(c, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(c[0])
    croppedimg2 = I[y:y + h, x:x + w]
    return croppedimg2


def singleGray(I):
    grey = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    return grey


def OverlayImg(I, Crop, arrayXYWH):
    #
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
    cv2.imshow("img", img)
    cv2.waitKey(0)


I = getImage()
CopiedImage = I.copy()
B = getChannelet(I)
Boundary = getBoundary(B)
C = getcontours(Boundary)
I = bindRectangle(I, C)
arrayXYWH = findXY(Boundary, I)
Croppedimg = crop(Boundary, I)
threshold = getThresh(Croppedimg)
getCont = findNucleus(threshold)
drawNucleus = DrawNucleus(getCont, arrayXYWH)
m = findCellMembrance(C, drawNucleus)
Croppedimg2 = crop2(Boundary, m)
grey = singleGray(CopiedImage)
FinalImg = OverlayImg(grey, Croppedimg2, arrayXYWH)
show(FinalImg)
key = cv2.waitKey(0)


"""
Conclusion 
In conclusion we are very happy with the outcome of our project. 
It fulfils the objectives which were set out during the programs development phase. 
It can identify, label and colour code the different parts of the cell.
The performance is uncanny as it runs in the blink of an eye improving the users experience.
The cell is easy to identify and the design is simplistic which is what we were aiming to achieve
"""
