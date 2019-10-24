import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

#task histogram

#take in an image
f = easygui.fileopenbox()
I = cv2.imread(f)

# convert to YUV
YUV = cv2.cvtColor(I, cv2.COLOR_BGR2YUV)
Single = YUV[:,:,2]

# Pass through a high pass filter
k =np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=float)
Filter =cv2.filter2D(I,ddepth=-1,kernel=k)

cv2.imshow("filter", Filter)
cv2.imshow("img", Single)
key = cv2.waitKey(0)