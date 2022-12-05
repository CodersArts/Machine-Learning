import cv2
import numpy as np

img = cv2.imread("myimg.jpeg")

hor = np.hstack((img,img))
ver = np.vstack((hor,hor))

#cv2.imshow("Horizontal",hor)
cv2.imshow("Vertical",ver)

cv2.waitKey(0)