import cv2
import numpy as np

#img = cv2.imread("myimg.jpeg")

img = np.zeros((512,512,3),np.uint8)

cv2.rectangle(img,(0,0),(400,250),(0,0,255),cv2.FILLED)
cv2.line(img,(0,0),(512,512),(0,255,0),3)
cv2.circle(img,(200,200),50,(255,255,0),3)

cv2.putText(img,"My Image",(40,40),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

cv2.imshow("Image",img)
cv2.waitKey(0)