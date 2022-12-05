import cv2


# reading image
"""
img = cv2.imread("myimg.jpeg")

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
#Reading a video

cap = cv2.VideoCapture("myvideo.mp4")
while True:
  success, img = cap.read()
  print(success)
  cv2.imshow("Video",img)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
"""
"""
# Read HD cam

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 10)
while True:
    success, img = cap.read()
    print(success)
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
"""