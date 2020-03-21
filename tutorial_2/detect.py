import cv2
import numpy as np

vid = cv2.VideoCapture('movingball.mp4')

if (vid.isOpened()== False):
  print("Error opening video stream or file")

while(vid.isOpened()):
  ret, frame = vid.read()
  if ret == True:
    # converting from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Range for lower red
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Range for upper range
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Generating the final mask to detect red color
    mask1 = mask1 + mask2

    # Transforms for better detection
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # calculate moments of binary image
    M = cv2.moments(mask1)
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
    cv2.circle(frame, (cX, cY), 100, (255, 255, 255), 4)

    cv2.imshow('Frame',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  else:
    break

# When everything done, release the video capture object
vid.release()
# Closes all the frames
cv2.destroyAllWindows()