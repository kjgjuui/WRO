from collections import deque
from imutils.video import VideoStream
import imutils
import time
import numpy as np
import cv2

# Initialize the camera using multithreading
camera = VideoStream(src=1).start()

# Allow the camera to warm up
time.sleep(2.0)

# Define a kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Create deques to store tracked pillars' points
green_points = deque(maxlen=10)
red_points = deque(maxlen=100)

# Define HSV color ranges for green and red
green_min_hsv = np.array([37, 38, 24])
green_max_hsv = np.array([99, 255, 255])

red_min_hsv = np.array([160, 100, 100])
red_max_hsv = np.array([190, 255, 255])

red_min_hsv = np.array([100, 100, 0])
red_max_hsv = np.array([255, 255, 255])

# Define area thresholds for detection
green_threshold_area = 300
red_threshold_area = 300

while True:

    frame = camera.read()
    frame = imutils.resize(frame, width=300)

    # Apply Gaussian blur and convert to HSV color space
    blurred_frame = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # Green Mask
    green_mask = cv2.inRange(hsv_frame, green_min_hsv, green_max_hsv)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    # Red Mask
    red_mask = cv2.inRange(hsv_frame, red_min_hsv, red_max_hsv) 
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # Find contours in masks
    green_contours = cv2.findContours(green_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    green_contours = imutils.grab_contours(green_contours)
    red_contours = imutils.grab_contours(red_contours)

    last_tracked_green_pillar_center_point = None
    last_tracked_green_radius = None
    last_tracked_red_pillar_center_point = None
    last_tracked_red_radius = None

    # Process red contours
    for red_contour in red_contours:
        red_max_area_contour = max(red_contours, key=cv2.contourArea)
        ((rx, ry), red_radius) = cv2.minEnclosingCircle(red_max_area_contour)
        rM = cv2.moments(red_max_area_contour)
        red_center = (int(rM["m10"] / rM["m00"]), int(rM["m01"] / rM["m00"]))

        if red_radius > 10:
            cv2.circle(frame, (int(rx), int(ry)), int(red_radius), (0, 0, 255), 2)
            cv2.circle(frame, red_center, 5, (0, 255, 255), -1)

            red_points.appendleft(red_center)
            last_tracked_red_pillar_center_point = red_center
            last_tracked_red_radius = red_radius

    # Process green contours
    for green_contour in green_contours:
        green_max_area_contour = max(green_contours, key=cv2.contourArea)
        ((gx, gy), green_radius) = cv2.minEnclosingCircle(green_max_area_contour)
        gM = cv2.moments(green_max_area_contour)
        green_center = (int(gM["m10"] / gM["m00"]), int(gM["m01"] / gM["m00"]))

        if green_radius > 10:
            cv2.circle(frame, (int(gx), int(gy)), int(green_radius), (0, 255, 0), 2)
            cv2.circle(frame, green_center, 5, (0, 255, 255), -1)

            green_points.appendleft(green_center)
            last_tracked_green_pillar_center_point = green_center
            last_tracked_green_radius = green_radius

    if last_tracked_green_radius is None:
        last_tracked_green_radius = 0

    if last_tracked_red_radius is None:
        last_tracked_red_radius = 0

    if last_tracked_red_radius < last_tracked_green_radius:
        print ("Start turn left from the center point: ", last_tracked_green_pillar_center_point)
    elif last_tracked_red_radius > last_tracked_green_radius:
        print ("Start turn right from the center point: ", last_tracked_red_pillar_center_point)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

camera.stop()
cv2.destroyAllWindows()
