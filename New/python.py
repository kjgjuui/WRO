import cv2
import numpy as np

def get_color_name(hue_value):
    if 0 <= hue_value <= 30:
        return "Red"
    elif 35 <= hue_value <= 85:
        return "Green"
    else:
        return "Unknown"

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of light red color in HSV
    lower_light_red = np.array([0, 100, 100])
    upper_light_red = np.array([10, 255, 255])

    # Define the range of dark red color in HSV
    lower_dark_red = np.array([160, 100, 100])
    upper_dark_red = np.array([180, 255, 255])

    # Define the range of light green color in HSV
    lower_light_green = np.array([35, 100, 100])
    upper_light_green = np.array([85, 255, 255])

    # Threshold the HSV image to get light and dark red and green regions
    mask_light_red = cv2.inRange(hsv, lower_light_red, upper_light_red)
    mask_dark_red = cv2.inRange(hsv, lower_dark_red, upper_dark_red)
    mask_light_green = cv2.inRange(hsv, lower_light_green, upper_light_green)

    # Combine the masks for light and dark colors
    mask_red = mask_light_red + mask_dark_red
    mask_green = mask_light_green

    # Find contours in the masks
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Determine if red or green is present
    if len(contours_red) > 0:
        shared_data = "Red is present"
        print(shared_data)
    elif len(contours_green) > 0:
        shared_data = "Green is present"
        print(shared_data)
    else:
        shared_data = "Neither red nor green"
        print(shared_data)

    # Display the processed frame
    cv2.imshow('Color Detection', frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
