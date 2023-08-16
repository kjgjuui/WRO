import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# GPIO setup
GPIO.setmode(GPIO.BOARD)
servo_pin = 18
GPIO.setup(servo_pin, GPIO.OUT)
pwm = GPIO.PWM(servo_pin, 50)  # 50 Hz (20 ms PWM period)

# Function to set the servo angle
def set_servo_angle(angle):
    duty = angle / 18 + 2
    GPIO.output(servo_pin, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)
    GPIO.output(servo_pin, False)
    pwm.ChangeDutyCycle(0)

# Start PWM
pwm.start(0)

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert BGR image to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of red color in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Define the range of green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Create a mask for red and green regions
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours in the masks
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if red is detected
    if len(contours_red) > 0:
        print("Red detected!")
        # Turn the servo to the right (adjust the angle as needed)
        set_servo_angle(90)

    # Check if green is detected
    elif len(contours_green) > 0:
        print("Green detected!")
        # Turn the servo to the left (adjust the angle as needed)
        set_servo_angle(0)

    # Display the original frame
    cv2.imshow('Frame', frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and clean up GPIO
cap.release()
cv2.destroyAllWindows()
pwm.stop()
GPIO.cleanup()
