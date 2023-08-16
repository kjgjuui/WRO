import RPi.GPIO as GPIO
import time

# Motor GPIO pins
enable_pin = 19
input1_pin = 18
input2_pin = 17

# Ultrasonic GPIO pins
trigger_pin = 23
echo_pin = 24

# Set GPIO mode and setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(enable_pin, GPIO.OUT)
GPIO.setup(input1_pin, GPIO.OUT)
GPIO.setup(input2_pin, GPIO.OUT)
GPIO.setup(trigger_pin, GPIO.OUT)
GPIO.setup(echo_pin, GPIO.IN)

# Function to control the motor
def motor_forward():
    GPIO.output(enable_pin, GPIO.HIGH)
    GPIO.output(input1_pin, GPIO.LOW)
    GPIO.output(input2_pin, GPIO.HIGH)

def motor_backward():
    GPIO.output(enable_pin, GPIO.HIGH)
    GPIO.output(input1_pin, GPIO.HIGH)
    GPIO.output(input2_pin, GPIO.LOW)

def motor_off():
    GPIO.output(enable_pin, GPIO.LOW)
    GPIO.output(input1_pin, GPIO.LOW)
    GPIO.output(input2_pin, GPIO.LOW)

# Function to measure distance using the Ultrasonic sensor
def get_distance():
    GPIO.output(trigger_pin, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(trigger_pin, GPIO.LOW)

    while GPIO.input(echo_pin) == 0:
        pulse_start = time.time()

    while GPIO.input(echo_pin) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150  # Speed of sound in cm/s

    return distance

try:
    while True:
        distance = get_distance()
        print("Distance: %.2f cm" % distance)

        if distance < 40:  # Adjust the threshold distance as needed
            # If the distance is less than 20 cm, turn back
            print("Obstacle detected. Turning back...")
            motor_off()
            time.sleep(1)  # Add a delay to stop the motor for a moment
            motor_backward()
            time.sleep(1)  # Add a delay to make the robot turn
            
        else:
            # If no obstacle, move forward
            motor_forward()

except KeyboardInterrupt:
    # Clean up GPIO on keyboard interrupt
    motor_off()
    GPIO.cleanup()