import RPi.GPIO as GPIO
import time

# Define GPIO pins for the ultrasonic sensor
GPIO.setwarnings(False)
trigger_pin = 23
echo_pin = 24

# Set GPIO mode
GPIO.setmode(GPIO.BCM)

# Setup GPIO pins for the ultrasonic sensor
GPIO.setup(trigger_pin, GPIO.OUT)
GPIO.setup(echo_pin, GPIO.IN)

def measure_distance():
    # Send ultrasonic pulse
    GPIO.output(trigger_pin, GPIO.HIGH)
    time.sleep(0.01)
    GPIO.output(trigger_pin, GPIO.LOW)

    # Wait for echo to start
    while GPIO.input(echo_pin) == GPIO.LOW:
        pulse_start = time.time()

    # Wait for echo to end
    while GPIO.input(echo_pin) == GPIO.HIGH:
        pulse_end = time.time()

    # Calculate distance
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 34300 / 2  # Speed of sound in cm/s

    return distance

try:
    while True:
        distance = measure_distance()
        print("Distance:", distance, "cm")
        time.sleep(1)

except KeyboardInterrupt:
    # Cleanup GPIO pins
    GPIO.cleanup()
