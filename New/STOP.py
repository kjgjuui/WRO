import RPi.GPIO as GPIO
import time
GPIO.setwarnings(False)
# Define GPIO pins for L298N motor driver
enable_pin = 19
input1_pin = 17
input2_pin = 18

# Set GPIO mode
GPIO.setmode(GPIO.BCM)

# Setup GPIO pins for DC motor
GPIO.setup(enable_pin, GPIO.OUT)
GPIO.setup(input1_pin, GPIO.OUT)
GPIO.setup(input2_pin, GPIO.OUT)

# Create PWM object for DC motor speed control
pwm = GPIO.PWM(enable_pin, 100)  # Frequency = 100 Hz

# Start PWM with 0% duty cycle to stop the motor initially
pwm.start(100)


    # Stop the motor
pwm.ChangeDutyCycle(0)

try:
    
    
   
    pwm.stop()
    
except KeyboardInterrupt:
    # Stop PWM and cleanup GPIO pins
    pwm.stop()
    GPIO.cleanup()
 





  