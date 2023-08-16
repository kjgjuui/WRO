import RPi.GPIO as GPIO
import time

# Set the GPIO mode
GPIO.setmode(GPIO.BCM)

# Set the pin for the servo motor
servo_pin = 12

# Set the frequency for the servo motor (usually 50 Hz)
servo_freq = 50

# Set the angle range for the servo motor (adjust according to your servo specs)
angle_min = 0
angle_max = 180

# Initialize the GPIO pin for the servo motor
GPIO.setup(servo_pin, GPIO.OUT)

# Create PWM object for the servo motor
servo_pwm = GPIO.PWM(servo_pin, servo_freq)

# Start PWM with 0% duty cycle (servo in the neutral position)
servo_pwm.start(0)

# Function to control servo motor position
def control_servo(angle):
    if angle < angle_min:
        angle = angle_min
    elif angle > angle_max:
        angle = angle_max

    duty_cycle = angle / 18 + 2
    servo_pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.3)  # Wait for the servo to move to the desired position

# zero(37) # lift(20) #ri(46)60
control_servo(40)
time.sleep(1)
servo_pwm.stop()
GPIO.cleanup()
