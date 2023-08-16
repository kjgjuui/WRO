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
pwm.start(60)

def run_dc_motor(speed_command):
    # Set the motor direction based on the speed command
    if speed_command == "run":
        GPIO.output(input1_pin, GPIO.HIGH)
        GPIO.output(input2_pin, GPIO.LOW)
    else:
        GPIO.output(input1_pin, GPIO.LOW)
        GPIO.output(input2_pin, GPIO.HIGH)

    # Calculate the duty cycle for a speed of 30
    initial_speed = 100

    # If the input was 'run,' set the motor to speed 30
    if speed_command == "run":
        pwm.ChangeDutyCycle(initial_speed)
    else:
        # If the input was 'slow,' gradually decrease the motor speed until it stops
        speed = initial_speed
        while speed > 0:
            pwm.ChangeDutyCycle(speed)
            time.sleep(0.1)
            speed -= 5

    # Stop the motor
    pwm.ChangeDutyCycle(0)

try:
   
   #GPIO.output(input1_pin, GPIO.LOW)#revers
   #GPIO.output(input2_pin, GPIO.HIGH)
################################################    
   GPIO.output(input1_pin, GPIO.HIGH)
   GPIO.output(input2_pin, GPIO.LOW)
   pwm.start(70)
   time.sleep(0.2)
   pwm.start(100)
   time.sleep(1-0.7)
   pwm.stop()
   #GPIO.output(input1_pin, GPIO.HIGH)
   #GPIO.output(input2_pin, GPIO.LOW)
   #pwm.start(70)
except Exception as e:
    # Handle the error gracefully
    pwm.stop()
    GPIO.cleanup()
    print("An error occurred:", e)
except KeyboardInterrupt:
    # Stop PWM and cleanup GPIO pins
    pwm.stop()
    GPIO.cleanup()
 


