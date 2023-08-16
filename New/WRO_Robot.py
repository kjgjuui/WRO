import RPi.GPIO as GPIO
import time
# Define GPIO pins for L298N motor driver
enable_pin = 19
input1_pin = 17
input2_pin = 18

# Define GPIO pins for ultrasonic sensors
Front_ultrasonic_trigger_pin = 23
Front_ultrasonic_echo_pin = 24
Right_ultrasonic_trigger_pin = 27
Right_ultrasonic_echo_pin = 22
Left_ultrasonic_trigger_pin = 5
Left_ultrasonic_echo_pin = 6

# Define GPIO pin for servo motor
servo_pin = 12

# Set GPIO mode
GPIO.setmode(GPIO.BCM)

# Setup GPIO pins for DC motor
GPIO.setup(enable_pin, GPIO.OUT)
GPIO.setup(input1_pin, GPIO.OUT)
GPIO.setup(input2_pin, GPIO.OUT)

# Setup GPIO pins for ultrasonic sensors
GPIO.setup(sensor1_trigger_pin, GPIO.OUT)
GPIO.setup(sensor1_echo_pin, GPIO.IN)
GPIO.setup(sensor2_trigger_pin, GPIO.OUT)
GPIO.setup(sensor2_echo_pin, GPIO.IN)
GPIO.setup(sensor3_trigger_pin, GPIO.OUT)
GPIO.setup(sensor3_echo_pin, GPIO.IN)

# Setup GPIO pin for servo motor
GPIO.setup(servo_pin, GPIO.OUT)

# Create PWM object for DC motor speed control
pwm = GPIO.PWM(enable_pin, 100)  # Frequency = 100 Hz

# Create PWM object for servo motor
servo_pwm = GPIO.PWM(servo_pin, 50)  # Frequency = 50 Hz

# Start PWM with 100% duty cycle for maximum speed
pwm.start(100)

# Start PWM with 0% duty cycle for initial servo position
servo_pwm.start(0)

# Function to control DC motor direction
def control_motor(direction):
    if direction == "forward":
        GPIO.output(input1_pin, GPIO.HIGH)
        GPIO.output(input2_pin, GPIO.LOW)
    elif direction == "reverse":
        GPIO.output(input1_pin, GPIO.LOW)
        GPIO.output(input2_pin, GPIO.HIGH)
    else:
        return  # Invalid direction

# Function to control servo motor position
def control_servo(angle):
    duty_cycle = angle / 18 + 2
    servo_pwm.ChangeDutyCycle(duty_cycle)

# Function to measure distance using ultrasonic sensor
def measure_distance(trigger_pin, echo_pin):
    # Send ultrasonic pulse
    GPIO.output(trigger_pin, GPIO.HIGH)
    time.sleep(0.1)
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
    servo_rotated = False

    while True:
        # Measure distance from sensor 1
        distance1 = measure_distance(sensor1_trigger_pin, sensor1_echo_pin)
        print("Distance from sensor 1:", distance1, "cm")

        # Check if distance1 is less than or equal to 50cm
        if distance1 <= 50:
            # Stop the motor
            control_motor("stop")

            # Measure distance from sensor 2
            distance2 = measure_distance(sensor2_trigger_pin, sensor2_echo_pin)
            print("Distance from sensor 2:", distance2, "cm")

            # Measure distance from sensor 3
            distance3 = measure_distance(sensor3_trigger_pin, sensor3_echo_pin)
            print("Distance from sensor 3:", distance3, "cm")

            # Compare distances from sensor 2 and sensor 3
            if distance2 > distance3:
                # Rotate the servo motor by 45 degrees
                if not servo_rotated:
                    control_servo(45)
                    print("Rotating servo motor by 45 degrees")
                    servo_rotated = True
            else:
                # Rotate the servo motor by -45 degrees
                if not servo_rotated:
                    control_servo(-45)
                    print("Rotating servo motor by -45 degrees")
                    servo_rotated = True
        else:
            # Run the motor in forward direction at maximum speed
            if servo_rotated:
                control_motor("forward")

        time.sleep(0.1)  # Refresh rate of 0.1 seconds

except KeyboardInterrupt:
    # Cleanup GPIO pins and stop PWM
    pwm.stop()
    servo_pwm.stop()
    GPIO.cleanup()
