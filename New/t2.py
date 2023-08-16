# importing libraries
import RPi.GPIO as GPIO
import time

GPIO.setwarnings(False)
# defining GPIOs

# Ultrasonic sensors GPIOs
Front_ultrasonic_trigger_pin = 23
Front_ultrasonic_echo_pin = 24
Right_ultrasonic_trigger_pin = 27
Right_ultrasonic_echo_pin = 22
Left_ultrasonic_trigger_pin = 5
Left_ultrasonic_echo_pin = 6

# DC motor GPIOs
enable_pin = 19
input1_pin = 18
input2_pin = 17

# Servo motor GPIO
servo_pin = 12

# setting GPIOs

# Set GPIO mode
GPIO.setmode(GPIO.BCM)

# Setup DC motor pins
GPIO.setup(enable_pin, GPIO.OUT)
GPIO.setup(input1_pin, GPIO.OUT)
GPIO.setup(input2_pin, GPIO.OUT)

# Setup ultrasonic sensors pins
GPIO.setup(Front_ultrasonic_trigger_pin, GPIO.OUT)
GPIO.setup(Front_ultrasonic_echo_pin, GPIO.IN)
GPIO.setup(Right_ultrasonic_trigger_pin, GPIO.OUT)
GPIO.setup(Right_ultrasonic_echo_pin, GPIO.IN)
GPIO.setup(Left_ultrasonic_trigger_pin, GPIO.OUT)
GPIO.setup(Left_ultrasonic_echo_pin, GPIO.IN)

# Setup GPIO pin for servo motor
GPIO.setup(servo_pin, GPIO.OUT)

# PWM objects
pwm = GPIO.PWM(enable_pin, 100)
servo_pwm = GPIO.PWM(servo_pin, 50)


# functions

# ultrasonic function
def measure_distance(trigger_pin, echo_pin):
    GPIO.output(trigger_pin, GPIO.HIGH)
    time.sleep(0.01)
    GPIO.output(trigger_pin, GPIO.LOW)
    while GPIO.input(echo_pin) == GPIO.LOW:
        pulse_start = time.time()
    while GPIO.input(echo_pin) == GPIO.HIGH:
        pulse_end = time.time()
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 34300 / 2  # in cm
    return distance


# servo motor function
def control_servo(angle):
    min_angle = 0
    max_angle = 50
    if angle < min_angle:
        angle = min_angle
    elif angle > max_angle:
        angle = max_angle
    duty_cycle = angle / 18 + 2
    servo_pwm.ChangeDutyCycle(duty_cycle)


# DC motor function
def control_motor(direction):
    if direction == "run":
        GPIO.output(input1_pin, GPIO.HIGH)
        GPIO.output(input2_pin, GPIO.LOW)
        speed = 30
    elif direction == "slow":
        GPIO.output(input1_pin, GPIO.HIGH)
        GPIO.output(input2_pin, GPIO.LOW)
        for i in range(4):
            speed -= 5
            pwm.ChangeDutyCycle(speed)
            time.sleep(0.1)
        return 0
    else:
        GPIO.output(input1_pin, GPIO.LOW)
        GPIO.output(input2_pin, GPIO.LOW)
        return 0
    pwm.ChangeDutyCycle(speed)


# main function
def main():
    control_servo(37)
    front_distance = measure_distance(23,24)
    right_distance = measure_distance(27,22)
    left_distance = measure_distance(5,6)
    try:
        while True:
            if front_distance > 50:
                control_motor("run")
            elif front_distance < 50:
                control_motor("slow")
                if right_distance > left_distance:
                    control_servo(50)
                    control_motor("run")
                    time.sleep(2) # spin time
                    control_servo(37)
                    control_motor("run")
                elif left_distance < left_distance:
                    control_servo(0)
                    control_motor("run")
                    time.sleep(2) # spin time
                    control_servo(37)
                    control_motor("run")
    except KeyboardInterrupt:
        pwm.stop()
        GPIO.cleanup()
main()