import RPi.GPIO as GPIO
import time
import numpy as np
from teachable_machine_lite import TeachableMachineLite
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
import YB_Pcb_Car
import cv2

#####################################################################
model_path = '/home/pi/Desktop/WRO/model.tflite'
# Ultrasonic sensors GPIOs
Front_ultrasonic_trigger_pin = 23
Front_ultrasonic_echo_pin = 24
Right_ultrasonic_trigger_pin = 27
Right_ultrasonic_echo_pin = 22
Left_ultrasonic_trigger_pin = 5
Left_ultrasonic_echo_pin = 6

# DC motor GPIOs
input1_pin = 18
input2_pin = 17
enable_pin = 19

# Constants for servo angles
dl = 50
dz = 37
dr = 25

# Servo motor GPIO
servo_pin = 12

# Ultrasonic distances
f1, r1, l1 = 0, 0, 0

# PWM objects
pwm = None
servo_pwm = None

##################################################
interpreter = Interpreter(model_path,
  experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
# Function to initialize GPIO pins
def setup_gpio():
    GPIO.setwarnings(False)
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
    global pwm, servo_pwm
    pwm = GPIO.PWM(enable_pin, 100)
    servo_pwm = GPIO.PWM(servo_pin, 50)

# Function to measure distance using ultrasonic sensor
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

# Function to control the servo motor
def control_servo(angle):
    if angle < 0:
        angle = 0
    elif angle > 180:
        angle = 180

    duty_cycle = angle / 18 + 2
    servo_pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.3)

# Function to control the DC motor
def control_motor(direction):
    global pwm
    speed = 0
    if direction == "run":
        GPIO.output(input1_pin, GPIO.HIGH)
        GPIO.output(input2_pin, GPIO.LOW)
    elif direction == "slow":
        GPIO.output(input1_pin, GPIO.HIGH)
        GPIO.output(input2_pin, GPIO.LOW)
        for i in range(10):
            speed -= 5
            pwm.ChangeDutyCycle(speed)
            time.sleep(0.05)
        return 0
    else:
        return 0
    pwm.ChangeDutyCycle(speed)

# Function to look at an object
def look_to_object():
    print('Start Look to object')
    # Add your code here for looking at an object

# Function to say "Cup"
def say_cup():
    print('Say Cup')
    # Add your code here for saying "Cup"

# Function to say "Card"
def say_card():
    print('Say Card')
    # Add your code here for saying "Card"

def main():
    global pwm, servo_pwm, f1, r1, l1
    GPIO.setmode(GPIO.BCM)
    setup_gpio()
    GPIO.output(input1_pin, GPIO.LOW)
    GPIO.output(input2_pin, GPIO.HIGH)
    pwm.start(100)  # Set PWM duty cycle to 100 (assuming full speed)

    cap = cv2.VideoCapture(0)
    model_path = 'model.tflite'
    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()
    dim = (224, 224)  # Replace with the correct image dimensions for your model

    while True:
        f1 = measure_distance(Front_ultrasonic_trigger_pin, Front_ultrasonic_echo_pin)
        r1 = measure_distance(Right_ultrasonic_trigger_pin, Right_ultrasonic_echo_pin)
        l1 = measure_distance(Left_ultrasonic_trigger_pin, Left_ultrasonic_echo_pin)
        print(f1, r1, l1)

        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dim)
        input_data = np.expand_dims(image, axis=0)
        input_data = input_data.astype('float32') / 255.0

        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        label_id = np.argmax(output_data)

        if label_id == 0:
            # Object Detected: Cup
            print('Object: Cup')
            if f1 < 85 and l1 > 200 and r1 < 80:
                control_motor("run")  # Forward motion for cup detection condition
            elif f1 < 85 and r1 > 200 and l1 < 80:
                control_motor("run")  # Forward motion for cup detection condition
            elif f1 < 85 and l1 < 45:
                control_motor("run")  # Forward motion for cup detection condition
                control_servo(dr)  # Right turn
                time.sleep(1.1)
                control_servo(dz)  # Center the servo
            elif f1 < 85 and r1 < 45:
                control_motor("run")  # Forward motion for cup detection condition
                control_servo(dl)  # Left turn
                time.sleep(1.1)
                control_servo(dz)  # Center the servo
            elif f1 < 53 and (r1 > 150 or l1 > 150):
                control_motor("run")  # Forward motion for cup detection condition
                pwm.start(40)
                GPIO.output(input1_pin, GPIO.LOW)
                GPIO.output(input2_pin, GPIO.HIGH)
                pwm.start(70)
                if r1 < 50:
                    control_servo(dl)  # Lift the servo
                    pwm.start(70)
                    time.sleep(2)
                    control_servo(dz)  # Center the servo
                elif l1 < 50:
                    control_servo(dr)  # Right turn
                    pwm.start(70)
                    time.sleep(2)
                    control_servo(dz)  # Center the servo
                else:
                    control_motor("slow")  # Slow down or stop the motor for other cases

        elif label_id == 1:
            # Object Detected: Card
            print('Object: Card')
            # Add your motor control logic for card detection here
            # For example, move the car forward, backward, turn left/right, etc.

        look_to_object()
        time.sleep(2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

try:
    if __name__ == '__main__':
        main()
except KeyboardInterrupt:
    pwm.stop()
    GPIO.cleanup()
