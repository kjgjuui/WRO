# importing libraries
import RPi.GPIO as GPIO
import time

GPIO.setwarnings(False)

#defining GPIOs

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

#setting GPIOs

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


#functions

#ultrasonic function
def measure_distance(trigger_pin, echo_pin):
    GPIO.output(trigger_pin, GPIO.HIGH)
    time.sleep(0.01)
    GPIO.output(trigger_pin, GPIO.LOW)
    while GPIO.input(echo_pin) == GPIO.LOW:
        pulse_start = time.time()
    while GPIO.input(echo_pin) == GPIO.HIGH:
        pulse_end = time.time()
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 34300 / 2 # in cm
    return distance

# servo motor function
def control_servo(angle):
    min_angle = 0
    max_angle = 26
    if angle < min_angle:
        angle = min_angle
    elif angle > max_angle:
        angle = max_angle
    duty_cycle = angle / 18 + 2
    servo_pwm.ChangeDutyCycle(duty_cycle)
    
# DC motor function
def control_motor(direction):
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
# main function 
def main():
    GPIO.output(input1_pin, GPIO.LOW)
    GPIO.output(input2_pin, GPIO.HIGH)
    pwm.ChangeDutyCycle(100)
    
    try:
        f=measure_distance(23,24)
        r=measure_distance(27, 22)
        l=measure_distance(5, 6)
        
        GPIO.output(input1_pin, GPIO.LOW)
        GPIO.output(input2_pin, GPIO.HIGH)
        servo_pwm.start(0)
        control_servo(15)
        pwm.start(100)
        while True :
              
            f1=measure_distance(23,24)
            r1=measure_distance(27, 22)
            l1=measure_distance(5, 6)
            print(f1,r1,l1)
   ####speed         
            if f1>100:
               pwm.start(100)
            elif f1<100 and f1>80 :
               pwm.start(90)
            elif f1<80 and f1>60  :
               pwm.start(70)
            elif f1<60 and f1>40  :
               pwm.start(50)
            elif f1<40   :
               pwm.start(40)
    ####speed end
               

           
##################################################################################               
            if f1<100 :
                if l1>100 :
                    while True :
                        f1=measure_distance(23,24)
                        r1=measure_distance(27, 22)
                        l1=measure_distance(5, 6)
                        GPIO.output(input1_pin, GPIO.LOW)
                        GPIO.output(input2_pin, GPIO.HIGH)
                        pwm.start(80)
                        if l1>100:
                            print("l1>100")
                            control_servo(0)
                            time.sleep(1)
                            control_servo(15)
                            if f1<10:
                              
                               GPIO.output(input1_pin, GPIO.HIGH)
                               GPIO.output(input2_pin, GPIO.LOW)
                               pwm.start(100)
                               time.sleep(3)
    #########right
               # elif r1> 100:
                
                    #while True :
                       # f1=measure_distance(23,24)
                       # r1=measure_distance(27, 22)
                       #l1=measure_distance(5, 6)
                        
                      #  if r1>100:
                            #print("rrrrrr1>100")
                           # pwm.start(60)
                           # control_servo(2)
                           # time.sleep(2)
                           # control_servo(15)
################################################## 
###########################################################################              
            if f1<30:
               pwm.start(20)
               GPIO.output(input1_pin, GPIO.HIGH)
               GPIO.output(input2_pin, GPIO.LOW)
               pwm.start(100)
               time.sleep(1)
               if r1<l1  :
                   control_servo(2)
                   
               else :
                   control_servo(25)
################################################       
        
 
    except KeyboardInterrupt:
        pwm.stop()
        GPIO.cleanup()
main()
