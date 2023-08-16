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
angle_min = 0
angle_max = 180
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
    if angle < angle_min:
        angle = angle_min
    elif angle > angle_max:
        angle = angle_max

    duty_cycle = angle / 18 + 2
    servo_pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.3)
    
    
# DC motor function

# main function 
def main():
    GPIO.output(input1_pin, GPIO.LOW)
    GPIO.output(input2_pin, GPIO.HIGH)
    pwm.ChangeDutyCycle(100)
    
    try:
        #f=measure_distance(23,24)
        #r=measure_distance(27, 22)
        #l=measure_distance(5, 6)
        
     
        servo_pwm.start(0)
        control_servo(40)
        time.sleep(.5)
        pwm.start(20)
        time.sleep(.1)
        pwm.start(80)
        while True :
              
            f1=measure_distance(23,24)
            r1=measure_distance(27, 22)
            l1=measure_distance(5, 6)
            print(f1,r1,l1)
            control_servo(40)
   ####speed         
            if f1>150:
               pwm.start(100)
            elif f1<150 and f1>100 :
               pwm.start(90)
            elif f1<100 and f1>60  :
               pwm.start(80)
            elif f1<60 and f1>0  :
               pwm.start(70)
           
            print("nomal")
            if l1>100 :
                print("cornal")
                pwm.start(50)
                control_servo(0)#lift
               
                time.sleep(.7)
                control_servo(40)
                time.sleep(.1)
                
    ####speed end
           

           
##################################################################################               
          
                
            if r1< 40 and f1<10  :
                pwm.stop()
                print("f1<10")
                control_servo(80)
              
                GPIO.output(input1_pin, GPIO.HIGH)
                GPIO.output(input2_pin, GPIO.LOW)
                pwm.start(100)
                time.sleep(1)
                GPIO.output(input1_pin, GPIO.LOW)
                GPIO.output(input2_pin, GPIO.HIGH)
                control_servo(0)

                pwm.start(100)
                time.sleep(1)
                control_servo(40)
                
                ###########
            elif r1< l1 and f1<10  :
                pwm.stop()
                print("f1<10  r1< l1")
                control_servo(80)
              
                GPIO.output(input1_pin, GPIO.HIGH)
                GPIO.output(input2_pin, GPIO.LOW)
                pwm.start(100)
                time.sleep(2)
                GPIO.output(input1_pin, GPIO.LOW)
                GPIO.output(input2_pin, GPIO.HIGH)
                control_servo(0)

                pwm.start(100)
                time.sleep(2)
                control_servo(40)
                
            elif r1<9 and f1>100:
                pwm.stop()
                print("r1<8 f>>>")
                control_servo(0)#lift
                     
                pwm.start(60)
                time.sleep(.5)
                 
            elif r1<15 and f1<15  :
                pwm.stop()
                print("r1<5")
                control_servo(80)
              
                GPIO.output(input1_pin, GPIO.HIGH)
                GPIO.output(input2_pin, GPIO.LOW)
                pwm.start(100)
                time.sleep(1.5)
                GPIO.output(input1_pin, GPIO.LOW)
                GPIO.output(input2_pin, GPIO.HIGH)
                control_servo(40)

                pwm.start(70)
            elif l1<15 and f1<50  :
                pwm.stop()
                print("l1<15")
                control_servo(0)
              
                GPIO.output(input1_pin, GPIO.HIGH)
                GPIO.output(input2_pin, GPIO.LOW)
                pwm.start(100)
                time.sleep(1)
                GPIO.output(input1_pin, GPIO.LOW)
                GPIO.output(input2_pin, GPIO.HIGH)
                control_servo(40)

                pwm.start(70)        
################################################       
        
 
    except KeyboardInterrupt:
        pwm.stop()
        GPIO.cleanup()
main()

