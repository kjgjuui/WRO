# importing libraries
import RPi.GPIO as GPIO
import time

GPIO.setwarnings(False)
#defining GPIOs
angle_min = 0
angle_max = 180
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
sa = 50
sr = 50
sa_if = 40
sa_r = 40
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
    if angle < angle_min:
        angle = angle_min
    elif angle > angle_max:
        angle = angle_max

    duty_cycle = angle / 18 + 2
    servo_pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.3)
    
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
      
        
        
        servo_pwm.start(0)
        #time.sleep(1)
        control_servo(37)
        time.sleep(0.19)
        pwm.start(sa)
            
        while True :
            
            GPIO.output(input1_pin, GPIO.LOW)
            GPIO.output(input2_pin, GPIO.HIGH)
            #pwm.ChangeDutyCycle(100)
            f1=measure_distance(23,24)
            r1=measure_distance(27, 22)
            l1=measure_distance(5, 6)
            print(f1,r1,l1)
            control_servo(37)
            time.sleep(0.01)
            pwm.start(sa)
#            if f1>100:
 #              pwm.start(100)
  #          else:
   #            pwm.start(70)
               
#            if f1<40:
 #              pwm.stop() 
  #             GPIO.output(input1_pin, GPIO.HIGH)
   #            GPIO.output(input2_pin, GPIO.LOW)
    #           pwm.start(70)
     #          time.sleep(1)
      #         GPIO.output(input1_pin, GPIO.LOW)
       #        GPIO.output(input2_pin, GPIO.HIGH)
#       pwm.start(60)
            if r1<15 or l1<15:
                pwm.stop()
                print("r1<15 or l1<15")
                GPIO.output(input1_pin, GPIO.HIGH)
                GPIO.output(input2_pin, GPIO.LOW)
                pwm.start(sr)
                time.sleep(1)
                pwm.stop() 
                GPIO.output(input1_pin, GPIO.LOW)
                GPIO.output(input2_pin, GPIO.HIGH)
                pwm.start(sa)
               # time.sleep(2)
            f1=measure_distance(23,24)
            r1=measure_distance(27, 22)
            l1=measure_distance(5, 6)   
            GPIO.output(input1_pin, GPIO.LOW)
            GPIO.output(input2_pin, GPIO.HIGH)
            #pwm.ChangeDutyCycle(100)
            pwm.start(sa)
            
            if f1<40:
                
                print("f1<40")
                #pwm.start(40)
                #time.sleep(0.1)
                pwm.stop() 
                GPIO.output(input1_pin, GPIO.HIGH)
                GPIO.output(input2_pin, GPIO.LOW)
                pwm.start(sa)
                time.sleep(0.5)
                
                r1=measure_distance(27, 22)
                l1=measure_distance(5, 6)
                if r1<50:
                    print("r1<50 rvers")
                    control_servo(46)#right
                    pwm.start(sr)
                    time.sleep(1.5)
                    r2=measure_distance(27, 22)
                    if r2>50:
                        control_servo(37)#zero
                    
                elif l1<50:
                    print("l1<50 revers")
                    control_servo(20)#lift
                    pwm.start(sr)
                    time.sleep(1.5)
                    l2=measure_distance(5, 6)
                    if l2>50:
                        control_servo(37)#zero
                
                elif l1<r1:
                    print("l1<r1 revers")
                    control_servo(20)#lift
                    pwm.start(sr)
                    time.sleep(0.5)
                    r2=measure_distance(27, 22)
                    l2=measure_distance(5, 6)
                    if l2>=r2:
                        control_servo(37)#zero
                elif r1<l1:
                    print("r1<l1 rvers")
                    control_servo(46)#right
                    pwm.start(sr)
                    time.sleep(0.5)
                    #f1=measure_distance(23,24)
                    r2=measure_distance(27, 22)
                    l2=measure_distance(5, 6)
                    if r2>=l2:
                        control_servo(37)#zero
                #time.sleep(2)
                pwm.start(sa)
                time.sleep(0.5)
                f2=measure_distance(23,24)
                if f2>40:
                    pwm.stop()                    
                    GPIO.output(input1_pin, GPIO.LOW)
                    GPIO.output(input2_pin, GPIO.HIGH)
                    #pwm.start(10)
                    pwm.start(sa)
            
            elif f1<53:
                print("f1<53")
                #pwm.start(40)
                #time.sleep(0.1)
                pwm.stop()
                GPIO.output(input1_pin, GPIO.HIGH)
                GPIO.output(input2_pin, GPIO.LOW)
                pwm.start(sr)
                if r1<50:
                    print("r1<50 rvers")
                    control_servo(20)#lift
                    pwm.start(sr)
                    time.sleep(1.5)
                    r2=measure_distance(27, 22)
                    if r2>50:
                        control_servo(37)#zero
                    
                elif l1<50:
                    print("l1<50 revers")
                    control_servo(60)#right
                    pwm.start(sr)
                    time.sleep(1.5)
                    l2=measure_distance(5, 6)
                    if l2>50:
                        control_servo(37)#zero
                pwm.start(sa_r) 
                GPIO.output(input1_pin, GPIO.LOW)
                GPIO.output(input2_pin, GPIO.HIGH)
                pwm.start(sa)    
###################################################            
       #     elif r1<15:
      #          pwm.start(40) 
     #           GPIO.output(input1_pin, GPIO.HIGH)
    #            GPIO.output(input2_pin, GPIO.LOW)
   #             print("r1<15")
  #              control_servo(46)#right
 #               pwm.start(60)
                #time.sleep(0.5)
               # r2=measure_distance(27, 22)
              #  if r2>15:
             #       control_servo(37)#zero
            #        pwm.start(50)
           #         time.sleep(0.2)
          #          pwm.start(10)
         #           GPIO.output(input1_pin, GPIO.LOW)
        #            GPIO.output(input2_pin, GPIO.HIGH)
       #             pwm.start(60)
####################################################                    
            #elif l1<15:
            #    pwm.start(40) 
                #GPIO.output(input1_pin, GPIO.HIGH)
                #GPIO.output(input2_pin, GPIO.LOW)
            #    print("l1<15")
            #    control_servo(20)#lift
             #   pwm.start(60)
             #   time.sleep(0.5)
             #   l2=measure_distance(5, 6)
             #   if l2>15:
             #       control_servo(37)#zero
             #       pwm.start(50)
              #      time.sleep(0.2)

              
                    #pwm.start(10)
                    #GPIO.output(input1_pin, GPIO.LOW)
                    #GPIO.output(input2_pin, GPIO.HIGH)
                    #pwm.start(60)
#################################################            
            
            
# zero(37) # lift(20) #ri(46)
            elif f1<85 and l1>200 and r1<80:
                
                pwm.start(sa_if)
                print("l1>200")
                control_servo(20)#lift
                time.sleep(1)
                control_servo(37)#zero
                
            elif f1<85 and r1>200 and l1<80:
                
                pwm.start(sa_if)
                print("r1>200")
                control_servo(46)#right
                time.sleep(1)
                control_servo(37)#
                
                
            
            elif f1<85 and l1<45 :
                pwm.start(sa_if)
                print("rrr")
                control_servo(43)#right
                time.sleep(0.5)
                control_servo(37)#zero
            elif f1<85 and r1<45:
                pwm.start(sa_if)
                print("lll")
                control_servo(23)#lift
                time.sleep(0.5)
                control_servo(37)#
            
            pwm.start(sa)
#            elif f1<80 and r1<l1:
 #               print("r1<l1")
  #              control_servo(23)#lift
   #             time.sleep(0.7)
    #            control_servo(37)#
     #       elif f1<80 and l1<r1:
      #          print("l1<r1")
       #         control_servo(46)#right
        #        time.sleep(0.7)
         #       control_servo(37)#zero   
                
#             if r1>l1 :
#                 print("rrr")
#                 control_servo(46)#right
#                 time.sleep(0.3)
#                 control_servo(37)#zero
#             elif l1>r1:
#                 print("lll")
#                 control_servo(23)#lift
#                 time.sleep(0.3)
#                 control_servo(37)#zero
            
  #  stop mator

                        
        
        
        
       # servo_pwm.start(0)
        #time.sleep(3)
        #pwm.stop()
    
        
        #control_motor("forward")
#         time.sleep(3)
#         control_motor("slow")
    except KeyboardInterrupt:
        pwm.stop()
        GPIO.cleanup()
main()
