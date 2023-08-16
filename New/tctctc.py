# importing libraries
import RPi.GPIO as GPIO
import time
from python import shared_data

GPIO.setwarnings(False)

angle_min = 0
angle_max = 180

Front_ultrasonic_trigger_pin = 23
Front_ultrasonic_echo_pin = 24
Right_ultrasonic_trigger_pin = 27
Right_ultrasonic_echo_pin = 22
Left_ultrasonic_trigger_pin = 5
Left_ultrasonic_echo_pin = 6


enable_pin = 19
input1_pin = 18
input2_pin = 17
servo_pin = 12
sa = 80


dr = 50
dz = 37
dl = 20


GPIO.setmode(GPIO.BCM)

GPIO.setup(enable_pin, GPIO.OUT)
GPIO.setup(input1_pin, GPIO.OUT)
GPIO.setup(input2_pin, GPIO.OUT)


GPIO.setup(Front_ultrasonic_trigger_pin, GPIO.OUT)
GPIO.setup(Front_ultrasonic_echo_pin, GPIO.IN)
GPIO.setup(Right_ultrasonic_trigger_pin, GPIO.OUT)
GPIO.setup(Right_ultrasonic_echo_pin, GPIO.IN)
GPIO.setup(Left_ultrasonic_trigger_pin, GPIO.OUT)
GPIO.setup(Left_ultrasonic_echo_pin, GPIO.IN)


GPIO.setup(servo_pin, GPIO.OUT)


pwm = GPIO.PWM(enable_pin, 100)
servo_pwm = GPIO.PWM(servo_pin, 50)

#functions time
start_time = time.time()
loop_duration = (112)  # 2 minutes * 60 seconds/minute

# #######



#ultrasonic %%%#
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

# servo motor $
def control_servo(angle):
    if angle < angle_min:
        angle = angle_min
    elif angle > angle_max:
        angle = angle_max

    duty_cycle = angle / 18 + 2
    servo_pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.3)
    


def main():
    GPIO.output(input1_pin, GPIO.LOW)
    GPIO.output(input2_pin, GPIO.HIGH)
    pwm.ChangeDutyCycle(100)
    
    try:
      
        
        
        servo_pwm.start(0)
        #time.sleep(1)
        control_servo(dz)
        time.sleep(0.01)
        pwm.start(sa)
            
        while True :
       
            Direction = counterclockwise
            elapsed_time = time.time() - start_time
            if elapsed_time >= loop_duration:
                pwm.stop()
                print("BREAK")
                break  
            f1=measure_distance(23,24)
            r1=measure_distance(27, 22)
            l1=measure_distance(5, 6)
            print(f1,r1,l1)
            GPIO.output(input1_pin, GPIO.LOW)
            GPIO.output(input2_pin, GPIO.HIGH)
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
            if shared_data == "Red is present":
                print("Red is present")
                if Direction == counterclockwise:
                    control_servo(dl)#lift
                else:
                    control_servo(dr)#right
                pwm.start(70)
                time.sleep(1.4)
                control_servo(dz)#zero
                
            elif shared_data == "Green is present":
                print("Green is present")
                if Direction == counterclockwise:
                    control_servo(dr)#right
                else:
                    control_servo(dl)#lift
                pwm.start(70)
                time.sleep(1.4)
                control_servo(dz)#zero
            
            if f1<45 and (r1>150 or l1>150):
                print("f1<45")
                #pwm.start(40)
                #time.sleep(0.1)
                pwm.start(40) 
                GPIO.output(input1_pin, GPIO.HIGH)
                GPIO.output(input2_pin, GPIO.LOW)
                pwm.start(70)
                if r1<50 and r1 < l1:
                    print("r1<50 rvers")
                    control_servo(dl)#lift
                    pwm.start(70)
                    time.sleep(1.2)
                    control_servo(dz)#zero
                    
                elif l1<50 and l1<r1:
                    print("l1<50 revers")
                    control_servo(dr)#right
                    pwm.start(70)
                    time.sleep(2)
                    control_servo(dz)#zero
                elif r1<l1 and l1 < 80:
                    print("r1<l1111 rvers")
                    control_servo(dl)#left
                    pwm.start(70)
                    time.sleep(2)
                    control_servo(dz)#zero
                    
                elif l1<r1 and r1 < 80:
                    print("l1<r1111 revers")
                    control_servo(dr)#right
                    pwm.start(70)
                    time.sleep(2)
                    control_servo(dz-2)#zero            
                GPIO.output(input1_pin, GPIO.LOW)
                GPIO.output(input2_pin, GPIO.HIGH)
            f1=measure_distance(23,24)
            #r1=measure_distance(27, 22)
            #l1=measure_distance(5, 6)
            if f1<40:
                
                print("f1<40")
                #pwm.start(40)
                #time.sleep(0.1)
                pwm.start(40) 
                GPIO.output(input1_pin, GPIO.HIGH)
                GPIO.output(input2_pin, GPIO.LOW)
                pwm.start(60)
                if r1<50:
                    print("r1<50 rvers")
                    control_servo(dr)#right
                    pwm.start(80)
                    time.sleep(1)
                    control_servo(dz-2)#zero
                    
                elif l1<50:
                    print("l1<50 revers")
                    control_servo(dl)#lift
                    pwm.start(80)
                    time.sleep(1)
                    control_servo(dz)#zero
                elif l1<r1:
                    print("l1<r1 revers")
                    control_servo(dl)#lift
                    pwm.start(70)
                    time.sleep(1)
                    control_servo(dz)#zero
                elif r1<l1:
                    print("r1<l1 rvers")
                    control_servo(dr)#right
                    pwm.start(70)
                    time.sleep(1)
                    control_servo(dz-2)#zero
                #time.sleep(2)
                pwm.start(50)
                time.sleep(0.2)
                pwm.start(10)
                GPIO.output(input1_pin, GPIO.LOW)
                GPIO.output(input2_pin, GPIO.HIGH)
                pwm.start(sa)
# zero(37) # lift(20) #ri(46)
            if r1<9:
                pwm.start(40) 
                GPIO.output(input1_pin, GPIO.HIGH)
                GPIO.output(input2_pin, GPIO.LOW)
                print("r1<9")
                control_servo(dr)#right
                pwm.start(60)
                time.sleep(2)
                control_servo(dz-2)#zero
                pwm.start(50)
                time.sleep(0.2)
                pwm.start(10)
                GPIO.output(input1_pin, GPIO.LOW)
                GPIO.output(input2_pin, GPIO.HIGH)
                #pwm.start(50)
                    
            elif l1<9:
                #pwm.start(40) 
                GPIO.output(input1_pin, GPIO.HIGH)
                GPIO.output(input2_pin, GPIO.LOW)
                print("l1<9")
                control_servo(dl)#lift
                pwm.start(60)
                time.sleep(1) 
   #         if l2>15:
                control_servo(dz)#zero
                #pwm.start(50)
                time.sleep(0.2)
                #pwm.start(10)
                GPIO.output(input1_pin, GPIO.LOW)
                GPIO.output(input2_pin, GPIO.HIGH)
#                    pwm.start(60)
            
            elif r1<30:
                #pwm.start(40) 
                #GPIO.output(input1_pin, GPIO.HIGH)
                #GPIO.output(input2_pin, GPIO.LOW)
                print("r1<30")
                control_servo(dl)#lift
                #pwm.start(60)
                time.sleep(1.3)
                control_servo(dz)#zero
                #pwm.start(50)
                #time.sleep(0.2)
                #pwm.start(10)
                #GPIO.output(input1_pin, GPIO.LOW)
                #GPIO.output(input2_pin, GPIO.HIGH)
                #pwm.start(50)
                    
            elif l1<30:
                #pwm.start(40) 
                #GPIO.output(input1_pin, GPIO.HIGH)
               # GPIO.output(input2_pin, GPIO.LOW)
                print("l1<30")
                control_servo(dr)#lift
                #pwm.start(60)
                time.sleep(1.3) 
   #         if l2>15:
                control_servo(dz)#zero
                #pwm.start(50)
                #time.sleep(0.2)
                #pwm.start(10)
                #GPIO.output(input1_pin, GPIO.LOW)
                #GPIO.output(input2_pin, GPIO.HIGH)
            elif f1<85 and l1>200 and r1<80:
                
                pwm.start(50)
                print("l1>200")
                control_servo(dl)#lift
                time.sleep(1.5)
                control_servo(dz)#zero
                
            elif f1<85 and r1>200 and l1<80:
                
                pwm.start(50)
                print("r1>200")
                control_servo(dr)#right
                time.sleep(1.5)
                control_servo(dz-2)#
                
                
            
            elif f1<85 and l1<45 :
                pwm.start(50)
                print("rrr")
                control_servo(dr)#right
                time.sleep(1.1)
                control_servo(dz-2)#zero
            elif f1<85 and r1<45:
                pwm.start(50)
                print("lll")
                control_servo(dl)#lift
                time.sleep(1.1)
                control_servo(dz)#
            
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
    except Exception as e:
    
        pwm.stop()
        GPIO.cleanup()
        print("An error occurred:", e)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        pwm.stop()
        GPIO.cleanup()
main()
