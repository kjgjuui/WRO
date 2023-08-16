import time
from Adafruit_PCA9685 import PCA9685

# Initialize the PCA9685
pwm = PCA9685()
pwm.set_pwm_freq(50)  # Set the PWM frequency (you can adjust this as needed)

# Define the servo motor channels
servo_channel = 0  # Use the appropriate channel for your servo motor

# Function to set the servo motor position
def set_servo_position(channel, position):
    pulse_length = 1000000  # 1,000,000 us per second
    pulse_length //= 50     # 50 Hz (20 ms per cycle)
    pulse_length //= 4096   # 12 bits of resolution (4096 steps)
    pulse = int(pulse_length * position)
    pwm.set_pwm(channel, 0, pulse)

try:
    while True:
        # Move the servo to the middle position
        set_servo_position(servo_channel, 0.5)
        time.sleep(1)

        # Move the servo to the full left position
        set_servo_position(servo_channel, 0)
        time.sleep(1)

        # Move the servo to the full right position
        set_servo_position(servo_channel, 1)
        time.sleep(1)

except KeyboardInterrupt:
    # Clean up on keyboard interrupt
    pwm.set_all_pwm(0, 0)
