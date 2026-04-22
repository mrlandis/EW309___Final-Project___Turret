import usb_cdc
import board
import busio
import adafruit_bno055
import time
import pwmio
import digitalio

def wrapToPi(ang):
    while(ang > 180.0): ang = ang - 360.0
    while(ang < -180.0): ang = ang + 360.0
    return ang

i2c = busio.I2C(board.GP3, board.GP2)
time.sleep(0.5)
sensor = adafruit_bno055.BNO055_I2C(i2c)

i2c0 = busio.I2C(board.GP1, board.GP0, frequency=400000)
INA260_ADDRESS = 0x40
INA260_CURRENT_REGISTER = 0x01

def read_current():
    """Read current from INA260."""
    try:
        while not i2c0.try_lock():
            pass
        i2c0.writeto(INA260_ADDRESS, bytes([INA260_CURRENT_REGISTER]))
        result = bytearray(2)
        i2c0.readfrom_into(INA260_ADDRESS, result)
        i2c0.unlock()
        raw_current = int.from_bytes(result, "big")
        if raw_current & 0x8000:
            raw_current -= 1 << 16
        return raw_current * 1.25 
    except OSError:
        return 0

loader = digitalio.DigitalInOut(board.GP14)
loader.direction = digitalio.Direction.OUTPUT

launcher = digitalio.DigitalInOut(board.GP15)
launcher.direction = digitalio.Direction.OUTPUT
launcher.value = True

class Turret:
    def __init__(self, direction, value):
        self.left = pwmio.PWMOut(board.GP9, frequency=1000)
        self.right = pwmio.PWMOut(board.GP10, frequency=1000)
        self.down = pwmio.PWMOut(board.GP12, frequency=1000)
        self.up = pwmio.PWMOut(board.GP13, frequency=1000)
        self.direction = direction
        self.value = max(-1.0, min(1.0, value))
        
    def stop(self):
        self.left.duty_cycle = 0
        self.right.duty_cycle = 0
        self.up.duty_cycle = 0
        self.down.duty_cycle = 0

    def pitch(self,speed):
        speed = max(-1.0, min(1.0, speed))
        pwm = int(abs(speed) * 65535)
        self.up.duty_cycle = pwm if speed > 0 else 0
        self.down.duty_cycle = pwm if speed < 0 else 0
       
    def yaw(self,speed1):
        speed1 = max(-1.0, min(1.0, speed1))
        pwm = int(abs(speed1) * 65535)
        self.left.duty_cycle = pwm if speed1 < 0 else 0
        self.right.duty_cycle = pwm if speed1 > 0 else 0

nar = Turret('up', 0)

# ========================================================
# MAIN LOOP
# ========================================================
while True:
    print("Waiting for data from PC...")
    
    yaw = pitch = yaw1 = pitch1 = 0.0
    received_valid_targets = False

    # 1. Safely wait for NEW coordinates, trashing old ones if a handshake occurs
    while not received_valid_targets:
        if usb_cdc.data and usb_cdc.data.in_waiting > 0:
            inMsg = usb_cdc.data.readline().decode().strip()
            
            # If the PC script restarted, it sends 'handshake'
            if inMsg == "handshake":
                print(">> PC Restarted. Flushing old commands...")
                usb_cdc.data.reset_input_buffer()  # Destroy any trapped 'p' commands
                usb_cdc.data.write(b"ready\n")
                continue
                
            angles = inMsg.split(',')
            if len(angles) == 4:
                rel_yaw1 = float(angles[0])
                rel_pitch1 = float(angles[1])
                rel_yaw2 = float(angles[2])
                rel_pitch2 = float(angles[3])
                
                # Get current baseline from IMU
                base_yaw_raw = None
                while base_yaw_raw is None:
                    euler = sensor.euler
                    base_yaw_raw = euler[0]
                    base_pitch = euler[2]
                    time.sleep(0.01)
                    
                base_yaw = wrapToPi(base_yaw_raw)
                
                # Add camera's relative offset to the absolute heading
                yaw = wrapToPi(base_yaw + rel_yaw1)
                pitch = base_pitch + rel_pitch1
                yaw1 = wrapToPi(base_yaw + rel_yaw2)
                pitch1 = base_pitch + rel_pitch2
                
                print(f"Base: Yaw {base_yaw:.2f}, Pitch {base_pitch:.2f}")
                print(f"Target 1 ABS: {yaw:.2f}, {pitch:.2f} | Target 2 ABS: {yaw1:.2f}, {pitch1:.2f}")
                received_valid_targets = True
        
        time.sleep(0.01)

    # ========================================================
    # TARGET 1 ENGAGEMENT
    # ========================================================
    kp_yaw = 0.175
    ki_yaw = 0.0172
    kd_yaw = 0.01   

    kp_pitch = 0.115
    ki_pitch = 0.00167
    kd_pitch = 0.01

    integral_yaw = 0
    integral_pitch = 0
    integral_limit = 50
    runtime = 3.5
    deadzone = 0.18

    last_error_yaw = 0.0
    last_error_pitch = 0.0

    start_time = time.monotonic()
    last_time = start_time 

    while time.monotonic() - start_time < runtime:
        now = time.monotonic()
        dt = max(now - last_time, 0.001) 
        last_time = now

        euler = sensor.euler
        if euler[0] is None: 
            continue 
            
        current_yaw = wrapToPi(euler[0])
        current_pitch = euler[2]

        error_yaw = wrapToPi(yaw - current_yaw)
        error_pitch = pitch - current_pitch

        integral_yaw = max(-integral_limit, min(integral_limit, integral_yaw + error_yaw * dt))
        integral_pitch = max(-integral_limit, min(integral_limit, integral_pitch + error_pitch * dt))

        derivative_yaw = (error_yaw - last_error_yaw) / dt
        derivative_pitch = (error_pitch - last_error_pitch) / dt

        last_error_yaw = error_yaw
        last_error_pitch = error_pitch

        control_signal_yaw = (kp_yaw * error_yaw) + (ki_yaw * integral_yaw) + (kd_yaw * derivative_yaw)
        control_signal_pitch = (kp_pitch * error_pitch) + (ki_pitch * integral_pitch) + (kd_pitch * derivative_pitch)

        control_signal_yaw = max(-1, min(1.0, control_signal_yaw))
        if abs(control_signal_yaw) < deadzone: control_signal_yaw = 0

        control_signal_pitch = max(-1, min(1.0, control_signal_pitch))
        if abs(control_signal_pitch) < deadzone: control_signal_pitch = 0

        nar.yaw(control_signal_yaw)
        nar.pitch(control_signal_pitch)
        time.sleep(0.05)
        
    nar.stop()

    # Fire Target 1
    loader.value = True
    i = 0
    while True:
        read = read_current()
        if read > 2970:
            loader.value = False
            time.sleep(0.25)
            loader.value = True
            i = i + 1
        if i >= 2:
            loader.value = False
            break

    # ========================================================
    # TARGET 2 ENGAGEMENT
    # ========================================================
    kp_yaw = 0.175
    ki_yaw = 0.00172
    kp_pitch = 0.115
    ki_pitch = 0.00167

    integral_yaw = 0
    integral_pitch = 0
    last_error_yaw = 0.0  
    last_error_pitch = 0.0
    last_time = time.monotonic()
    start_time = time.monotonic()

    while time.monotonic() - start_time < runtime:
        now = time.monotonic()
        dt = max(now - last_time, 0.001) 
        last_time = now

        euler = sensor.euler
        if euler[0] is None: 
            continue
            
        current_yaw = wrapToPi(euler[0])
        current_pitch = euler[2]

        error_yaw = wrapToPi(yaw1 - current_yaw)
        error_pitch = pitch1 - current_pitch

        integral_yaw = max(-integral_limit, min(integral_limit, integral_yaw + error_yaw * dt))
        integral_pitch = max(-integral_limit, min(integral_limit, integral_pitch + error_pitch * dt))

        derivative_yaw = (error_yaw - last_error_yaw) / dt
        derivative_pitch = (error_pitch - last_error_pitch) / dt

        last_error_yaw = error_yaw
        last_error_pitch = error_pitch

        control_signal_yaw = (kp_yaw * error_yaw) + (ki_yaw * integral_yaw) + (kd_yaw * derivative_yaw)
        control_signal_pitch = (kp_pitch * error_pitch) + (ki_pitch * integral_pitch) + (kd_pitch * derivative_pitch)

        control_signal_yaw = max(-1, min(1.0, control_signal_yaw))
        if abs(control_signal_yaw) < deadzone: control_signal_yaw = 0

        control_signal_pitch = max(-1, min(1.0, control_signal_pitch))
        if abs(control_signal_pitch) < deadzone: control_signal_pitch = 0

        nar.yaw(control_signal_yaw)
        nar.pitch(control_signal_pitch)
        time.sleep(0.05)
       
    nar.stop()

    # Fire Target 2
    loader.value = True
    i = 0
    while True:
        read = read_current()
        if read > 2970:
            loader.value = False
            time.sleep(0.25)
            loader.value = True
            i = i + 1
        if i >= 2:
            loader.value = False
            break

    # ========================================================
    # TRIAL CLEANUP
    # ========================================================
    print("Trial Complete. Resetting...")
    if usb_cdc.data:
        usb_cdc.data.reset_input_buffer()  # Flush buffer one final time before looping
    
    # Do NOT turn off the launcher here, or you will have to wait for the flywheels 
    # to spin back up on your next trial!