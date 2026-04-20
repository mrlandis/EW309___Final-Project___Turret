import time
import board
import busio
import pwmio
import digitalio
import math
import usb_cdc

# ==========================================
# 0. EASY CONFIGURATION
# ==========================================
DEADBAND_DEG = 0.25 
MIN_PWR_PAN = 26.0  
MIN_PWR_TILT = 28.0
MAX_PWR_PID = 50.0  

# TUNED FOR STABILITY (Low Gain, High Damping)
K_PAN, P_PAN, A_PAN = 40.0, 4.00, 5.0
K_TILT, P_TILT, A_TILT = 45.0, 4.00, 5.0

# ==========================================
# 1. Controller Class
# ==========================================
class TurretController:
    def __init__(self, K, p, a, min_pwr, max_pid_pwr, dt=0.016):
        self.K, self.p, self.a, self.dt = K, p, a, dt 
        self.min_pwr = min_pwr
        self.max_pid_pwr = max_pid_pwr
        self.u1, self.u2, self.e1 = 0.0, 0.0, 0.0 
        self.stuck_timer = 0.0
        self.last_e = 0.0
        self.stiction_bump = 0.0
        
    def update(self, e):
        # 1. MOVEMENT DETECTION (Strong brake when moving)
        if abs(e - self.last_e) > math.radians(0.25):
            self.stiction_bump *= 0.3 
            self.stuck_timer = 0.0
        elif abs(e) > math.radians(DEADBAND_DEG):
            self.stuck_timer += self.dt
        else:
            self.stuck_timer = 0.0
            self.stiction_bump = 0.0

        # 2. SLOW RAMP
        if self.stuck_timer > 0.15:
            error_deg = math.degrees(abs(e))
            self.stiction_bump = min(max(1.5, error_deg * 2.0), self.stiction_bump + 0.02)

        self.last_e = e

        u = (1 / (1 + self.p * self.dt)) * (
            -self.u1 * (-2 - self.p * self.dt) - self.u2 
            + e * (self.K * self.dt + self.K * self.a * (self.dt ** 2)) 
            + self.e1 * (-self.K * self.dt)
        )
        
        u = max(-20.0, min(20.0, u))
        self.u2, self.u1, self.e1 = self.u1, u, e
        
        if u > 0: u_dz = u + self.min_pwr + self.stiction_bump
        elif u < 0: u_dz = u - self.min_pwr - self.stiction_bump
        else: u_dz = 0.0
            
        return max(-self.max_pid_pwr, min(self.max_pid_pwr, u_dz))

# Hardware Setup
i2c0 = busio.I2C(board.GP1, board.GP0, frequency=400000)
loader = digitalio.DigitalInOut(board.GP14)
loader.direction = digitalio.Direction.OUTPUT
launcher = digitalio.DigitalInOut(board.GP15)
launcher.direction = digitalio.Direction.OUTPUT
launcher.value = False

class Turret:
    def __init__(self):
        self.left = pwmio.PWMOut(board.GP9, frequency=1000)
        self.right = pwmio.PWMOut(board.GP10, frequency=1000)
        self.down = pwmio.PWMOut(board.GP12, frequency=1000)
        self.up = pwmio.PWMOut(board.GP13, frequency=1000)
        
    def yaw(self, speed):
        pwm = int(abs(speed) * 65535)
        self.left.duty_cycle = pwm if speed < 0 else 0
        self.right.duty_cycle = pwm if speed > 0 else 0

    def pitch(self, speed):
        pwm = int(abs(speed) * 65535)
        self.up.duty_cycle = pwm if speed > 0 else 0
        self.down.duty_cycle = pwm if speed < 0 else 0

nar = Turret()
serial = usb_cdc.data
pan_ctrl = TurretController(K_PAN, P_PAN, A_PAN, MIN_PWR_PAN, MAX_PWR_PID)
tilt_ctrl = TurretController(K_TILT, P_TILT, A_TILT, MIN_PWR_TILT, MAX_PWR_PID)

yaw_err_rad, pitch_err_rad = 0.0, 0.0
last_track_time = time.monotonic()

while True:
    loop_start = time.monotonic()
    latest_msg = None
    if serial and serial.in_waiting > 0:
        while serial.in_waiting > 0:
            try: latest_msg = serial.readline().decode('utf-8').strip()
            except: pass
                
    if latest_msg:
        if latest_msg == "handshake": serial.write(b"ready\n")
        elif latest_msg.startswith("TRACK"):
            parts = latest_msg.split(',')
            yaw_err_rad = math.radians(float(parts[1]))
            pitch_err_rad = math.radians(float(parts[2]))
            last_track_time = time.monotonic()

    if time.monotonic() - last_track_time > 0.3:
        yaw_err_rad = pitch_err_rad = 0.0

    # Driving Motors
    nar.yaw(pan_ctrl.update(yaw_err_rad) / 100.0)
    nar.pitch(tilt_ctrl.update(pitch_err_rad) / 100.0)

    elapsed = time.monotonic() - loop_start
    if elapsed < 0.016: time.sleep(0.016 - elapsed)