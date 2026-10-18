"""
self.? - self is like this.? in JAVA

motor_driver_L298n.py
===============
Driver for L298n dual H-bridge motor controller.
All directions are from behind.
Controls 4 DC motors:
  - MOT1 + MOT2  →  Right side  (Channel A)
  - MOT3 + MOT4  →  Left side (Channel B)

Wiring (BCM numbering):
  IN1  → GPIO5   | IN2  → GPIO6   (Channel A direction)
  ENA  → GPIO18  (Channel A PWM speed)
  IN3  → GPIO13  | IN4  → GPIO16  (Channel B direction)
  ENB  → GPIO19  (Channel B PWM speed)
"""

import RPi.GPIO as GPIO
import time

# ────────────────────────────────────────────────────

# Pin definitions (BCM) 
IN1 = 5
IN2 = 6
IN3 = 13
IN4 = 16
ENA = 18   # PWM – right motors (Channel A)
ENB = 19   # PWM – left motors (Channel B)

PWM_FREQUENCY = 1000   # Hz

# ────────────────────────────────────────────────────

class MotorDriverL298n:
    """
    Simple driver for the L298n module.

    speed values are always 0–100 (percent duty cycle).
    """

    def __init__(self, default_speed: int = 50):
        """
        Initialise GPIO and PWM channels.
        """
        self.default_speed = self._clamp(default_speed)

        # GPIO setup
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        for pin in (IN1, IN2, IN3, IN4, ENA, ENB):
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

        # PWM setup
        self._pwm_a = GPIO.PWM(ENA, PWM_FREQUENCY)
        self._pwm_b = GPIO.PWM(ENB, PWM_FREQUENCY)
        self._pwm_a.start(0)
        self._pwm_b.start(0)

        print("[MotorDriver] Initialised — GPIO BCM mode, PWM @ "f"{PWM_FREQUENCY} Hz")

    # ────────────────────────────────────────────────────

    @staticmethod
    def _clamp(value: int, lo: int = 0, hi: int = 100) -> int:
        """Clamp an integer to [lo, hi]."""
        return max(lo, min(hi, value))

    def _set_speed(self, right: int, left: int) -> None:
        """Apply PWM duty cycles for right and left channels."""
        self._pwm_a.ChangeDutyCycle(self._clamp(right))
        self._pwm_b.ChangeDutyCycle(self._clamp(left))

    def _set_direction(self,
                       in1: bool, in2: bool,
                       in3: bool, in4: bool) -> None:
        """Write direction bits to IN1-IN4."""
        GPIO.output(IN1, GPIO.HIGH if in1 else GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH if in2 else GPIO.LOW)
        GPIO.output(IN3, GPIO.HIGH if in3 else GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH if in4 else GPIO.LOW)

    # ── Public movement API ───────────────────────────────────────────────

    def forward(self, speed: int = None) -> None:
        """
        Channel A (right): IN1=0, IN2=1 
        Channel B (left) : IN3=0, IN4=1
        """   
        spd = self._clamp(speed if speed is not None else self.default_speed)
        self._set_direction(False, True, False, True)
        self._set_speed(spd, spd)
        print(f"[MotorDriverL298n] FORWARD  speed={spd}%")

    def backward(self, speed: int = None) -> None:  
        """
        Channel A (right): IN1=1, IN2=0 
        Channel B (left) : IN3=1, IN4=0
        """
        spd = self._clamp(speed if speed is not None else self.default_speed)
        self._set_direction(True, False, True, False)
        self._set_speed(spd, spd)
        print(f"[MotorDriverL298n] BACKWARD  speed={spd}%")
        
    def turn_right(self, speed: int = None) -> None:
        """
        Channel A (right): IN1=1, IN2=0 
        Channel B (left) : IN3=0, IN4=1
        """
        spd = self._clamp(speed if speed is not None else self.default_speed)
        self._set_direction(True, False, False, True)
        self._set_speed(spd, spd)
        print(f"[MotorDriverL298n] RIGHT speed={spd}%") 
        
    def turn_left(self, speed: int = None) -> None:
        """
        Channel A (right): IN1=0, IN2=1 
        Channel B (left) : IN3=1, IN4=0
        """
        spd = self._clamp(speed if speed is not None else self.default_speed)
        self._set_direction(False, True, True, False)
        self._set_speed(spd, spd)
        print(f"[MotorDriverL298n] LEFT speed={spd}%") 

    def stop(self) -> None:
        """
        Channel A (right): IN1=0, IN2=0 
        Channel B (left) : IN3=0, IN4=0
        """
        self._set_direction(False, False, False, False)
        self._set_speed(0, 0)
        print("[MotorDriver] STOP")

    def brake(self) -> None:
        """
        Channel A (right): IN1=1, IN2=1 
        Channel B (left) : IN3=1, IN4=1
        """
        self._set_direction(True, True, True, True)
        self._set_speed(100, 100)
        print("[MotorDriver] BRAKE")

    def set_speed(self, right: int, left: int) -> None:
        """
        Independent speed control for each side — useful for gentle curves.

        :param left:  duty cycle % for left motors  (0–100)
        :param right: duty cycle % for right motors (0–100)
        """
        self._set_speed(right, left)
        print(f"[MotorDriver] SET_SPEED left={left}% right={right}%")

    def curve(self, action: str, speed: int = None) -> None:
        """
        Diagonal drive, one side at 1/4 speed — the app's G/H/I/J buttons:
          G = forward-left    H = forward-right
          I = backward-left   J = backward-right
        Shared by bluetooth_driver.py (manual drive) and auto_drive.py
        (replay), so a recorded curve is replayed exactly as it was driven.
        """
        spd  = self._clamp(speed if speed is not None else self.default_speed)
        slow = spd // 4
        if action in ('G', 'H'):
            self._set_direction(False, True, False, True)    # both sides forward
        else:
            self._set_direction(True, False, True, False)    # both sides backward
        if action in ('G', 'I'):
            self.set_speed(right=spd, left=slow)             # left side slower
        else:
            self.set_speed(right=slow, left=spd)             # right side slower

    def drive(self, left: float, right: float) -> None:
        """
        Each side on its own, -100 .. +100 (minus = backward) — "tank" steering.
        Used by ml_driver.py in CRUISE mode (joystick -> two motor speeds).
        """
        left, right = max(-100.0, min(100.0, left)), max(-100.0, min(100.0, right))
        # Channel A (right): forward IN1=0 IN2=1, backward IN1=1 IN2=0 ; 0 = both low
        # Channel B (left) : forward IN3=0 IN4=1, backward IN3=1 IN4=0
        self._set_direction(right < 0, right > 0, left < 0, left > 0)
        self._set_speed(int(abs(right)), int(abs(left)))

    def cleanup(self) -> None:
        """Stop motors and release all GPIO resources."""
        self.stop()
        self._pwm_a.stop()
        self._pwm_b.stop()
        GPIO.cleanup()
        print("[MotorDriver] GPIO cleaned up")

    # ── Self Test API ───────────────────────────────────────────────
            
    def self_test(self) -> None:
        """Test all functions of this driver"""
        print("\nMOTOR DRIVER L298n - Self test")
        
        self.forward()
        time.sleep(2)
        self.stop()
        time.sleep(2)
        
        self.backward()
        time.sleep(2)
        self.stop()
        time.sleep(2) 
        
        self.turn_right()
        time.sleep(2)
        self.stop()
        time.sleep(2)  
        
        self.turn_left()
        time.sleep(2)
        self.stop()
        time.sleep(2)  
        
        self.brake()
        time.sleep(2)
        self.stop()   
                                      
        print("\nMOTOR DRIVER L298n - Self test - Done")
