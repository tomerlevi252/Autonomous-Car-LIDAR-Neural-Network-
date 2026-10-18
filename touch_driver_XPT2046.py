"""
touch_driver_XPT2046.py
=======================
Driver for XPT2046 touch controller over SPI.

Wiring (BCM numbering):
  MOSI  → GPIO10  (shared with ILI9341)
  MISO  → GPIO9   (shared with ILI9341)
  SCK   → GPIO11  (shared with ILI9341)
  T_CS  → GPIO7   (SPI0 CE1)
  T_IRQ → GPIO17
"""

import spidev
import RPi.GPIO as GPIO
import time

# ── Pin definitions (BCM) ─────────────────────────────────────
T_CS  = 7
T_IRQ = 17

# ── Display dimensions (for coordinate mapping) ───────────────
WIDTH  = 320
HEIGHT = 240

# ── Calibration values (measured from this screen) ───────────
X_MIN = 564
X_MAX = 3288
Y_MIN = 504
Y_MAX = 3532


class TouchDriverXPT2046:
    """Driver for XPT2046 touch controller over SPI."""

    def __init__(self):
        """Initialise SPI and GPIO for XPT2046."""

        # GPIO setup
        GPIO.setwarnings(False)
        GPIO.setup(T_IRQ, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        # SPI setup — CE1 (GPIO7) for touch
        self._spi = spidev.SpiDev()
        self._spi.open(0, 1)           # bus 0, device 1 (CE1 = GPIO7)
        self._spi.max_speed_hz = 500_000
        self._spi.mode = 0

        print("[TouchDriverXPT2046] Initialised — SPI CE1, IRQ GPIO17")

    # ── Internal helpers ──────────────────────────────────────

    def _read_raw(self, cmd: int) -> int:
        """Send command and read 12-bit raw value from XPT2046."""
        r = self._spi.xfer2([cmd, 0x00, 0x00])
        return ((r[1] << 8) | r[2]) >> 3

    def _read_averaged(self, cmd: int, samples: int = 5) -> int:
        """Read multiple samples and return the average (reduces noise)."""
        readings = [self._read_raw(cmd) for _ in range(samples)]
        readings.sort()
        trimmed = readings[1:-1]
        return sum(trimmed) // len(trimmed)

    # ── Public API ────────────────────────────────────────────

    def is_touched(self) -> bool:
        """Return True if the screen is currently being touched."""
        return GPIO.input(T_IRQ) == GPIO.LOW

    def read_raw(self):
        """
        Read raw X/Y values from the touch controller.
        :return: (raw_x, raw_y) or None if not touched
        """
        if not self.is_touched():
            return None
        x = self._read_averaged(0xD0)  # X position command
        y = self._read_averaged(0x90)  # Y position command
        return (x, y)

    def read_position(self):
        """
        Read touch position mapped to screen coordinates (0-320, 0-240).
        :return: (x, y) in screen pixels, or None if not touched
        """
        raw = self.read_raw()
        if raw is None:
            return None

        raw_x, raw_y = raw

        # Map raw values to screen coordinates — swap X/Y for landscape
        x = int((raw_y - Y_MIN) * WIDTH  / (Y_MAX - Y_MIN))
        y = int((raw_x - X_MIN) * HEIGHT / (X_MAX - X_MIN))
        x = WIDTH  - 1 - x  # flip X axis

        # Clamp to screen bounds
        x = max(0, min(WIDTH  - 1, x))
        y = max(0, min(HEIGHT - 1, y))

        return (x, y)

    def wait_for_touch(self, timeout: float = 0) -> tuple:
        """
        Wait until screen is touched and released.
        :param timeout: seconds to wait (0 = wait forever)
        :return: (x, y) position or None if timeout
        """
        start = time.time()
        while True:
            if self.is_touched():
                time.sleep(0.02)  # debounce
                pos = self.read_position()
                if pos:
                    # Wait for release
                    while self.is_touched():
                        time.sleep(0.01)
                    time.sleep(0.05)  # debounce after release
                    return pos
            if timeout > 0 and (time.time() - start) > timeout:
                return None
            time.sleep(0.01)

    def cleanup(self) -> None:
        """Release SPI resources."""
        self._spi.close()
        print("[TouchDriverXPT2046] Cleaned up.")

    # ── Self Test ─────────────────────────────────────────────

    def self_test(self, duration: int = 15) -> None:
        """
        Print touch coordinates for 'duration' seconds.
        Touch the screen to see X/Y values printed.
        """
        print(f"\n[TouchDriverXPT2046] --- Self-test START ({duration} sec) ---")
        print("[TouchDriverXPT2046] Touch the screen to see coordinates...")

        start = time.time()
        while time.time() - start < duration:
            if self.is_touched():
                pos = self.read_position()
                raw = self.read_raw()
                if pos and raw:
                    print(f"  Touch: screen=({pos[0]:3d}, {pos[1]:3d})  "
                          f"raw=({raw[0]:4d}, {raw[1]:4d})")
                time.sleep(0.1)

        print("[TouchDriverXPT2046] --- Self-test COMPLETE ---\n")


# ── Run directly ──────────────────────────────────────────────
if __name__ == "__main__":
    GPIO.setmode(GPIO.BCM)
    touch = TouchDriverXPT2046()
    try:
        touch.self_test(duration=15)
    except KeyboardInterrupt:
        print("\n[TouchDriverXPT2046] Interrupted.")
    finally:
        touch.cleanup()
        GPIO.cleanup()
