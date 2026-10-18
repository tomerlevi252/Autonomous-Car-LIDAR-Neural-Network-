"""
display_driver_ILI9341.py
=========================
Driver for ILI9341 TFT display (320x240, RGB565, SPI).

Wiring (BCM numbering):
  MOSI  → GPIO10
  MISO  → GPIO9
  SCK   → GPIO11
  DC    → GPIO22
  RESET → GPIO27
  CS    → GPIO8  (SPI0 CE0)
"""

import spidev
import RPi.GPIO as GPIO
import time
from PIL import Image

# ── Pin definitions (BCM) ─────────────────────────────────────
DC    = 22
RESET = 27
CS    = 8

# ── Display dimensions ────────────────────────────────────────
WIDTH  = 320
HEIGHT = 240

# ── ILI9341 commands ──────────────────────────────────────────
ILI9341_SWRESET    = 0x01
ILI9341_SLPOUT     = 0x11
ILI9341_DISPON     = 0x29
ILI9341_CASET      = 0x2A
ILI9341_PASET      = 0x2B
ILI9341_RAMWR      = 0x2C
ILI9341_MADCTL     = 0x36
ILI9341_COLMOD     = 0x3A
ILI9341_FRMCTR1    = 0xB1
ILI9341_DFUNCTR    = 0xB6
ILI9341_PWCTR1     = 0xC0
ILI9341_PWCTR2     = 0xC1
ILI9341_VMCTR1     = 0xC5
ILI9341_VMCTR2     = 0xC7
ILI9341_GMCTRP1    = 0xE0
ILI9341_GMCTRN1    = 0xE1


class DisplayDriverILI9341:
    """Driver for ILI9341 320x240 TFT display over SPI."""

    def __init__(self):
        """Initialise SPI, GPIO and the ILI9341 controller."""

        # GPIO setup
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(DC,    GPIO.OUT)
        GPIO.setup(RESET, GPIO.OUT)

        # SPI setup — 64MHz for faster screen updates
        self._spi = spidev.SpiDev()
        self._spi.open(0, 0)          # bus 0, device 0 (CE0 = GPIO8)
        self._spi.max_speed_hz = 64_000_000
        self._spi.mode = 0

        # Initialise display
        self._reset()
        self._init_display()

        print("[DisplayDriverILI9341] Initialised — 320x240 RGB565 @ 64MHz")

    # ── Internal helpers ──────────────────────────────────────

    def _reset(self) -> None:
        """Hardware reset the display."""
        GPIO.output(RESET, GPIO.HIGH)
        time.sleep(0.01)
        GPIO.output(RESET, GPIO.LOW)
        time.sleep(0.02)
        GPIO.output(RESET, GPIO.HIGH)
        time.sleep(0.15)

    def _write_cmd(self, cmd: int) -> None:
        """Send a command byte."""
        GPIO.output(DC, GPIO.LOW)
        self._spi.writebytes([cmd])

    def _write_data(self, data) -> None:
        """Send data bytes (list or single int)."""
        GPIO.output(DC, GPIO.HIGH)
        if isinstance(data, int):
            self._spi.writebytes([data])
        else:
            data = list(data)
            for i in range(0, len(data), 4096):
                self._spi.writebytes(data[i:i + 4096])

    def _write_buf(self, buf: bytes) -> None:
        """Send a raw bytes buffer — fastest method for pixel data."""
        GPIO.output(DC, GPIO.HIGH)
        self._spi.writebytes2(buf)

    def _init_display(self) -> None:
        """Send ILI9341 initialisation sequence."""
        self._write_cmd(ILI9341_SWRESET)
        time.sleep(0.15)

        self._write_cmd(ILI9341_PWCTR1);  self._write_data(0x23)
        self._write_cmd(ILI9341_PWCTR2);  self._write_data(0x10)
        self._write_cmd(ILI9341_VMCTR1);  self._write_data([0x3E, 0x28])
        self._write_cmd(ILI9341_VMCTR2);  self._write_data(0x86)

        # Memory access control — landscape mode (0xE8 = correct orientation)
        self._write_cmd(ILI9341_MADCTL);  self._write_data(0xE8)

        # 16-bit colour (RGB565)
        self._write_cmd(ILI9341_COLMOD);  self._write_data(0x55)

        self._write_cmd(ILI9341_FRMCTR1); self._write_data([0x00, 0x18])
        self._write_cmd(ILI9341_DFUNCTR); self._write_data([0x08, 0x82, 0x27])

        # Gamma
        self._write_cmd(ILI9341_GMCTRP1)
        self._write_data([0x0F,0x31,0x2B,0x0C,0x0E,0x08,
                          0x4E,0xF1,0x37,0x07,0x10,0x03,
                          0x0E,0x09,0x00])
        self._write_cmd(ILI9341_GMCTRN1)
        self._write_data([0x00,0x0E,0x14,0x03,0x11,0x07,
                          0x31,0xC1,0x48,0x08,0x0F,0x0C,
                          0x31,0x36,0x0F])

        self._write_cmd(ILI9341_SLPOUT)
        time.sleep(0.12)
        self._write_cmd(ILI9341_DISPON)

    def _set_window(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Set the drawing window on the display."""
        self._write_cmd(ILI9341_CASET)
        self._write_data([x0 >> 8, x0 & 0xFF, x1 >> 8, x1 & 0xFF])
        self._write_cmd(ILI9341_PASET)
        self._write_data([y0 >> 8, y0 & 0xFF, y1 >> 8, y1 & 0xFF])
        self._write_cmd(ILI9341_RAMWR)

    # ── Public API ────────────────────────────────────────────

    @staticmethod
    def color565(r: int, g: int, b: int) -> int:
        """Convert RGB888 to RGB565."""
        return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)

    def fill_screen(self, color: int) -> None:
        """Fill the entire screen with a RGB565 color — uses writebytes2 for speed."""
        self._set_window(0, 0, WIDTH - 1, HEIGHT - 1)
        hi = (color >> 8) & 0xFF
        lo = color & 0xFF
        buf = bytes([hi, lo] * (WIDTH * HEIGHT))
        self._write_buf(buf)
        print(f"[DisplayDriverILI9341] fill_screen color=0x{color:04X}")

    def show_image(self, image_path: str) -> None:
        """
        Display a image on screen.
        Image is resized to 320x240 and converted to RGB565 automatically.
        :param image_path: path to image file (jpg, png, bmp...)
        """
        img = Image.open(image_path).convert('RGB').resize((WIDTH, HEIGHT))
        pixels = img.load()

        buf = bytearray(WIDTH * HEIGHT * 2)
        idx = 0
        for y in range(HEIGHT):
            for x in range(WIDTH):
                r, g, b = pixels[x, y]
                color = self.color565(r, g, b)
                buf[idx]     = (color >> 8) & 0xFF
                buf[idx + 1] = color & 0xFF
                idx += 2

        self._set_window(0, 0, WIDTH - 1, HEIGHT - 1)
        self._write_buf(bytes(buf))
        print(f"[DisplayDriverILI9341] show_image: {image_path}")

    def cleanup(self) -> None:
        """Release SPI and GPIO resources."""
        self._spi.close()
        GPIO.cleanup()
        print("[DisplayDriverILI9341] Cleaned up.")

    # ── Self Test ─────────────────────────────────────────────

    def self_test(self) -> None:
        """Fill screen with red, green, blue, white, black — 2 seconds each."""
        print("\n[DisplayDriverILI9341] --- Self-test START ---")

        print("[DisplayDriverILI9341] RED")
        self.fill_screen(self.color565(255, 0, 0))
        time.sleep(2)

        print("[DisplayDriverILI9341] GREEN")
        self.fill_screen(self.color565(0, 255, 0))
        time.sleep(2)

        print("[DisplayDriverILI9341] BLUE")
        self.fill_screen(self.color565(0, 0, 255))
        time.sleep(2)

        print("[DisplayDriverILI9341] WHITE")
        self.fill_screen(self.color565(255, 255, 255))
        time.sleep(2)

        print("[DisplayDriverILI9341] BLACK")
        self.fill_screen(self.color565(0, 0, 0))
        time.sleep(1)

        print("[DisplayDriverILI9341] --- Self-test COMPLETE ---\n")


# ── Run directly ──────────────────────────────────────────────
if __name__ == "__main__":
    display = DisplayDriverILI9341()
    try:
        display.self_test()
    except KeyboardInterrupt:
        print("\n[DisplayDriverILI9341] Interrupted.")
    finally:
        display.cleanup()
