"""
ui.py
=====
UI layer combining ILI9341 display + XPT2046 touch.
Style: Technological — dark blue, cyan accents, robotic feel.
"""

import glob
import os
import time
import math
import re
import threading
import subprocess
from PIL import Image, ImageChops, ImageDraw, ImageFont
from display_driver_ILI9341 import DisplayDriverILI9341
from touch_driver_XPT2046   import TouchDriverXPT2046
from scan_recorder          import ScanRecorder

# ── Display dimensions ────────────────────────────────────────
W, H = 320, 240

# ── Color palette — Technological ────────────────────────────
C_BG          = (5,   10,  25)
C_BG2         = (8,   20,  45)
C_CYAN        = (0,   220, 255)
C_CYAN_DIM    = (0,   100, 130)
C_BLUE        = (20,  80,  200)
C_BLUE_DIM    = (10,  40,  100)
C_RED         = (200, 30,  60)
C_RED_DIM     = (100, 15,  30)
C_GRAY        = (60,  70,  90)
C_WHITE       = (255, 255, 255)
C_BLACK       = (0,   0,   0)

# ── Button regions ────────────────────────────────────────────
BTN_BT_X,   BTN_BT_Y,   BTN_BT_W,   BTN_BT_H   = 15,  80,  135, 70
BTN_ML_X,   BTN_ML_Y,   BTN_ML_W,   BTN_ML_H   = 170, 80,  135, 70
BTN_BACK_X, BTN_BACK_Y, BTN_BACK_W, BTN_BACK_H = 85, 195, 150, 30

# AUTO DRIVE LIDAR MAP submenu — RECORDING ROUTE / CAR RUN, and below them TRAIN / CLEAR
BTN_RECORD_X, BTN_RECORD_Y, BTN_RECORD_W, BTN_RECORD_H = 15,  80,  135, 70
BTN_RUNML_X,  BTN_RUNML_Y,  BTN_RUNML_W,  BTN_RUNML_H  = 170, 80,  135, 70
BTN_TRAIN_X,  BTN_TRAIN_Y,  BTN_TRAIN_W,  BTN_TRAIN_H  = 60,  158, 200, 28
# V5: the CLEAR DATABASE button was removed (one bad drive: python3 route_database.py delete <n>)


# The neural network learns from MANY recorded drives ("runs", PDF section 8:
# Run 1 ... Run 20). The recording screen counts them; you can record more.
RECORDING_TARGET = 20

# CAR RUN: while the car drives by itself the screen shows the live LIDAR map
# (what the LIDAR sees), like on the recording screen. False = the old plain
# "DRIVING" text screen.
RUN_SHOW_LIDAR   = True
RUN_VIEW_PERIOD  = 0.25   # seconds between two pictures of the map while driving by itself

# A recorded action shorter than this is not a real hold (press + release
# handled back-to-back) and gets no row of its own in the DB.
RECORD_MIN_SEGMENT = 0.02

# Bluetooth screen's status box (the box that shows SEARCHING.../CONNECTED) —
# doubles as a button once connected, to jump to the LIDAR view.
BTN_STATUS_X, BTN_STATUS_Y, BTN_STATUS_W, BTN_STATUS_H = 30, 48, 260, 85

# ── LIDAR map geometry ────────────────────────────────────────
MAP_CX      = W // 2       
MAP_CY      = H // 2 + 10  
MAP_R       = 95           
CAR_W       = 16
CAR_H       = 28


class UI:
    """UI manager — display + touch combined."""

    def __init__(self, display: DisplayDriverILI9341,
                       touch:   TouchDriverXPT2046):
        self._display        = display
        self._touch          = touch
        self._bt_driver      = None
        self._lidar          = None
        self._lidar_thread   = None
        self._lidar_running  = False
        self._bt_connected   = False

        # RECORDING ROUTE (AUTO DRIVE > A) — driven by the app's HORN button
        # over Bluetooth (toggle: odd press starts, even press stops, saves
        # it and bumps the "DONE: X / 5" counter).
        self._recording_count         = 0
        self._recording_active        = False
        self._recording_screen_active = False   # guards the callbacks below
        self._route_db                = None
        self._recording_session_id    = None

        # Event-based segment currently being timed — the action last sent
        # over Bluetooth and the moment it started, so when the NEXT action
        # arrives (or recording stops) we know exactly how long it was held.
        self._record_segment_action = None
        self._record_segment_start  = None
        self._record_segment_zones  = None   # LIDAR 8-zone snapshot taken when the segment started

        # Finished segments of the recording in progress. They are kept in
        # memory while driving and written to the DB in one go when the
        # recording stops (see _save_recording) — no disk I/O on the
        # Bluetooth thread while the car is being driven.
        self._record_buffer     = []
        self._record_started_at = None
        self._record_lock       = threading.Lock()

        # TRAIN (AUTO DRIVE > TRAIN) — trains the neural network on the Pi (ml_train.py)
        self._training       = False
        self._train_lines    = []
        self._train_result   = None     # None while running / not started, else (ok, text)
        self._recordings_total = 0      # scans/scan_REC_*.csv.gz files = training runs

        # CAR RUN (AUTO DRIVE > B) — the neural network drives (auto_drive.py NAV_MODE='ml').
        self._auto_drive     = None
        self._run_ml_started = False   # True only once a replay actually began
        self._nav_text       = ""      # the latest line from the autonomous driving

        # Raw LIDAR log of every drive (RECORDING ROUTE and CAR RUN) -> scans/
        # It only listens to the LIDAR; it changes nothing in how the car drives.
        self._scan_rec = ScanRecorder()

        print("[UI] Initialised")

    def set_bluetooth_driver(self, bt_driver) -> None:
        self._bt_driver = bt_driver
        if bt_driver is not None:
            # HORN on the phone: start/stop a recording, or (V5) stop the car during CAR RUN
            bt_driver.set_on_horn(self._on_horn_pressed)

    def set_lidar_driver(self, lidar) -> None:
        self._lidar = lidar
        if lidar is not None and hasattr(lidar, 'set_on_packet'):
            lidar.set_on_packet(self._scan_rec.add_packet)

    def set_route_database(self, route_db) -> None:
        """RouteDatabase instance — stores the actual RECORDING ROUTE samples."""
        self._route_db = route_db

    def set_auto_drive(self, auto_drive) -> None:
        """AutoDrive instance — replays a recorded route for CAR RUN."""
        self._auto_drive = auto_drive
        if auto_drive is not None and hasattr(auto_drive, 'set_event_sink'):
            # every navigation step also goes into the scan log of the drive
            auto_drive.set_event_sink(self._on_nav_event)

    def _on_nav_event(self, text: str) -> None:
        """a line from the autonomous driving: into the scan log, and kept for the CAR RUN screen"""
        self._scan_rec.add_event(text)
        self._nav_text = text

    def _pil_to_display(self, img: Image.Image) -> None:
        img = img.convert('RGB')
        if img.size != (W, H):
            img = img.resize((W, H))
        try:
            # RGB888 -> RGB565 (high byte, low byte per pixel) done by Pillow itself: the same
            # bytes as the pixel-by-pixel loop below, but about 100 times faster — so drawing
            # the screen does not slow the car down while it is driving
            r, g, b = img.split()
            hi = ImageChops.add(r.point(self._LUT_R_HI), g.point(self._LUT_G_HI))
            lo = ImageChops.add(g.point(self._LUT_G_LO), b.point(self._LUT_B_LO))
            buf = Image.merge('LA', (hi, lo)).tobytes()
        except Exception:
            pixels = img.load()
            buf = bytearray(W * H * 2)
            idx = 0
            for y in range(H):
                for x in range(W):
                    r, g, b = pixels[x, y]
                    color = self._display.color565(r, g, b)
                    buf[idx]     = (color >> 8) & 0xFF
                    buf[idx + 1] = color & 0xFF
                    idx += 2
            buf = bytes(buf)
        self._display._set_window(0, 0, W - 1, H - 1)
        self._display._write_buf(buf)

    # lookup tables for the conversion above: colour = (r & 0xF8) << 8 | (g & 0xFC) << 3 | b >> 3
    _LUT_R_HI = [v & 0xF8 for v in range(256)]
    _LUT_G_HI = [v >> 5 for v in range(256)]
    _LUT_G_LO = [(v & 0x1C) << 3 for v in range(256)]
    _LUT_B_LO = [v >> 3 for v in range(256)]

    def _make_canvas(self, bg=None) -> tuple:
        img  = Image.new('RGB', (W, H), bg or C_BG)
        draw = ImageDraw.Draw(img)
        return img, draw

    def _font(self, size: int):
        try:
            return ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except Exception:
            return ImageFont.load_default()

    def _draw_grid(self, draw):
        for x in range(0, W, 32):
            draw.line([(x, 0), (x, H)], fill=(15, 25, 50), width=1)
        for y in range(0, H, 32):
            draw.line([(0, y), (W, y)], fill=(15, 25, 50), width=1)

    def _draw_rounded_rect(self, draw, x, y, w, h, r, fill, outline=None, ow=2):
        draw.rounded_rectangle([x, y, x+w, y+h],
                               radius=r, fill=fill, outline=outline, width=ow)

    def _draw_corner_marks(self, draw):
        L, T, c = 12, 2, C_CYAN_DIM
        draw.line([(2, 2), (2+L, 2)], fill=c, width=T)
        draw.line([(2, 2), (2, 2+L)], fill=c, width=T)
        draw.line([(W-2, 2), (W-2-L, 2)], fill=c, width=T)
        draw.line([(W-2, 2), (W-2, 2+L)], fill=c, width=T)
        draw.line([(2, H-2), (2+L, H-2)], fill=c, width=T)
        draw.line([(2, H-2), (2, H-2-L)], fill=c, width=T)
        draw.line([(W-2, H-2), (W-2-L, H-2)], fill=c, width=T)
        draw.line([(W-2, H-2), (W-2, H-2-L)], fill=c, width=T)

    def show_splash(self) -> None:
        print("[UI] Splash screen")
        f_big   = self._font(36)
        f_small = self._font(12)
        f_sub   = self._font(14)

        for pct in range(0, 101, 2):
            img, draw = self._make_canvas()
            self._draw_grid(draw)
            self._draw_corner_marks(draw)
            draw.line([(0, 0), (W, 0)], fill=C_CYAN, width=2)
            draw.line([(0, H-1), (W, H-1)], fill=C_CYAN, width=2)
            draw.text((W//2, 65),  "CAR", font=f_big,   fill=C_WHITE,   anchor="mm")
            draw.line([(W//2-45, 88), (W//2+45, 88)], fill=C_CYAN, width=2)
            draw.text((W//2, 115), "ML",  font=f_big,   fill=C_CYAN,    anchor="mm")
            draw.text((W//2, 145), "AUTONOMOUS VEHICLE", font=f_small,
                      fill=C_CYAN_DIM, anchor="mm")
            bx, by, bw, bh = 50, 175, 220, 14
            self._draw_rounded_rect(draw, bx, by, bw, bh, 6,
                                    fill=C_BG2, outline=C_CYAN_DIM)
            fill_w = int(bw * pct / 100)
            if fill_w > 8:
                self._draw_rounded_rect(draw, bx+2, by+2,
                                        fill_w-4, bh-4, 4, fill=C_CYAN)
            draw.text((W//2, 203), f"{pct}%", font=f_sub,
                      fill=C_CYAN, anchor="mm")
            self._pil_to_display(img)
        time.sleep(0.5)

    def show_home_screen(self) -> None:
        f_title = self._font(13)
        f_btn   = self._font(14)
        f_sub   = self._font(10)

        img, draw = self._make_canvas()
        self._draw_grid(draw)
        self._draw_corner_marks(draw)

        draw.rectangle([(0, 0), (W, 36)], fill=C_BG2)
        draw.line([(0, 36), (W, 36)], fill=C_CYAN, width=1)
        draw.text((W//2, 18), "CAR ML  //  CONTROL CENTER",
                  font=f_title, fill=C_CYAN, anchor="mm")

        # Bluetooth button
        self._draw_rounded_rect(draw, BTN_BT_X, BTN_BT_Y, BTN_BT_W, BTN_BT_H, 10,
                                fill=C_BLUE_DIM, outline=C_CYAN, ow=2)
        cx = BTN_BT_X + BTN_BT_W // 2
        cy = BTN_BT_Y + 22
        draw.line([(cx-6, cy+6), (cx+6, cy-6)], fill=C_CYAN, width=2)
        draw.line([(cx-6, cy-6), (cx+6, cy+6)], fill=C_CYAN, width=2)
        draw.line([(cx, cy-10), (cx, cy+10)],   fill=C_CYAN, width=2)
        draw.text((cx, BTN_BT_Y+46), "BLUETOOTH", font=f_btn, fill=C_WHITE, anchor="mm")
        draw.text((cx, BTN_BT_Y+60), "CONNECT", font=f_sub, fill=C_CYAN_DIM, anchor="mm")

        # ML / LIDAR button
        self._draw_rounded_rect(draw, BTN_ML_X, BTN_ML_Y, BTN_ML_W, BTN_ML_H, 10,
                                fill=C_RED_DIM, outline=C_RED, ow=2)
        mx = BTN_ML_X + BTN_ML_W // 2
        my = BTN_ML_Y + 22
        for dy in (-8, 0, 8):
            draw.ellipse([(mx-18+dy//2-3, my+dy-3), (mx-18+dy//2+3, my+dy+3)], fill=C_RED)
            draw.ellipse([(mx+12-3, my+dy-3), (mx+12+3, my+dy+3)],       fill=C_RED)
            draw.line([(mx-15+dy//2, my+dy), (mx+9, my+dy)], fill=C_RED_DIM, width=1)
        draw.text((mx, BTN_ML_Y+46), "AUTO DRIVE", font=f_btn, fill=C_WHITE, anchor="mm")
        draw.text((mx, BTN_ML_Y+60), "LIDAR MAP", font=f_sub, fill=(200, 80, 80), anchor="mm")

        draw.rectangle([(0, H-22), (W, H)], fill=C_BG2)
        draw.text((10, H-11), "SYS: ONLINE", font=f_sub, fill=C_CYAN_DIM, anchor="lm")
        self._pil_to_display(img)

    def show_auto_drive_menu(self) -> None:
        """AUTO DRIVE LIDAR MAP submenu — RECORDING ROUTE / CAR RUN."""
        f_title = self._font(13)
        f_btn   = self._font(13)
        f_sub   = self._font(10)

        img, draw = self._make_canvas()
        self._draw_grid(draw)
        self._draw_corner_marks(draw)

        draw.rectangle([(0, 0), (W, 36)], fill=C_BG2)
        draw.line([(0, 36), (W, 36)], fill=C_CYAN, width=1)
        draw.text((W//2, 18), "AUTO DRIVE  //  SELECT MODE",
                  font=f_title, fill=C_CYAN, anchor="mm")

        # RECORDING ROUTE button
        self._draw_rounded_rect(draw, BTN_RECORD_X, BTN_RECORD_Y,
                                BTN_RECORD_W, BTN_RECORD_H, 10,
                                fill=C_BLUE_DIM, outline=C_CYAN, ow=2)
        rx = BTN_RECORD_X + BTN_RECORD_W // 2
        draw.ellipse([(rx-7, BTN_RECORD_Y+14), (rx+7, BTN_RECORD_Y+28)],
                    outline=C_CYAN, width=2)
        draw.ellipse([(rx-3, BTN_RECORD_Y+18), (rx+3, BTN_RECORD_Y+24)], fill=C_RED)
        draw.text((rx, BTN_RECORD_Y+44), "RECORDING", font=f_btn, fill=C_WHITE, anchor="mm")
        draw.text((rx, BTN_RECORD_Y+58), "ROUTE", font=f_sub, fill=C_CYAN_DIM, anchor="mm")

        # CAR RUN button
        self._draw_rounded_rect(draw, BTN_RUNML_X, BTN_RUNML_Y,
                                BTN_RUNML_W, BTN_RUNML_H, 10,
                                fill=C_RED_DIM, outline=C_RED, ow=2)
        mx = BTN_RUNML_X + BTN_RUNML_W // 2
        my = BTN_RUNML_Y + 21
        pts = [(mx-9, my-10), (mx-9, my+10), (mx+11, my)]
        draw.polygon(pts, fill=C_RED)
        draw.text((mx, BTN_RUNML_Y+44), "CAR RUN", font=f_btn, fill=C_WHITE, anchor="mm")
        draw.text((mx, BTN_RUNML_Y+58), "NEURAL NETWORK", font=f_sub, fill=(200, 80, 80), anchor="mm")

        # what the car knows: recorded runs + the trained network
        draw.text((W//2, 58), self._model_status_text(), font=f_sub, fill=C_CYAN_DIM, anchor="mm")

        # TRAIN button (trains the neural network on the recorded runs)
        self._draw_rounded_rect(draw, BTN_TRAIN_X, BTN_TRAIN_Y, BTN_TRAIN_W, BTN_TRAIN_H, 6,
                                fill=C_BLUE_DIM, outline=C_CYAN)
        draw.text((BTN_TRAIN_X+BTN_TRAIN_W//2, BTN_TRAIN_Y+BTN_TRAIN_H//2),
                  "TRAIN NETWORK", font=self._font(11), fill=C_WHITE, anchor="mm")

        self._draw_rounded_rect(draw, BTN_BACK_X, BTN_BACK_Y, BTN_BACK_W, BTN_BACK_H, 6,
                                fill=C_GRAY, outline=C_CYAN_DIM)
        draw.text((BTN_BACK_X+BTN_BACK_W//2, BTN_BACK_Y+BTN_BACK_H//2), "◀  BACK",
                  font=f_btn, fill=C_WHITE, anchor="mm")

        self._pil_to_display(img)

    def show_run_ml_screen(self, status: str) -> None:
        """
        CAR RUN — status in {'driving', 'no_route', 'not_configured'}.
        'driving'  : replay is active, in progress right now.
        'no_route' : no recorded session was found in the database.
        'not_configured' : set_auto_drive() was never called from main.py.
        """
        f_title = self._font(13)
        f_msg   = self._font(11)
        f_btn   = self._font(13)

        img, draw = self._make_canvas()
        self._draw_grid(draw)
        self._draw_corner_marks(draw)

        draw.rectangle([(0, 0), (W, 36)], fill=C_BG2)
        draw.line([(0, 36), (W, 36)], fill=C_RED, width=1)
        draw.text((W//2, 18), "CAR RUN", font=f_title, fill=C_RED, anchor="mm")

        msg2 = ""
        if status == 'driving':
            msg, col = "●  DRIVING BY ITSELF", C_RED
        elif status == 'no_model':
            msg, col = "NO TRAINED NEURAL NETWORK YET", C_CYAN
            msg2 = "record runs, then press TRAIN NETWORK"
        elif status == 'no_route':
            msg, col = "NO RECORDED ROUTE FOUND IN DB", C_CYAN
        else:
            msg, col = "AUTO DRIVE NOT CONNECTED", C_CYAN
        draw.text((W//2, 100), msg, font=f_msg, fill=col, anchor="mm")
        if msg2:
            draw.text((W//2, 122), msg2, font=self._font(10), fill=C_GRAY, anchor="mm")

        btn_label = "■  STOP" if status == 'driving' else "◀  BACK"
        self._draw_rounded_rect(draw, BTN_BACK_X, BTN_BACK_Y, BTN_BACK_W, BTN_BACK_H, 6,
                                fill=C_GRAY, outline=C_CYAN_DIM)
        draw.text((BTN_BACK_X+BTN_BACK_W//2, BTN_BACK_Y+BTN_BACK_H//2), btn_label,
                  font=f_btn, fill=C_WHITE, anchor="mm")

        self._pil_to_display(img)

    def show_bluetooth_screen(self, status: str = "waiting") -> None:
        self._bt_connected = (status == "connected")

        f_title  = self._font(13)
        f_status = self._font(16)
        f_sub    = self._font(10)
        f_back   = self._font(11)

        img, draw = self._make_canvas()
        self._draw_grid(draw)
        self._draw_corner_marks(draw)

        draw.rectangle([(0, 0), (W, 36)], fill=C_BG2)
        draw.text((W//2, 18), "BLUETOOTH  //  CONNECTION", font=f_title, fill=C_CYAN, anchor="mm")

        if status == "connected":
            box_color, status_text = (0, 60, 30), "CONNECTED"
            status_col = (0, 255, 120)
            sub_text   = "Device linked — ready to drive"
        else:
            box_color, status_text = (5, 20, 50), "SEARCHING..."
            status_col = C_CYAN
            sub_text   = "Open app on phone and connect"

        self._draw_rounded_rect(draw, 30, 48, 260, 85, 10, fill=box_color, outline=C_CYAN)
        cx, cy = W//2, 78
        draw.ellipse([(cx-18, cy-18), (cx+18, cy+18)], outline=status_col, width=2)
        draw.text((W//2, 108), status_text, font=f_status, fill=status_col, anchor="mm")
        draw.text((W//2, 126), sub_text, font=f_sub, fill=C_GRAY, anchor="mm")

        self._draw_rounded_rect(draw, BTN_BACK_X, BTN_BACK_Y, BTN_BACK_W, BTN_BACK_H, 6, fill=C_GRAY, outline=C_CYAN_DIM)
        draw.text((BTN_BACK_X+BTN_BACK_W//2, BTN_BACK_Y+BTN_BACK_H//2), "◀  BACK", font=f_back, fill=C_WHITE, anchor="mm")
        self._pil_to_display(img)

    # ── Screen 4 — LIDAR Point Cloud Map ─────────────────────

    def _run_status(self) -> tuple:
        """(text, colour) for the strip on the CAR RUN screen, from the latest navigation line"""
        text = self._nav_text or ""
        if text.startswith("ML FINISHED"):
            return text[3:60], (0, 255, 150)
        if text.startswith("ML #"):
            # "ML #12 joy x+0.31 y+0.69 -> net F | ... | do F 0.12s | ahead 420mm"
            do = re.search(r"do (\w)", text)
            steer = re.search(r"x([+-]\d\.\d\d)", text)
            act = {'F': 'FORWARD', 'L': 'LEFT', 'R': 'RIGHT', 'B': 'BACK', 'S': 'STOP'}.get(
                do.group(1) if do else '', '')
            return (f"●  NN 100%  {act}" + (f"   steer {steer.group(1)}" if steer else ""), C_RED)
        if text.startswith("ML start"):
            return "●  NEURAL NETWORK - STARTING", C_RED
        if "FINISHED" in text:
            return "FINISHED ✓", (0, 255, 150)
        lap = re.search(r"lap (\d+)/(\d+)", text)
        if lap:
            return f"●  DRIVING   LAP {lap.group(1)} / {lap.group(2)}", C_RED
        lap = re.search(r"lap (\d+)\.\d+/(\d+)", text)              # the ring track counts 0.00 .. N
        if lap:
            return f"●  DRIVING   LAP {min(int(lap.group(1)) + 1, int(lap.group(2)))} / {lap.group(2)}", C_RED
        return "●  DRIVING", C_RED

    def _draw_lidar_map(self, points: list, recording: bool = False, running: bool = False) -> None:
        """ציור מפת נקודות ליידאר חיה וגולמית ללא סינון שמעלים מידע"""
        f_title = self._font(11)
        f_back  = self._font(11)
        f_ring  = self._font(7)
        f_msg   = self._font(10)

        img, draw = self._make_canvas(bg=(5, 8, 20))
        cx, cy, R = MAP_CX, MAP_CY, MAP_R
        DISPLAY_MAX_MM = 500 

        # טבעות טווח במפה
        ring_color = (25, 40, 70)
        for ring_mm in [100, 300, 500]:
            ring_r = int(R * ring_mm / DISPLAY_MAX_MM)
            draw.ellipse([cx-ring_r, cy-ring_r, cx+ring_r, cy+ring_r], outline=ring_color, width=1)

        # צירי הצלבה[cite: 5]
        axis_color = (20, 35, 65)
        draw.line([(cx, cy-R), (cx, cy+R)], fill=axis_color, width=1)
        draw.line([(cx-R, cy), (cx+R, cy)], fill=axis_color, width=1)

        # ── ציור נקודות הליידאר (הצגה ישירה של כל מה שנקלט) ──
        for p in points:
            if p.distance < 50 or p.distance > DISPLAY_MAX_MM:
                continue

            # המרה לקואורדינטות מסך[cite: 5]
            scale = p.distance / DISPLAY_MAX_MM
            r_px  = int(scale * R)
            rad   = math.radians(p.angle)
            px    = cx + int(r_px * math.sin(rad))
            py    = cy - int(r_px * math.cos(rad))

            # קביעת אזור לפי זווית
            a = p.angle % 360
            if a >= 337.5 or a < 22.5:   zg = 'front'
            elif a < 67.5:               zg = 'front-right'
            elif a < 112.5:              zg = 'right'
            elif a < 157.5:              zg = 'rear-right'
            elif a < 202.5:              zg = 'rear'
            elif a < 247.5:              zg = 'rear-left'
            elif a < 292.5:              zg = 'left'
            else:                        zg = 'front-left'

            # קביעת צבע לפי אזור ומרחק
            if zg in ('front', 'front-right', 'front-left', 'left'):
                # אזורים 1,2,7,8 — רגישות גבוהה
                if p.distance < 150:
                    dot_color = (255, 50,  50)   # אדום
                elif p.distance < 250:
                    dot_color = (255, 200, 0)    # צהוב
                else:
                    dot_color = (0,   255, 180)  # ירוק-ציאן
            elif zg in ('right', 'rear-left'):
                # אזורים 3,6 — רגישות בינונית
                if p.distance < 70:
                    dot_color = (255, 50,  50)   # אדום
                elif p.distance < 250:
                    dot_color = (255, 200, 0)    # צהוב
                else:
                    dot_color = (0,   255, 180)  # ירוק-ציאן
            else:
                # אזורים 4,5 — אחורה, רגישות נמוכה
                if p.distance < 300:
                    dot_color = (255, 50,  50)   # אדום
                elif p.distance < 350:
                    dot_color = (255, 200, 0)    # צהוב
                else:
                    dot_color = (0,   255, 180)  # ירוק-ציאן

            draw.ellipse([px-1, py-1, px+2, py+2], fill=dot_color)

        # אייקון המכונית במרכז[cite: 5]
        cw, ch = CAR_W, CAR_H
        draw.rounded_rectangle([cx-cw//2, cy-ch//2, cx+cw//2, cy+ch//2], radius=3, fill=(40, 80, 220), outline=(100, 160, 255), width=1)
        draw.line([(cx-cw//2+2, cy-ch//2+3), (cx+cw//2-2, cy-ch//2+3)], fill=C_CYAN, width=2)

        # סרגל עליון[cite: 5]
        draw.rectangle([(0, 0), (W, 26)], fill=(5, 10, 30))
        draw.line([(0, 26), (W, 26)], fill=C_CYAN_DIM, width=1)
        if recording:
            draw.text((W//2, 13), f"RUNS RECORDED: {self._recordings_total}   (this time: "
                      f"{self._recording_count})", font=f_title, fill=C_CYAN, anchor="mm")
        elif running:
            draw.text((W//2, 13), "CAR RUN  //  LIVE LIDAR", font=f_title, fill=C_RED, anchor="mm")
        else:
            draw.text((W//2, 13), "LIVE LIDAR MAP (ALL POINTS)", font=f_title, fill=C_CYAN, anchor="mm")

        # הודעת הקלטה (HORN) — עכשיו למעלה, כרצועה על גבי מפת הליידאר[cite: 5]
        if recording:
            if self._recording_active:
                msg, msg_col = "● RECORDING - 'HORN' TO STOP & SAVE", C_RED
            elif self._recordings_total >= RECORDING_TARGET:
                msg, msg_col = f"{self._recordings_total} RUNS ✓ READY TO TRAIN ('HORN' = 1 MORE)", (0, 255, 150)
            else:
                msg, msg_col = f"PRESS 'HORN' TO RECORD RUN {self._recordings_total + 1}", C_CYAN
            draw.rectangle([(0, 26), (W, 44)], fill=(5, 10, 30))
            draw.line([(0, 44), (W, 44)], fill=C_CYAN_DIM, width=1)
            draw.text((W//2, 35), msg, font=f_msg, fill=msg_col, anchor="mm")

        # CAR RUN: what the car is doing, as a strip over the map
        if running:
            msg, msg_col = self._run_status()
            draw.rectangle([(0, 26), (W, 44)], fill=(5, 10, 30))
            draw.line([(0, 44), (W, 44)], fill=C_CYAN_DIM, width=1)
            draw.text((W//2, 35), msg, font=f_msg, fill=msg_col, anchor="mm")

        # כפתור חזרה[cite: 5]
        draw.rounded_rectangle([BTN_BACK_X, BTN_BACK_Y, BTN_BACK_X+BTN_BACK_W, BTN_BACK_Y+BTN_BACK_H], radius=6, fill=(30, 40, 70), outline=C_RED if running else C_CYAN_DIM, width=1)
        draw.text((BTN_BACK_X+BTN_BACK_W//2, BTN_BACK_Y+BTN_BACK_H//2), "■  STOP" if running else "◀  BACK", font=f_back, fill=C_WHITE, anchor="mm")

        if self._lidar_running:          # the screen was left meanwhile: do not draw over the next one
            self._pil_to_display(img)

    def show_lidar_screen(self, recording: bool = False, running: bool = False) -> None:
        print("[UI] LIDAR map screen" + (" (recording)" if recording else " (car run)" if running else ""))
        self._lidar_running = True

        def _loop():
            while self._lidar_running:
                # שינוי קריטי: מושכים את כל הנקודות (get_points) במקום get_filtered_points[cite: 5, 6]
                points = self._lidar.get_points() if self._lidar else []
                self._draw_lidar_map(points, recording=recording, running=running)
                time.sleep(RUN_VIEW_PERIOD if running else 0.15)

        self._lidar_thread = threading.Thread(target=_loop, daemon=True)
        self._lidar_thread.start()

    def hide_lidar_screen(self) -> None:
        self._lidar_running = False
        if self._lidar_thread:
            self._lidar_thread.join(timeout=0.5)

    def _is_pressed(self, tx, ty, bx, by, bw, bh) -> bool:
        return bx <= tx <= bx+bw and by <= ty <= by+bh

    def _start_bluetooth(self) -> None:
        subprocess.run(['bluetoothctl', 'discoverable', 'on'], capture_output=True)
        subprocess.run(['bluetoothctl', 'pairable', 'on'], capture_output=True)
        # If we already connected earlier and just left the connection running
        # in the background, reflect that instead of showing "waiting" again.
        self.show_bluetooth_screen(status='connected' if self._bt_connected else 'waiting')
        if self._bt_driver:
            self._bt_driver.set_on_connect(lambda: self.show_bluetooth_screen(status='connected'))
            self._bt_driver.set_on_disconnect(lambda: self.show_bluetooth_screen(status='waiting'))
            if not self._bt_driver.is_running:
                self._bt_driver.start_async()

    def _stop_bluetooth(self) -> None:
        """Fully stop the Bluetooth server (not used from BACK anymore —
        kept for a real 'disconnect' action if ever needed)."""
        if self._bt_driver:
            self._bt_driver.stop()
        subprocess.run(['bluetoothctl', 'discoverable', 'off'], capture_output=True)

    def _leave_bluetooth_screen(self) -> None:
        """
        BACK from the Bluetooth screen — just stop advertising as discoverable.
        The RFCOMM server/connection itself is left running in the background,
        so the phone stays connected and HORN keeps working later on the
        RECORDING ROUTE screen.
        """
        subprocess.run(['bluetoothctl', 'discoverable', 'off'], capture_output=True)

    def _on_horn_pressed(self) -> None:
        """
        App's HORN button toggles recording — odd presses start, even presses
        stop/save & bump the counter (LIGHTS is not used: it sticks on-press
        on the app, so it's unreliable for this). Once RECORDING_TARGET
        routes are done, further HORN presses are ignored.

        Recording itself is event-driven (see _on_drive_action_changed /
        _close_recording_segment / _save_recording below): instead of
        sampling on a timer, each driving command is timed for exactly how
        long it was held, once the NEXT command (or HORN-stop) ends that
        segment; the whole route is written to the DB when recording stops.
        """
        # V5: during CAR RUN the HORN is an emergency stop — the car stops driving at once
        # (the screen then goes back to the menu by itself, exactly as after STOP)
        if self._auto_drive is not None and self._auto_drive.is_running:
            print("[UI] HORN pressed during CAR RUN -> stopping the car")
            self._auto_drive.stop()
            return
        if not self._recording_screen_active:
            return
        if not self._recording_active:
            self._recording_active = True
            self._recording_session_id = (
                self._route_db.start_session() if self._route_db else None
            )
            if self._bt_driver:
                action, _ = self._bt_driver.get_action_state()
            else:
                action = 'S'
            now = time.time()
            self._scan_rec.start('REC')
            self._scan_rec.add_event(f"CMD {action}", now)
            with self._record_lock:
                self._record_buffer     = []
                self._record_started_at = now
                self._open_recording_segment(action, now)
            if self._bt_driver:
                self._bt_driver.set_on_action_change(self._on_drive_action_changed)
            print("[UI] Recording STARTED (HORN)")
        else:
            if self._bt_driver:
                self._bt_driver.set_on_action_change(None)
            self._recording_active = False
            self._save_recording()
            self._recording_count += 1
            self._recordings_total += 1
            print(f"[UI] Recording STOPPED & SAVED (HORN) — run {self._recordings_total} "
                  f"({self._recording_count} this time)")

    def _on_drive_action_changed(self, new_action: str, now: float = None) -> None:
        """
        Called by BluetoothDriver the instant a new driving command arrives
        while a RECORDING ROUTE session is active. `now` is the moment the
        command was received: that ONE instant both ends the previous action
        and starts the new one, so no time is lost between two events.
        """
        if not self._recording_active:
            return
        if now is None:
            now = time.time()
        self._scan_rec.add_event(f"CMD {new_action}", now)
        with self._record_lock:
            self._close_recording_segment(now)
            self._open_recording_segment(new_action, now)

    def _open_recording_segment(self, action: str, now: float) -> None:
        """Start timing `action` from `now` (caller holds _record_lock)."""
        self._record_segment_action = action
        self._record_segment_start  = now
        self._record_segment_zones  = self._lidar.get_ml_vector() if self._lidar else [None] * 8

    def _close_recording_segment(self, now: float) -> None:
        """
        End the segment being timed at `now` and add it to the in-memory
        recording (caller holds _record_lock). Nothing is written to disk
        here. The printed event number is the row's position in the saved
        session, so the terminal output and the DB rows match one to one.
        """
        action = self._record_segment_action
        start  = self._record_segment_start
        self._record_segment_action = None
        self._record_segment_start  = None
        if action is None or start is None:
            return
        duration = max(0.0, now - start)
        buf = self._record_buffer

        if buf and buf[-1]['action'] == action:
            # Same action as the previous row (whatever came in between was
            # too short to count) — it is one continuous hold, keep one row.
            buf[-1]['duration'] += duration
            print(f"[UI] event #{len(buf)} extended: action='{action}' "
                  f"duration={buf[-1]['duration']:.3f}s")
        elif duration < RECORD_MIN_SEGMENT:
            # Press + release processed back-to-back (they arrived in the same
            # Bluetooth packet) — the motors never really did this, so it gets
            # no row of its own; its time stays with the previous row so the
            # recording's total length is still exact.
            if buf:
                buf[-1]['duration'] += duration
            print(f"[UI] skipped: action='{action}' duration={duration:.3f}s "
                  f"(shorter than {RECORD_MIN_SEGMENT}s)")
        else:
            buf.append({'action': action, 'start': start, 'duration': duration,
                        'zones': self._record_segment_zones})
            print(f"[UI] event #{len(buf)}: action='{action}' duration={duration:.3f}s")

    def _save_recording(self) -> None:
        """
        Close the segment in progress and write the whole recording to the DB
        in one transaction. Called when recording stops (HORN, or BACK while
        recording). Prints what was saved so the output can be checked
        against the DB file.
        """
        now = time.time()
        sid = self._recording_session_id
        # V4: once the LIDAR log is written, its pictures go into car_routes.db (ml_samples),
        # next to this drive's button presses (action_events) — the network's training data
        self._scan_rec.stop_and_save(f"recording session_id={sid}",
                                     on_saved=(lambda path: self._store_in_db(path, sid))
                                     if self._route_db and sid is not None else None)
        with self._record_lock:
            self._close_recording_segment(now)
            events = self._record_buffer
            started_at = self._record_started_at
            self._record_buffer     = []
            self._record_started_at = None
        if not events:
            return
        total = sum(ev['duration'] for ev in events)
        if not self._route_db or self._recording_session_id is None:
            print(f"[UI] No route database — {len(events)} events NOT saved")
            return
        saved = self._route_db.log_action_events(self._recording_session_id, events)
        print(f"[UI] Saved {saved} events to DB (session #{self._recording_session_id}) — "
              f"sum of durations {total:.3f}s, recording length {now - started_at:.3f}s")

    # ── Neural network: status, TRAIN screen ──────────────────

    @staticmethod
    def _here(*parts) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)

    def _count_recordings(self) -> int:
        """recorded drives in the database car_routes.db (the training data of the network)"""
        if self._route_db is not None and hasattr(self._route_db, 'ml_sessions'):
            try:
                return len(self._route_db.ml_sessions())
            except Exception:
                pass
        return len(glob.glob(self._here("scans", "scan_REC_*.csv.gz")))

    def _store_in_db(self, path, session_id) -> None:
        """(background thread) a new recording's LIDAR pictures -> car_routes.db"""
        try:
            import ml_dataset
            ml_dataset.store_recording(self._route_db, path, session_id, quiet=False)
        except Exception as e:
            print(f"[UI] Could not store the drive in the database: {e!r}")

    def _model_status_text(self) -> str:
        runs = self._count_recordings()
        path = self._here("ml_model.npz")
        if not os.path.isfile(path):
            return f"DB RUNS: {runs}   |   NETWORK: NOT TRAINED YET"
        try:
            import json
            import numpy as np
            with np.load(path) as z:
                meta = json.loads(str(z['info'])).get('meta', {})
            return (f"DB RUNS: {runs}   |   NETWORK: {meta.get('runs', '?')} runs, "
                    f"test {meta.get('test_accuracy', 0) * 100:.0f}%")
        except Exception:
            return f"DB RUNS: {runs}   |   NETWORK: trained"

    def show_training_screen(self) -> None:
        f_title = self._font(13)
        f_line  = self._font(9)
        f_big   = self._font(12)
        img, draw = self._make_canvas()
        self._draw_grid(draw)
        self._draw_corner_marks(draw)
        draw.rectangle([(0, 0), (W, 36)], fill=C_BG2)
        draw.line([(0, 36), (W, 36)], fill=C_CYAN, width=1)
        draw.text((W//2, 18), "TRAINING THE NEURAL NETWORK", font=f_title, fill=C_CYAN, anchor="mm")
        lines = [l for l in self._train_lines if l.strip()][-9:]
        for i, line in enumerate(lines):
            draw.text((10, 46 + 14 * i), line.strip()[:54], font=f_line, fill=C_WHITE)
        if self._training:
            draw.text((W//2, 182), "PLEASE WAIT (about 5-10 minutes) ...", font=f_big, fill=(255, 200, 0), anchor="mm")
        elif self._train_result:
            ok, text = self._train_result
            draw.text((W//2, 182), text[:46], font=f_big, fill=(0, 255, 150) if ok else (255, 120, 120),
                      anchor="mm")
            self._draw_rounded_rect(draw, BTN_BACK_X, BTN_BACK_Y, BTN_BACK_W, BTN_BACK_H, 6,
                                    fill=C_GRAY, outline=C_CYAN_DIM)
            draw.text((BTN_BACK_X+BTN_BACK_W//2, BTN_BACK_Y+BTN_BACK_H//2), "◀  BACK",
                      font=f_title, fill=C_WHITE, anchor="mm")
        self._pil_to_display(img)

    def _start_training(self) -> None:
        """TRAIN NETWORK: build the dataset from scans/ and train (ml_train.py) in the background."""
        if self._training:
            return
        self._training     = True
        self._train_result = None
        self._train_lines  = ["reading the training data from car_routes.db ..."]
        self.show_training_screen()
        last_draw = [0.0]

        skip = ("Layer", "Input", "Dense", "Total parameters", "  saved", "  report", "  decision time",
                "    ", "  confusion", "             ", "  training took", "  keeping", "  no improvement")

        def on_log(text):
            if not text.strip() or text.startswith(skip):
                return                                   # the full details are in ml_data/train_report.txt
            if "epoch" in text and "/" in text and not text.strip().startswith("epoch 1/"):
                # one line per epoch is too much for the screen: keep only the latest one
                if self._train_lines and "epoch" in self._train_lines[-1]:
                    self._train_lines[-1] = text
                else:
                    self._train_lines.append(text)
            else:
                self._train_lines.append(text)
            if time.time() - last_draw[0] > 1.0:
                last_draw[0] = time.time()
                self.show_training_screen()

        def work():
            try:
                import ml_train
                code = ml_train.main(["--fast"], on_log=on_log)
                result = [l for l in self._train_lines if l.startswith("RESULT:")]
                if code == 0:
                    acc = re.search(r"test accuracy (\d+)%", result[-1]) if result else None
                    self._train_result = (True, f"DONE ✓  test accuracy {acc.group(1)}%" if acc
                                          else "DONE ✓  network saved")
                else:
                    self._train_result = (False, "NOT TRAINED - record more runs")
            except Exception as e:
                self._train_lines.append(f"ERROR: {e!r}")
                self._train_result = (False, "TRAINING FAILED (see the lines above)")
            finally:
                self._training = False
                self.show_training_screen()

        threading.Thread(target=work, daemon=True).start()

    def run(self) -> None:
        self.show_splash()
        self.show_home_screen()
        current = 'home'

        while True:
            pos = self._touch.wait_for_touch(timeout=0.1)
            if not pos:
                # CAR RUN finished on its own (ran out of recorded data) —
                # leave the "DRIVING..." screen without needing a touch.
                if (current == 'run_ml' and self._run_ml_started
                        and self._auto_drive and not self._auto_drive.is_running):
                    self._run_ml_started = False
                    self.hide_lidar_screen()
                    self._scan_rec.stop_and_save("replay finished")
                    if self._lidar:
                        self._lidar.stop_motor()
                    current = 'auto_drive_menu'
                    self.show_auto_drive_menu()
                continue
            x, y = pos

            if current == 'home':
                if self._is_pressed(x, y, BTN_BT_X, BTN_BT_Y, BTN_BT_W, BTN_BT_H):
                    current = 'bluetooth'
                    self._start_bluetooth()
                elif self._is_pressed(x, y, BTN_ML_X, BTN_ML_Y, BTN_ML_W, BTN_ML_H):
                    current = 'auto_drive_menu'
                    self.show_auto_drive_menu()

            elif current == 'training':
                # BACK works once training has finished (it takes about a minute)
                if not self._training and self._is_pressed(
                        x, y, BTN_BACK_X, BTN_BACK_Y, BTN_BACK_W, BTN_BACK_H + 50):
                    current = 'auto_drive_menu'
                    self.show_auto_drive_menu()

            elif current == 'bluetooth':
                if self._is_pressed(x, y, BTN_BACK_X, BTN_BACK_Y, BTN_BACK_W, BTN_BACK_H + 50):
                    self._leave_bluetooth_screen()
                    current = 'home'
                    self.show_home_screen()
                elif self._bt_connected and self._is_pressed(
                        x, y, BTN_STATUS_X, BTN_STATUS_Y, BTN_STATUS_W, BTN_STATUS_H):
                    # Bluetooth stays connected/running in the background —
                    # just switch the screen to the LIDAR view.
                    current = 'lidar_bt'
                    if self._lidar:
                        self._lidar.start_motor()
                    self.show_lidar_screen()

            elif current == 'auto_drive_menu':
                if self._is_pressed(x, y, BTN_RECORD_X, BTN_RECORD_Y, BTN_RECORD_W, BTN_RECORD_H):
                    current = 'lidar'
                    self._recordings_total        = self._count_recordings()
                    self._recording_active        = False
                    self._recording_screen_active = True
                    if self._bt_driver:
                        self._bt_driver.set_on_horn(self._on_horn_pressed)
                    if self._lidar:
                        self._lidar.start_motor()
                    self.show_lidar_screen(recording=True)
                elif self._is_pressed(x, y, BTN_RUNML_X, BTN_RUNML_Y, BTN_RUNML_W, BTN_RUNML_H):
                    current = 'run_ml'
                    if not self._auto_drive:
                        self._run_ml_started = False
                        self.show_run_ml_screen('not_configured')
                    else:
                        # LIDAR motor must be spinning for the replay's live
                        # LIDAR correction to have real data to compare against.
                        if self._lidar:
                            self._lidar.start_motor()
                        self._nav_text = ""
                        self._run_ml_started = self._auto_drive.start()
                        if self._run_ml_started:
                            self._scan_rec.start('RUN')   # raw LIDAR log of the drive
                        if self._run_ml_started and RUN_SHOW_LIDAR and self._lidar:
                            self.show_lidar_screen(running=True)     # live: what the LIDAR sees
                        else:
                            self.show_run_ml_screen('driving' if self._run_ml_started else
                                                    (getattr(self._auto_drive, 'last_problem', '')
                                                     or 'no_route'))
                        if not self._run_ml_started and self._lidar:
                            self._lidar.stop_motor()
                elif self._is_pressed(x, y, BTN_TRAIN_X, BTN_TRAIN_Y, BTN_TRAIN_W, BTN_TRAIN_H):
                    current = 'training'
                    self._start_training()
                elif self._is_pressed(x, y, BTN_BACK_X, BTN_BACK_Y, BTN_BACK_W, BTN_BACK_H + 50):
                    current = 'home'
                    self.show_home_screen()

            elif current == 'run_ml':
                if self._is_pressed(x, y, BTN_BACK_X, BTN_BACK_Y, BTN_BACK_W, BTN_BACK_H + 50):
                    if self._auto_drive and self._auto_drive.is_running:
                        self._auto_drive.stop()   # pressed STOP mid-drive
                    self.hide_lidar_screen()
                    self._scan_rec.stop_and_save("replay stopped with BACK")
                    if self._lidar:
                        self._lidar.stop_motor()
                    self._run_ml_started = False
                    current = 'auto_drive_menu'
                    self.show_auto_drive_menu()

            elif current == 'lidar':
                if self._is_pressed(x, y, BTN_BACK_X, BTN_BACK_Y, BTN_BACK_W, BTN_BACK_H + 50):
                    self.hide_lidar_screen()
                    was_recording = self._recording_active
                    if self._bt_driver:
                        self._bt_driver.set_on_action_change(None)
                    self._recording_screen_active = False
                    self._recording_active        = False
                    if was_recording:
                        self._save_recording()  # BACK mid-recording — keep what was driven so far
                        self._recording_count += 1
                    if self._lidar:
                        self._lidar.stop_motor()
                    current = 'auto_drive_menu'
                    self.show_auto_drive_menu()

            elif current == 'lidar_bt':
                if self._is_pressed(x, y, BTN_BACK_X, BTN_BACK_Y, BTN_BACK_W, BTN_BACK_H + 50):
                    self.hide_lidar_screen()
                    if self._lidar:
                        self._lidar.stop_motor()
                    # Came from the Bluetooth screen — go back there, not home,
                    # since the car may still be driving via the phone app.
                    current = 'bluetooth'
                    self.show_bluetooth_screen(
                        status='connected' if self._bt_connected else 'waiting')

    def self_test(self) -> None:
        self.run()
