"""
lidar_driver.py
===============
Hardware driver for the D200 (LD14P) LiDAR sensor over UART.
Based on LD14P Development Manual V0.2 + verified against:
  - github.com/kaiaai/rpi5_ldrobot_ld14p
  - github.com/Poly-Mentor/D200-LiDAR-python

Packet format (47 bytes):
  [0]     = 0x54  start
  [1]     = 0x2C  verlen (12 points)
  [2-3]   = speed (deg/sec, little-endian)
  [4-5]   = start_angle (0.01 deg, little-endian)
  [6-41]  = 12 × (dist_lo, dist_hi, intensity)
  [42-43] = end_angle (0.01 deg, little-endian)
  [44-45] = timestamp (ms)
  [46]    = CRC8

Wiring (BCM):
  LIDAR TX → GPIO15 (UART0 RX, pin 10)
  LIDAR RX → GPIO14 (UART0 TX, pin 8)   — needed for start_motor()/stop_motor()
  VCC      → 5V
  GND      → GND
"""

import struct
import serial
import threading
import time
from collections import namedtuple, deque

LidarPoint = namedtuple('LidarPoint', ['angle', 'distance', 'intensity'])

# ── get_profile(): the distance profile used by RUN CAR ML's finish check ──
PROFILE_BINS         = 120           # 120 slices x 3 degrees around the car
PROFILE_BIN_DEG      = 360.0 / PROFILE_BINS
PROFILE_MIN_VALID_MM = 80            # closer than this is the car itself / an invalid reading
PROFILE_KEEP_SECONDS = 1.0           # how much point history is kept in memory
DEFAULT_REV_TIME     = 1.0 / 6.0     # LD14P spins at ~6 revolutions per second

# ── CRC8 table (poly 0x4D, from LD14P datasheet) ─────────────
CRC_TABLE = bytes([
    0x00,0x4d,0x9a,0xd7,0x79,0x34,0xe3,0xae,0xf2,0xbf,0x68,0x25,0x8b,0xc6,0x11,0x5c,
    0xa9,0xe4,0x33,0x7e,0xd0,0x9d,0x4a,0x07,0x5b,0x16,0xc1,0x8c,0x22,0x6f,0xb8,0xf5,
    0x1f,0x52,0x85,0xc8,0x66,0x2b,0xfc,0xb1,0xed,0xa0,0x77,0x3a,0x94,0xd9,0x0e,0x43,
    0xb6,0xfb,0x2c,0x61,0xcf,0x82,0x55,0x18,0x44,0x09,0xde,0x93,0x3d,0x70,0xa7,0xea,
    0x3e,0x73,0xa4,0xe9,0x47,0x0a,0xdd,0x90,0xcc,0x81,0x56,0x1b,0xb5,0xf8,0x2f,0x62,
    0x97,0xda,0x0d,0x40,0xee,0xa3,0x74,0x39,0x65,0x28,0xff,0xb2,0x1c,0x51,0x86,0xcb,
    0x21,0x6c,0xbb,0xf6,0x58,0x15,0xc2,0x8f,0xd3,0x9e,0x49,0x04,0xaa,0xe7,0x30,0x7d,
    0x88,0xc5,0x12,0x5f,0xf1,0xbc,0x6b,0x26,0x7a,0x37,0xe0,0xad,0x03,0x4e,0x99,0xd4,
    0x7c,0x31,0xe6,0xab,0x05,0x48,0x9f,0xd2,0x8e,0xc3,0x14,0x59,0xf7,0xba,0x6d,0x20,
    0xd5,0x98,0x4f,0x02,0xac,0xe1,0x36,0x7b,0x27,0x6a,0xbd,0xf0,0x5e,0x13,0xc4,0x89,
    0x63,0x2e,0xf9,0xb4,0x1a,0x57,0x80,0xcd,0x91,0xdc,0x0b,0x46,0xe8,0xa5,0x72,0x3f,
    0xca,0x87,0x50,0x1d,0xb3,0xfe,0x29,0x64,0x38,0x75,0xa2,0xef,0x41,0x0c,0xdb,0x96,
    0x42,0x0f,0xd8,0x95,0x3b,0x76,0xa1,0xec,0xb0,0xfd,0x2a,0x67,0xc9,0x84,0x53,0x1e,
    0xeb,0xa6,0x71,0x3c,0x92,0xdf,0x08,0x45,0x19,0x54,0x83,0xce,0x60,0x2d,0xfa,0xb7,
    0x5d,0x10,0xc7,0x8a,0x24,0x69,0xbe,0xf3,0xaf,0xe2,0x35,0x78,0xd6,0x9b,0x4c,0x01,
    0xf4,0xb9,0x6e,0x23,0x8d,0xc0,0x17,0x5a,0x06,0x4b,0x9c,0xd1,0x7f,0x32,0xe5,0xa8
])

PACKET_HEADER    = 0x54
PACKET_VER_LEN   = 0x2C
POINTS_PER_PACK  = 12
PACKET_SIZE      = 47

# ── Motor control commands (require LIDAR RX wired to Pi TX — GPIO14) ──
# Reference: LD14P Development Manual §2.2.2/2.2.3, verified against our
# own CRC8 table (0x54 A0 04 00 00 00 00 -> 0x5E, 0x54 A1 04 00 00 00 00 -> 0x4A)
CMD_MODE_START = 0xA0
CMD_MODE_STOP  = 0xA1


def _build_command(mode: int) -> bytes:
    """Build a 8-byte LD14P command packet: 54 <mode> 04 00 00 00 00 <crc>."""
    payload = bytes([PACKET_HEADER, mode, 0x04, 0x00, 0x00, 0x00, 0x00])
    return payload + bytes([_crc8(payload)])


def _crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = CRC_TABLE[crc ^ b]
    return crc


def _parse_packet(pkt: bytes, angle_offset: float = 0.0):
    """
    Parse a 47-byte LD14P packet.
    Returns list of LidarPoint or [] on error.
    Verified against kaiaai/rpi5_ldrobot_ld14p implementation.
    """
    if len(pkt) != PACKET_SIZE:
        return []
    if pkt[0] != PACKET_HEADER or pkt[1] != PACKET_VER_LEN:
        return []
    if _crc8(pkt[:-1]) != pkt[-1]:
        return []

    # ── Unpack using struct (little-endian, verified offsets) ─
    speed,   = struct.unpack_from('<H', pkt, 2)   # deg/sec
    start_a, = struct.unpack_from('<H', pkt, 4)   # 0.01 deg
    end_a,   = struct.unpack_from('<H', pkt, 42)  # 0.01 deg  ← FIX: was 43
    ts,      = struct.unpack_from('<H', pkt, 44)  # ms

    start_deg = start_a / 100.0
    end_deg   = end_a   / 100.0

    # ── Handle wrap-around ────────────────────────────────────
    span = end_a - start_a
    if span < 0:
        span += 36000
    step = span / (POINTS_PER_PACK - 1)

    points = []
    for i in range(POINTS_PER_PACK):
        dist_mm, intensity = struct.unpack_from('<HB', pkt, 6 + i * 3)
        if dist_mm == 0:
            continue
        angle = ((start_a + i * step) % 36000) / 100.0
        angle = (angle + angle_offset) % 360.0
        points.append(LidarPoint(angle=angle,
                                 distance=dist_mm,
                                 intensity=intensity))
    return points


class LidarDriver:
    """
    Driver for D200 LiDAR (LD14P) over UART.

    Usage:
        lidar = LidarDriver(angle_offset=0.0)
        lidar.start()
        points = lidar.get_points()   # list of LidarPoint
        lidar.cleanup()
    """

    # 8 zones (0°=front, clockwise)
    ALERT_DIST = {
        'front[1]':       300,
        'front-right[2]': 300,
        'right[3]':       300,
        'rear-right[4]':  400,   # אזור 4 — אחורה-ימין
        'rear[5]':        400,   # אזור 5 — אחורה
        'rear-left[6]':   300,
        'left[7]':        300,
        'front-left[8]':  300,
    }

    def __init__(self,
                 port: str = '/dev/serial0',
                 baudrate: int = 230400,
                 angle_offset: float = 0.0):
        self.port         = port
        self.baudrate     = baudrate
        self.angle_offset = angle_offset

        self._ser              = None
        self._lock             = threading.Lock()
        self._thread           = None
        self._running          = False
        self._current_points   = []
        self._on_scan_callback = None

        # Recent packets as (arrival time, [LidarPoint, ...]) — get_profile()
        # builds its answer from the last revolution's worth of these.
        self._recent      = deque()
        self._recent_lock = threading.Lock()
        self._rev_time    = DEFAULT_REV_TIME   # updated from the packets' speed field
        self._on_packet   = None               # see set_on_packet()
        self._lidar_clock = 0.0                # time of the newest packet (Pi time, LIDAR-clock precision)
        self._last_ts     = None

        print(f"[LidarDriver] Initialised — offset={self.angle_offset}°")

    # ── Callbacks ─────────────────────────────────────────────

    def set_on_packet(self, callback) -> None:
        """
        callback(now, speed_deg_s, lidar_ts_ms, points) for EVERY packet as it
        arrives (about 330 per second) — used by scan_recorder.py to save the
        raw LIDAR data of a drive. It runs on the read thread, so it must be
        quick. Pass None to switch it off.
        """
        self._on_packet = callback

    def set_on_scan(self, callback) -> None:
        """Called with list[LidarPoint] after every full 360° scan."""
        self._on_scan_callback = callback

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        """Open UART and start background reading thread."""
        if self._running:
            return
        self._ser = serial.Serial(self.port, self.baudrate, timeout=1.0)
        self._ser.flushInput()
        self._running = True
        self._thread  = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print(f"[LidarDriver] Started — {self.port} @ {self.baudrate}")

    # ── Motor control (needs LIDAR RX → Pi TX, GPIO14) ─────────
    # Independent of start()/cleanup(): the UART port stays open and the
    # background thread keeps running the whole time the app is up —
    # these just tell the motor to spin or not, so the sensor isn't
    # wearing out while nobody's looking at the LIDAR screen.

    def _send_command(self, mode: int) -> None:
        if not self._ser or not self._ser.is_open:
            print("[LidarDriver] Can't send command — UART not open "
                  "(call start() first)")
            return
        with self._lock:
            self._ser.write(_build_command(mode))

    def start_motor(self) -> None:
        """Send the Start command — motor spins up, scanning resumes."""
        self._send_command(CMD_MODE_START)
        print("[LidarDriver] Motor START command sent")

    def stop_motor(self) -> None:
        """Send the Stop command — motor stops spinning, saves wear."""
        self._send_command(CMD_MODE_STOP)
        print("[LidarDriver] Motor STOP command sent")

    def cleanup(self) -> None:
        """Stop reading thread and close UART."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._ser and self._ser.is_open:
            self._ser.close()
        print("[LidarDriver] Cleaned up.")

    # ── Read loop ─────────────────────────────────────────────

    def _read_loop(self) -> None:
        """
        Background thread — reads packets using buffer approach
        (verified against kaiaai implementation):
        - Read chunks of 256 bytes
        - Search for valid 0x54 0x2C header
        - Validate CRC before processing
        - Detect full revolution by end_angle < start_angle of next packet
        """
        buf         = bytearray()
        scan_points = []
        last_end_a  = -1

        while self._running:
            try:
                chunk = self._ser.read(256)
                if not chunk:
                    continue
                buf.extend(chunk)

                while len(buf) >= PACKET_SIZE:
                    # Find header
                    if buf[0] != PACKET_HEADER or buf[1] != PACKET_VER_LEN:
                        found = -1
                        for i in range(1, len(buf) - 1):
                            if buf[i] == PACKET_HEADER and buf[i+1] == PACKET_VER_LEN:
                                found = i
                                break
                        if found == -1:
                            del buf[:-1]
                            break
                        del buf[:found]

                    if len(buf) < PACKET_SIZE:
                        break

                    pkt = bytes(buf[:PACKET_SIZE])

                    # CRC check
                    if _crc8(pkt[:-1]) != pkt[-1]:
                        del buf[0]
                        continue

                    # Parse
                    points = _parse_packet(pkt, self.angle_offset)
                    del buf[:PACKET_SIZE]

                    if not points:
                        continue

                    # Keep the points for get_profile() + the sensor's real speed.
                    speed_raw, = struct.unpack_from('<H', pkt, 2)   # deg/sec
                    now = time.time()
                    ts_raw, = struct.unpack_from('<H', pkt, 44)         # LIDAR's own clock, ms
                    self._ingest(points, now, speed_raw, ts_raw)
                    on_packet = self._on_packet
                    if on_packet:
                        on_packet(now, speed_raw, ts_raw, points)

                    # Get current packet end_angle (raw, for revolution detection)
                    end_a_raw,   = struct.unpack_from('<H', pkt, 42)
                    start_a_raw, = struct.unpack_from('<H', pkt, 4)

                    # Detect full revolution: end < start means wrap around 0°
                    if end_a_raw < start_a_raw and scan_points:
                        with self._lock:
                            self._current_points = list(scan_points)
                        if self._on_scan_callback:
                            self._on_scan_callback(self._current_points)
                        scan_points = []

                    scan_points.extend(points)

            except Exception as e:
                if self._running:
                    print(f"[LidarDriver] Warning: {e}")
                time.sleep(0.02)

    # ── Public API ────────────────────────────────────────────

    def get_points(self) -> list:
        """Return all raw points from the latest full scan."""
        with self._lock:
            return list(self._current_points)

    def get_closest(self):
        """Return the closest LidarPoint in the latest scan."""
        with self._lock:
            pts = [p for p in self._current_points if p.distance > 0]
        return min(pts, key=lambda p: p.distance) if pts else None

    @staticmethod
    def _angle_to_zone(angle: float) -> str:
        a = angle % 360
        if a >= 337.5 or a < 22.5:  return 'front[1]'
        elif a < 67.5:               return 'front-right[2]'
        elif a < 112.5:              return 'right[3]'
        elif a < 157.5:              return 'rear-right[4]'
        elif a < 202.5:              return 'rear[5]'
        elif a < 247.5:              return 'rear-left[6]'
        elif a < 292.5:              return 'left[7]'
        else:                        return 'front-left[8]'

    def get_zone_summary(self) -> dict:
        """
        Analyse latest scan by zone.
        Returns: {zone: {points_count, min_distance, wall_detected}}
        wall_detected = True for the dominant zone with >35 points in range.
        """
        with self._lock:
            points = list(self._current_points)

        zones = {z: {'points_count': 0,
                     'min_distance': float('inf'),
                     'wall_detected': False}
                 for z in self.ALERT_DIST}

        for p in points:
            if p.distance < 80:
                continue
            zone  = self._angle_to_zone(p.angle)
            limit = self.ALERT_DIST[zone]
            if p.distance <= limit:
                zones[zone]['points_count'] += 1
                if p.distance < zones[zone]['min_distance']:
                    zones[zone]['min_distance'] = p.distance

        # Mark dominant zone
        dominant = max(zones.items(), key=lambda x: x[1]['points_count'])
        if dominant[1]['points_count'] > 35:
            zones[dominant[0]]['wall_detected'] = True

        for data in zones.values():
            if data['min_distance'] == float('inf'):
                data['min_distance'] = None

        return zones

    def get_ml_vector(self) -> list:
        """
        Returns list of 8 values (min_distance mm or None) per zone.
        Order: front, front-right, right, rear-right,
               rear,  rear-left,  left,  front-left
        """
        summary = self.get_zone_summary()
        order = ['front[1]', 'front-right[2]', 'right[3]', 'rear-right[4]',
                 'rear[5]',  'rear-left[6]',   'left[7]',  'front-left[8]']
        return [summary[z]['min_distance']
                if summary[z]['wall_detected'] else None
                for z in order]

    # ── Distance profile (RUN CAR ML finish check) ────────────

    def _ingest(self, points: list, now: float, speed_deg_s: int = 0, ts_ms: int = None) -> None:
        """Add one packet's points to the rolling history (called by the read thread).
        ts_ms is the LIDAR's own millisecond clock for the packet: packets reach
        the Pi in bursts, so the LIDAR's clock says much better WHEN each one
        was measured."""
        with self._recent_lock:
            if 720 <= speed_deg_s <= 7200:            # 2..20 rev/sec — anything else is junk
                self._rev_time = 360.0 / speed_deg_s
            if ts_ms is None:
                self._lidar_clock = now
            else:
                if self._last_ts is not None:
                    step = (ts_ms - self._last_ts) % 65536
                    self._lidar_clock += (step if step < 500 else 3) / 1000.0
                else:
                    self._lidar_clock = now
                self._last_ts = ts_ms
                # keep the LIDAR clock tied to the Pi clock (the two drift apart slowly): it may
                # never run ahead of it, and a packet is never more than a moment old on arrival
                if self._lidar_clock > now:
                    self._lidar_clock = now
                elif now - self._lidar_clock > 0.06:
                    self._lidar_clock = now - 0.06
            self._recent.append((self._lidar_clock, points))
            limit = self._lidar_clock - PROFILE_KEEP_SECONDS
            while self._recent and self._recent[0][0] < limit:
                self._recent.popleft()

    @property
    def rev_time(self) -> float:
        """Seconds per LIDAR revolution (about 0.167)."""
        return self._rev_time

    def get_last_revolution(self) -> list:
        """
        Every point of the last full revolution (the last ~1/6 s), as
        [(angle_deg, distance_mm, age_s), ...] — angle 0 = straight ahead,
        clockwise; age = how long ago (seconds) the point was measured.
        Empty when the LIDAR is not delivering data. Used by ring_nav.py.
        """
        now = time.time()
        with self._recent_lock:
            if not self._recent or now - self._recent[-1][0] > 0.5:
                return []
            newest  = self._recent[-1][0]
            window  = min(0.40, max(0.10, self._rev_time * 1.05))
            packets = [(t, pts) for (t, pts) in self._recent if newest - t <= window]
        out = []
        for t, pts in packets:
            age = now - t
            for p in pts:
                out.append((p.angle, p.distance, age))
        return out

    def get_profile(self) -> list:
        """
        The shape of the surroundings right now: PROFILE_BINS distances (mm),
        one per 3-degree slice around the car, clockwise from straight ahead
        (slice 0 is -1.5..+1.5 degrees). None where nothing was seen.
        Each value is the middle (median) reading of its slice, taken from
        the points of the last full revolution (~1/6 s). Returns [] when the
        LIDAR is not delivering data (motor stopped).

        Used only by auto_drive.py's finish check; get_ml_vector() and the
        recording are not affected by it.
        """
        now = time.time()
        with self._recent_lock:
            window  = min(0.40, max(0.10, self._rev_time * 1.1))
            packets = [pts for (t, pts) in self._recent if now - t <= window]

        bins  = [[] for _ in range(PROFILE_BINS)]
        count = 0
        for pts in packets:
            for p in pts:
                if p.distance >= PROFILE_MIN_VALID_MM:
                    bins[int(((p.angle + PROFILE_BIN_DEG / 2.0) % 360.0) // PROFILE_BIN_DEG)].append(p.distance)
                    count += 1
        if count < PROFILE_BINS:
            return []

        profile = []
        for ds in bins:
            if ds:
                ds.sort()
                profile.append(ds[len(ds) // 2])
            else:
                profile.append(None)
        return profile

    # ── Self Test ─────────────────────────────────────────────

    def self_test(self, duration: int = 15) -> None:
        """Print live scan data for duration seconds."""
        print(f"\n[LidarDriver] --- Self-test START ({duration} sec) ---")
        count = [0]

        def on_scan(pts):
            count[0] += 1
            if not pts:
                return
            closest = min(pts, key=lambda p: p.distance)
            print(f"  Scan #{count[0]:3d} | pts={len(pts):4d} | "
                  f"closest: {closest.distance:4d}mm @ {closest.angle:6.1f}°")

        self.set_on_scan(on_scan)
        self.start()
        time.sleep(duration)
        self.cleanup()
        print(f"[LidarDriver] Total scans: {count[0]}")
        print("[LidarDriver] --- Self-test COMPLETE ---\n")


if __name__ == "__main__":
    lidar = LidarDriver()
    try:
        lidar.self_test(duration=15)
    except KeyboardInterrupt:
        print("\n[LidarDriver] Interrupted.")
    finally:
        lidar.cleanup()
