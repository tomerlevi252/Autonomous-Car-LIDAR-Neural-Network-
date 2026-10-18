"""
scan_recorder.py
================
Saves everything the LIDAR sees during a drive to a file, for building the
map of the route offline (step 1 of the "drive like a robot vacuum" work).

It does NOT change how the car drives or what goes into car_routes.db:
while the car is moving the LIDAR packets are only kept in memory (one list
append per packet), and the file is written afterwards, in a background
thread.

One file per drive, in the folder  scans/  next to this file:
    scans/scan_REC_20261005_123000.csv.gz    a RECORDING ROUTE drive
    scans/scan_RUN_20261005_123100.csv.gz    a RUN CAR ML drive

File format (gzip-compressed text, one record per line):
    # ...                                    header / comment lines
    E,<time>,<text>                          an event (a driving command, start, stop)
    P,<time>,<speed>,<ts>,<a>,<d>,<i>,...    one LIDAR packet:
         time   Pi clock (time.time()) when the packet arrived
         speed  LIDAR rotation speed, deg/sec
         ts     the LIDAR's own timestamp of the packet, ms (wraps at 65536)
         a,d,i  angle (deg, 0 = straight ahead, clockwise), distance (mm),
                intensity — repeated for every valid point of the packet
"""

import gzip
import os
import threading
import time

SCAN_DIR    = "scans"
MAX_PACKETS = 150000      # about 7.5 minutes of driving — a safety limit on memory


class ScanRecorder:
    """Collects LIDAR packets + events in memory and writes them to a file."""

    def __init__(self):
        self._lock      = threading.Lock()
        self._active    = False
        self._kind      = ""
        self._started   = 0.0
        self._packets   = []
        self._events    = []
        self._truncated = False

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self, kind: str) -> None:
        """Begin a new log. kind: 'REC' (manual recording) or 'RUN' (replay)."""
        with self._lock:
            self._kind      = kind
            self._started   = time.time()
            self._packets   = []
            self._events    = [(self._started, f"START {kind}")]
            self._truncated = False
            self._active    = True

    def add_packet(self, now: float, speed: int, ts_ms: int, points: list) -> None:
        """Called by LidarDriver for every packet (read thread) — must stay cheap."""
        if not self._active:
            return
        if len(self._packets) >= MAX_PACKETS:
            self._truncated = True
            return
        self._packets.append((now, speed, ts_ms, points))

    def add_event(self, text: str, now: float = None) -> None:
        """Note something that happened (e.g. a driving command) with its time."""
        if not self._active:
            return
        with self._lock:
            self._events.append((time.time() if now is None else now, text))

    def stop_and_save(self, note: str = "", on_saved=None) -> None:
        """End the log and write it to scans/ in the background.
        on_saved(path): called (on that background thread) once the file is written —
        V4 uses it to put the drive into the database car_routes.db."""
        with self._lock:
            if not self._active:
                return
            self._active = False
            now = time.time()
            self._events.append((now, f"STOP {self._kind}"))
            packets, events = self._packets, self._events
            kind, started, truncated = self._kind, self._started, self._truncated
            self._packets, self._events = [], []
        threading.Thread(target=self._write, daemon=True,
                         args=(kind, started, now, packets, events, truncated, note, on_saved)).start()

    @staticmethod
    def _write(kind, started, stopped, packets, events, truncated, note, on_saved=None) -> None:
        try:
            folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), SCAN_DIR)
            os.makedirs(folder, exist_ok=True)
            name = f"scan_{kind}_{time.strftime('%Y%m%d_%H%M%S', time.localtime(started))}.csv.gz"
            path = os.path.join(folder, name)
            points = 0
            with gzip.open(path + ".tmp", "wt", encoding="utf-8", errors="replace", compresslevel=4) as f:
                f.write("# CAR ML raw LIDAR log v1\n")
                f.write(f"# kind={kind} started={started:.4f} stopped={stopped:.4f} "
                        f"packets={len(packets)} truncated={int(truncated)}\n")
                if note:
                    f.write(f"# {note}\n")
                f.write("# E,time,text | P,time,speed_deg_s,lidar_ts_ms,angle,dist_mm,intensity,...\n")
                for t, text in events:
                    text = str(text).replace("\n", " ")
                    f.write(f"E,{t:.4f},{text}\n")
                for t, speed, ts_ms, pts in packets:
                    parts = [f"P,{t:.4f},{speed},{ts_ms}"]
                    for p in pts:
                        parts.append(f"{p.angle:.2f},{p.distance},{p.intensity}")
                    points += len(pts)
                    f.write(",".join(parts) + "\n")
            os.replace(path + ".tmp", path)
            print(f"[ScanRecorder] Saved {len(packets)} LIDAR packets ({points} points, "
                  f"{stopped - started:.1f}s), {len(events)} events -> {path}"
                  + ("  (TRUNCATED: drive longer than the memory limit)" if truncated else ""))
        except Exception as e:
            print(f"[ScanRecorder] Could not save the scan log: {e!r}")
            return
        if on_saved:
            try:
                on_saved(path)
            except Exception as e:
                print(f"[ScanRecorder] after saving: {e!r}")
