"""
ml_simulate.py
==============
A VIRTUAL TEST DRIVE of the trained neural network — on a computer (or the Pi),
without the car moving at all.

    python3 ml_simulate.py                         newest manual recording = the track, 90 s
    python3 ml_simulate.py scans/scan_REC_x.csv.gz 120     a chosen track, 120 s

How it works:
  1. The map of the track is rebuilt from one of your recordings (the same
     way scan_view.py does it): every wall point the LIDAR saw.
  2. A pretend car is put on the start point of that recording. A pretend
     LIDAR "sees" the walls of the map from where the pretend car stands
     (720 rays, like the real D200, with a little noise).
  3. The REAL driving code (ml_driver.py, with the real trained network
     ml_model.npz and the safety layer) drives the pretend car. A tap moves it
     as much as your taps moved the real car (measured from your recordings:
     forward ~360 mm/s, turning ~100 deg/s), with random variation.
  4. Result: how many laps, how many times it touched a wall, and a picture:
     ml_data/sim_<track>.png

This is how you can check a newly trained network BEFORE putting the car on
the floor — and compare models for the report (PDF section 11).
It is only a simulation: the real car will behave a little differently.
"""

import glob
import math
import os
import random
import sys

import numpy as np

import ml_driver
import scan_view as sv

HERE = os.path.dirname(os.path.abspath(__file__))

CELL_MM        = 10.0      # map resolution
CAR_RADIUS_MM  = 75.0      # a wall closer than this to the LIDAR centre = touching (your own drives
                           # pass walls at 85 mm, so the real car is a bit smaller than that)
FWD_MM_S       = 360.0     # measured from the recordings (a 0.11 s tap = ~41 mm)
TURN_DEG_S     = 100.0     # measured (a 0.11 s tap = ~11 deg)
NOISE          = 0.25      # +/- 25 % random variation of every tap (floor, battery ...)
LIDAR_RAYS     = 720
LIDAR_NOISE_MM = 5.0
LIDAR_MAX_MM   = 3000.0


class Clock:
    """Simulated time: sleep() does not wait, it just moves the clock on."""

    def __init__(self):
        self.t = 1_000_000.0

    def time(self):
        return self.t

    def sleep(self, dt):
        self.t += max(0.0, dt)

    def __getattr__(self, name):          # anything else from the time module
        import time as _t
        return getattr(_t, name)


class World:
    def __init__(self, cloud, start=(0.0, 0.0, 0.0), seed=0):
        pts = np.array(cloud, dtype=np.float64)
        self.x0, self.y0 = pts[:, 0].min() - 200, pts[:, 1].min() - 200
        nx = int((pts[:, 0].max() + 200 - self.x0) / CELL_MM) + 1
        ny = int((pts[:, 1].max() + 200 - self.y0) / CELL_MM) + 1
        ix = ((pts[:, 0] - self.x0) / CELL_MM).astype(int)
        iy = ((pts[:, 1] - self.y0) / CELL_MM).astype(int)
        hits = np.zeros((nx, ny), dtype=np.int32)
        np.add.at(hits, (ix, iy), 1)
        occ = hits >= 2                                     # seen at least twice = a real wall ...
        near = np.zeros(occ.shape, dtype=np.int32)          # ... and not a lonely stray point
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                near += np.roll(np.roll(occ, dx, 0), dy, 1)
        self.occ = occ & (near >= 5)
        # the LIDAR never sees closer than 80 mm, so "wall" right where the recording started
        # is a mapping error: clear it so the car can start
        gx, gy = np.meshgrid(np.arange(nx) * CELL_MM + self.x0, np.arange(ny) * CELL_MM + self.y0, indexing='ij')
        self.occ &= np.hypot(gx - start[0], gy - start[1]) > CAR_RADIUS_MM + 10
        self.wall_xy = np.argwhere(self.occ) * CELL_MM + [self.x0, self.y0]
        self.x, self.y, self.th = start
        self.path = [(self.x, self.y, self.th)]
        self.touches = 0
        self.rng = random.Random(seed)
        self.start = (self.x, self.y)
        self.laps, self._away = 0, False                    # a lap = back near the start after being far away
        self.distance = 0.0

    def clearance(self, x, y):
        d = np.hypot(self.wall_xy[:, 0] - x, self.wall_xy[:, 1] - y)
        return float(d.min())

    def move(self, fwd_mm, turn_deg):
        """Move in small steps; a step that would bring the car into a wall is blocked (= a touch)."""
        steps = max(1, int(max(abs(fwd_mm) / 5.0, abs(turn_deg) / 2.0)))
        here = self.clearance(self.x, self.y)
        for _ in range(steps):
            th = self.th + turn_deg / steps
            r = math.radians(th)
            nx = self.x + fwd_mm / steps * math.sin(r)
            ny = self.y + fwd_mm / steps * math.cos(r)
            there = self.clearance(nx, ny) if fwd_mm else here
            if there < CAR_RADIUS_MM and there < here:
                self.touches += 1                         # bumped into the wall: stays where it is
                self.th = th                              # (turning on the spot still works)
                break
            here = there
            self.distance += abs(fwd_mm) / steps
            self.x, self.y, self.th = nx, ny, th
            from_start = math.hypot(self.x - self.start[0], self.y - self.start[1])
            if from_start > 600.0:
                self._away = True
            elif self._away and from_start < 150.0:
                self.laps += 1
                self._away = False
        self.path.append((self.x, self.y, self.th))

    def scan(self):
        """What the LIDAR sees from here: [(angle_deg, distance_mm, age)] (0 = ahead, clockwise)."""
        a = np.arange(LIDAR_RAYS) * 360.0 / LIDAR_RAYS
        w = np.radians(a + self.th)
        d = np.arange(20.0, LIDAR_MAX_MM, CELL_MM / 2)
        px = self.x + np.outer(np.sin(w), d)
        py = self.y + np.outer(np.cos(w), d)
        ix = ((px - self.x0) / CELL_MM).astype(int)
        iy = ((py - self.y0) / CELL_MM).astype(int)
        inside = (ix >= 0) & (iy >= 0) & (ix < self.occ.shape[0]) & (iy < self.occ.shape[1])
        hit = np.zeros_like(inside)
        hit[inside] = self.occ[ix[inside], iy[inside]]
        first = hit.argmax(axis=1)
        out = []
        for k in range(LIDAR_RAYS):
            if hit[k, first[k]]:
                out.append((float(a[k]), int(d[first[k]] + self.rng.gauss(0, LIDAR_NOISE_MM)), 0.0))
        return out


class SimMotors:
    """Same methods as MotorDriverL298n; moves the World instead of real wheels."""

    def __init__(self, world, clock):
        self.w, self.clock = world, clock
        self.state, self.since = ('S', 0.0, 0.0), clock.time()
        self.default_speed = 100

    def _settle(self):
        dt = self.clock.time() - self.since
        kind, v, om = self.state
        if dt > 0 and kind != 'S':
            k = 1.0 + self.w.rng.uniform(-NOISE, NOISE)
            self.w.move(v * dt * k, om * dt * k)
        self.since = self.clock.time()

    def _set(self, kind, v, om):
        self._settle()
        self.state = (kind, v, om)

    def forward(self, speed=None):   self._set('F', FWD_MM_S, 0.0)
    def backward(self, speed=None):  self._set('B', -FWD_MM_S, 0.0)
    def turn_left(self, speed=None): self._set('L', 0.0, -TURN_DEG_S)
    def turn_right(self, speed=None): self._set('R', 0.0, TURN_DEG_S)
    def stop(self):                  self._set('S', 0.0, 0.0)

    def drive(self, left, right):
        v = (left + right) / 200.0 * FWD_MM_S
        om = (left - right) / 200.0 * TURN_DEG_S
        self._set('D', v, om)


class SimLidar:
    def __init__(self, world, motors):
        self.w, self.m = world, motors

    def get_last_revolution(self):
        self.m._settle()                     # the car is where it is NOW
        return self.w.scan()


def draw(world, path_png, title, lines):
    from PIL import Image, ImageDraw
    S = 900
    xs, ys = world.wall_xy[:, 0], world.wall_xy[:, 1]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    sc = (S - 140) / max(x1 - x0, y1 - y0)

    def px(x, y):
        return 60 + (x - x0) * sc, 110 + (y1 - y) * sc
    img = Image.new("RGB", (S, S), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for x, y in world.wall_xy[::2]:
        u, v = px(x, y)
        d.rectangle([u, v, u + 1, v + 1], fill=(70, 70, 70))
    pts = [px(p[0], p[1]) for p in world.path]
    if len(pts) > 1:
        d.line(pts, fill=(31, 119, 180), width=3)
    u, v = pts[0]
    d.ellipse([u - 9, v - 9, u + 9, v + 9], fill=(0, 170, 0))
    u, v = pts[-1]
    d.rectangle([u - 7, v - 7, u + 7, v + 7], fill=(220, 0, 0))
    d.text((20, 15), title, fill=(0, 0, 0))
    for i, t in enumerate(lines):
        d.text((20, 38 + 18 * i), t, fill=(60, 60, 60))
    img.save(path_png)


def simulate(track_file, seconds=90.0, model_path=None, seed=0, quiet=True, mode=None):
    points, events, _ = sv.read_scan(track_file)
    looks = sv._Looks(points)
    m, way = sv.build_from_scans(looks, events, points[0][0], points[-1][0])
    world = World(m.cloud, start=way[0][:3] if way else (0.0, 0.0, 0.0), seed=seed)
    # start where your recording started — or, if the car stood squeezed against a wall
    # there, at the first place of your drive with a little room around it
    for p in way:
        if world.clearance(p[0], p[1]) >= CAR_RADIUS_MM + 25:
            if (p[0], p[1]) != (world.x, world.y):
                world = World(m.cloud, start=p[:3], seed=seed)
            break
    clock = Clock()
    motors = SimMotors(world, clock)
    lidar = SimLidar(world, motors)

    saved = (ml_driver.time, ml_driver.START_DELAY_S, ml_driver.MAX_RUN_S, ml_driver.MODE)
    ml_driver.time = clock
    ml_driver.START_DELAY_S = 0.2
    ml_driver.MAX_RUN_S = seconds
    if mode:
        ml_driver.MODE = mode
    try:
        drv = ml_driver.MLDriver(motors, lidar, model_path or ml_driver.MODEL_PATH,
                                 log=(lambda s: None) if quiet else print)
        if not drv.load():
            return None
        drv._running = True
        drv._loop()
    finally:
        ml_driver.time, ml_driver.START_DELAY_S, ml_driver.MAX_RUN_S, ml_driver.MODE = saved
    return {'laps': world.laps, 'touches': world.touches, 'distance_m': world.distance / 1000.0,
            'decisions': drv.steps, 'world': world, 'seconds': seconds}


def main(argv):
    files = sorted(glob.glob(os.path.join(HERE, "scans", "scan_REC_*.csv.gz")))
    track = argv[0] if argv else (files[-1] if files else None)
    seconds = float(argv[1]) if len(argv) > 1 else 90.0
    if not track:
        print("No recording to build the track from.")
        return 1
    print(f"Virtual test drive on the track of {os.path.basename(track)}, {seconds:.0f} s simulated ...")
    r = simulate(track, seconds, quiet=False)
    if r is None:
        return 1
    name = os.path.basename(track).replace(".csv.gz", "")
    out = os.path.join(HERE, "ml_data", f"sim_{name}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    lines = [f"{r['seconds']:.0f} s simulated: {r['laps']} laps, {r['distance_m']:.1f} m, "
             f"{r['decisions']} decisions, touched a wall {r['touches']} times",
             "grey = walls (from the recording), blue = the network's drive, green = start, red = end"]
    draw(r['world'], out, f"CAR ML - virtual test drive of the neural network ({name})", lines)
    print(lines[0])
    print(f"picture -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
