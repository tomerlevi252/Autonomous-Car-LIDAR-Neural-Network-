"""
ml_dataset.py
=============
STEP 2 of the PDF — "Dataset collection".
Turns the recorded drives in  scans/  into the training table for the
neural network:   LIDAR picture  ->  what the driver pressed.

Where the data comes from (V4: the database car_routes.db)
---------------------------------------------------------
Every time you drive the car on the RECORDING ROUTE screen (HORN = start /
stop), the drive is saved in the database car_routes.db:

    route_sessions  one row = one drive ("Run" in the sense of PDF section 8)
    action_events   every button press: which button, when, for how long
                    -> the LABELS ("what the driver did")
    ml_samples      10 times a second: what the LIDAR saw (the 16 inputs and
                    the whole revolution) -> the INPUTS ("what the car saw")

The training table is built ONLY from these tables. (Every drive is also kept
as a raw backup file scans/scan_REC_<date>_<time>.csv.gz. Backup files that are
not in the database yet — e.g. the drives recorded before V4 — are copied
into the database automatically the first time; see import_scans().)

What this script makes
----------------------
For every LIDAR picture in ml_samples (one per 0.1 s = 10 Hz, as in the PDF)
it computes the network's inputs with ml_features.py and puts them next to
the LABEL = what the driver did (from action_events):

    ml_data/dataset.npz          everything, for ml_train.py
    ml_data/csv/<run>.csv        the same as readable tables (open in Excel):
        timestamp, s00_0deg, ..., s15_338deg, button, joy_x, joy_y, action

The labels — "the joystick provides the labels" (PDF sections 1, 4, 12)
-----------------------------------------------------------------------
Our app has buttons, not a joystick. Each button is turned into a joystick
position (PDF section 4: "record the raw joystick X/Y values"):

    button      joy_x (steering)   joy_y (throttle)
    F forward         0                 +1
    B backward        0                 -1
    L left           -1                  0      (turn on the spot)
    R right          +1                  0
    G fwd-left      -0.5               +0.5     (the app's diagonal buttons)
    H fwd-right     +0.5               +0.5
    I / J back-left / back-right   -/+0.5  -0.5

IMPORTANT — you drive in short TAPS (F for ~0.12 s, stop, look, tap again).
One single tap is a noisy label: on the same picture the driver sometimes taps
F and sometimes R. What IS clear is what he does over the next moment. So the
label of a sample is the AVERAGE joystick over the next LOOKAHEAD_S seconds
of driving (only the time the car was really moving counts):

    next second:  F F R F F R  ->  joy_x = +0.33 (a third of the time turning right),
                                   joy_y = +0.67

  joy_x / joy_y are the labels of the joystick model (PDF sections 12-13).
  "action" = the same thing as ONE button (ml_model.joystick_to_action):
      turning more than a third of the time to the right (joy_x > 0.33) -> R,
      to the left (joy_x < -0.33) -> L, going back -> B, otherwise F
  — the label of the classification model (PDF section 7).
  A sample where the car does not move at all in the next second (the driver
  had a pause) teaches nothing and is skipped.

Recovery augmentation (V3: the network must drive with NO safety layer)
-----------------------------------------------------------------------
In your recordings the car is nearly always well placed in the lane, so the
network never sees "I am too close to the wall / at an angle — what now?".
Once it drifts a little off (it always does), it is somewhere it has never
learned about. The standard cure (NVIDIA's self-driving car "PilotNet" does
the same with side cameras): from every recorded LIDAR picture a few extra
pictures are computed as if the car stood a bit to the side (up to
AUG_MAX_SHIFT_MM) and turned a bit (up to AUG_MAX_ROT_DEG). With a LIDAR this
is exact geometry: every point is just moved and turned. The label of such a
picture is YOUR steering plus the steering that brings the car back onto your
line. This way the network itself learns to recover — no hand-written wall
rules are needed while driving. Marked augmented=1 in the file; they are
only used for training, never for testing.

Mirror augmentation
-------------------
Most of the recordings go clockwise. To teach the network both directions,
every sample is also stored mirrored (left <-> right: joy_x -> -joy_x, L <-> R).
In the file they are marked mirrored=1, so ml_train.py can switch them off.

How to use (in the car_ml folder, on the Pi or on a computer):
    python3 ml_dataset.py            build the training table from car_routes.db
                                     (first copies scans/ files that are not in the database yet)
    python3 route_database.py        list the drives in the database
    python3 route_database.py delete 7    delete drive (session) number 7 — e.g. a bad one
"""

import glob
import json
import gzip
import os
import sys
import zlib

import numpy as np

import ml_features as mf
from ml_model import joystick_to_action

SCAN_DIR          = "scans"
OUT_DIR           = "ml_data"
SAMPLE_PERIOD_S   = 0.10     # one sample every 100 ms (PDF: "every 50-100 ms")
LOOKAHEAD_S       = 1.0      # label = what the driver did in the next second
MIN_MOVING_S      = 0.05     # less driving than this in the next second = a pause -> sample skipped
MIN_SAMPLES       = 50       # a recording with fewer samples than this is skipped (a mis-press)
MIRROR            = True     # also store every sample mirrored left <-> right

# Recovery augmentation (V3) — see the top of this file
AUGMENT_COPIES    = 4        # extra "the car is a bit off" examples per real sample (0 = off)
AUG_MAX_ROT_DEG   = 30.0     # turned up to this much to either side ...
AUG_MAX_SHIFT_MM  = 90.0     # ... and moved up to this much to either side
AUG_LOOKAHEAD_MM  = 300.0    # the way back to your line is aimed this far ahead
AUG_DEG_PER_JOY   = 30.0     # how firmly the label steers back: 30 degrees of error -> joy_x one full
                             # step (1.0) to the other side. Tested in the simulator: 70 / 45 were too
                             # soft (the network noticed being off course too late), 30 never got stuck

# app button -> joystick position (joy_x = steering, joy_y = throttle)
JOYSTICK = {'F': (0.0, 1.0), 'B': (0.0, -1.0), 'L': (-1.0, 0.0), 'R': (1.0, 0.0),
            'G': (-0.5, 0.5), 'H': (0.5, 0.5), 'I': (-0.5, -0.5), 'J': (0.5, -0.5)}
ACTION_MAP = {'F': 'F', 'B': 'B', 'L': 'L', 'R': 'R', 'S': 'S',      # for the button hold times
              'G': 'L', 'H': 'R', 'I': 'B', 'J': 'B'}
MIRROR_ACTION = {'F': 'F', 'B': 'B', 'S': 'S', 'L': 'R', 'R': 'L'}



# ══════════════════════════════════════════════════════════════
#  Reading one recording
# ══════════════════════════════════════════════════════════════

class _Clock:
    """
    The SAME time keeping as LidarDriver._ingest(): packets reach the Pi in
    bursts, so the LIDAR's own millisecond clock is followed, tied to the Pi
    clock. Copied here so the revolutions cut out of a recording are exactly
    the ones LidarDriver.get_last_revolution() would have returned live.
    """

    def __init__(self):
        self.clock, self.last_ts, self.rev_time = 0.0, None, 1.0 / 6.0

    def step(self, now, speed, ts):
        if 720 <= speed <= 7200:
            self.rev_time = 360.0 / speed
        if self.last_ts is not None:
            st = (ts - self.last_ts) % 65536
            self.clock += (st if st < 500 else 3) / 1000.0
        else:
            self.clock = now
        self.last_ts = ts
        if self.clock > now:
            self.clock = now
        elif now - self.clock > 0.06:
            self.clock = now - 0.06
        return self.clock


def read_recording(path):
    """-> (packets [(t, rev_time, [(angle, dist), ...])], commands [(t, 'F'), ...])"""
    packets, commands = [], []
    clk = _Clock()
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("P,"):
                a = line.rstrip().split(",")
                try:
                    now, speed, ts = float(a[1]), int(a[2]), int(a[3])
                    v = a[4:]
                    pts = [(float(v[i]), int(v[i + 1])) for i in range(0, len(v) - 2, 3)]
                except (ValueError, IndexError):
                    continue
                t = clk.step(now, speed, ts)
                packets.append((t, clk.rev_time, pts))
            elif line.startswith("E,"):
                parts = line.rstrip("\n").split(",", 2)
                if len(parts) == 3 and parts[2].startswith("CMD "):
                    try:
                        commands.append((float(parts[1]), parts[2][4:].strip()))
                    except ValueError:
                        pass
    commands.sort()
    return packets, commands


def label_window(t, segments, i_seg):
    """
    What the driver did during [t, t + LOOKAHEAD_S]:
    -> (joy_x, joy_y, action) or None when the car hardly moved (= skip).
    segments = [(start, end, button), ...];  i_seg = index of the segment active at t.
    """
    t_end = t + LOOKAHEAD_S
    moving = jx = jy = 0.0
    for a0, a1, btn in segments[i_seg:]:
        if a0 >= t_end:
            break
        if btn not in JOYSTICK:                       # 'S' or unknown: not moving
            continue
        dt = min(a1, t_end) - max(a0, t)
        if dt <= 0:
            continue
        moving += dt
        jx += dt * JOYSTICK[btn][0]
        jy += dt * JOYSTICK[btn][1]
    if moving < MIN_MOVING_S:
        return None
    jx, jy = jx / moving, jy / moving
    return jx, jy, joystick_to_action(jx, jy)


def pulse_lengths(commands):
    """How long each (non-S) button was held: {'F': [0.12, 0.10, ...], ...}"""
    out = {}
    for (t, a), (t2, _) in zip(commands, commands[1:]):
        c = ACTION_MAP.get(a, 'S')
        if c != 'S' and 0.02 <= t2 - t <= 3.0:
            out.setdefault(c, []).append(t2 - t)
    return out


def augment(points, jx, jy, rng, n_sectors, copies=None):
    """recovery copies of one picture: [(features, joy_x, joy_y, action), ...]"""
    copies = AUGMENT_COPIES if copies is None else copies
    if copies <= 0:
        return []
    a = np.array([p[0] for p in points], dtype=np.float64)
    d = np.array([p[1] for p in points], dtype=np.float64)
    out = []
    for _ in range(copies):
        rot = rng.uniform(-AUG_MAX_ROT_DEG, AUG_MAX_ROT_DEG)
        shift = rng.uniform(-AUG_MAX_SHIFT_MM, AUG_MAX_SHIFT_MM)
        a2, d2 = mf.shift_points(a, d, rot, shift)
        # turned right / standing to the right of your line -> steer back to the left
        back_deg = rot + np.degrees(np.arctan2(shift, AUG_LOOKAHEAD_MM))
        jx2 = float(np.clip(jx - back_deg / AUG_DEG_PER_JOY, -1.0, 1.0))
        jy2 = float(np.clip(jy - (abs(jx2) - abs(jx)), -1.0, 1.0))     # more turning = less forward
        out.append((mf.features_from_arrays(a2, d2, n_sectors), jx2, jy2, joystick_to_action(jx2, jy2)))
    return out


# ══════════════════════════════════════════════════════════════
#  Recording -> database
# ══════════════════════════════════════════════════════════════

def pictures_from_recording(path):
    """
    One LIDAR picture every SAMPLE_PERIOD_S of a scans/ file:
    -> (pictures [(t, [(angle, dist), ...]), ...], commands [(t, button), ...], end_time)
    Each picture is the last full revolution, exactly as LidarDriver.get_last_revolution()
    gives it while driving.
    """
    packets, commands = read_recording(path)
    if not packets or not commands:
        return [], commands, None
    pictures = []
    next_t = packets[0][0] + 0.5                          # the first half second: LIDAR still spinning up
    start = 0
    for k, (t, rev_time, _) in enumerate(packets):
        if t < next_t or t < commands[0][0]:
            continue
        next_t = t + SAMPLE_PERIOD_S
        window = min(0.40, max(0.10, rev_time * 1.05))   # same as get_last_revolution()
        while packets[start][0] < t - window:
            start += 1
        points = [p for (_, _, pts) in packets[start:k + 1] for p in pts]
        if len(points) >= 200:                            # else the LIDAR did not deliver a full picture
            pictures.append((t, points))
    return pictures, commands, packets[-1][0]


def encode_points(points) -> bytes:
    """A LIDAR revolution -> compact bytes for the database (angle in 0.01 deg, distance in mm)."""
    arr = np.array([(int(round(a * 100)) % 36000, min(int(d), 65535)) for a, d, *_ in points],
                   dtype=np.uint16)
    return zlib.compress(arr.tobytes(), 6)


def decode_points(blob):
    """bytes from the database -> (angles_deg, distances_mm) as NumPy arrays."""
    arr = np.frombuffer(zlib.decompress(blob), dtype=np.uint16).reshape(-1, 2)
    return arr[:, 0].astype(np.float64) / 100.0, arr[:, 1].astype(np.float64)


def store_recording(db, path, session_id=None, quiet=True):
    """
    Put one recorded drive into the database.
      session_id given (a new recording from the RECORDING ROUTE screen): its button
        presses are already in action_events — only the LIDAR pictures are added.
      session_id None (an older scans/ file): a new session is made and its button
        presses are written to action_events too.
    Returns (session_id, number of LIDAR pictures stored).
    """
    name = os.path.basename(path)
    pictures, commands, t_end = pictures_from_recording(path)
    if not pictures:
        db.mark_imported(name, None)
        return None, 0
    if session_id is None:
        session_id = db.start_session(name.replace(".csv.gz", ""))
        ends = [t for t, _ in commands[1:]] + [t_end]
        db.log_action_events(session_id, [{'action': a, 'start': t, 'duration': max(0.0, e - t), 'zones': None}
                                          for (t, a), e in zip(commands, ends)])
    rows = [(t, mf.features_from_points(pts, 16), encode_points(pts)) for t, pts in pictures]
    db.add_ml_samples(session_id, rows)
    db.mark_imported(name, session_id)
    if not quiet:
        print(f"  {name} -> database session #{session_id}: {len(rows)} LIDAR pictures")
    return session_id, len(rows)


def import_scans(db, quiet=False):
    """Copy every scans/scan_REC_*.csv.gz that is not in the database yet (done once per file)."""
    here = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(here, SCAN_DIR, "scan_REC_*.csv.gz")))
    new = [f for f in files if not db.is_imported(os.path.basename(f))]
    if new and not quiet:
        print(f"Copying {len(new)} recorded drive(s) from scans/ into the database ...")
    for f in new:
        try:
            store_recording(db, f, None, quiet)
        except (OSError, EOFError) as e:
            if not quiet:
                print(f"  {os.path.basename(f)}: cannot read ({e}) - skipped")
    return len(new)


# ══════════════════════════════════════════════════════════════
#  Database -> training samples
# ══════════════════════════════════════════════════════════════

def samples_from_db(db, session_id, n_sectors=mf.N_SECTORS, rng=None, augment_copies=None):
    """
    The training samples of ONE drive, from the database:
    -> list of (t, features, button, joy_x, joy_y, action, [recovery copies])
    Inputs from ml_samples, labels from action_events.
    """
    segments = sorted(db.get_segments(session_id))
    pictures = db.get_ml_pictures(session_id)
    if not segments or not pictures:
        return []
    rng = rng if rng is not None else np.random.default_rng(session_id)
    last_end = max(e for _, e, _ in segments)
    out, i_seg = [], 0
    for t, blob in pictures:
        if t < segments[0][0] or t > last_end - LOOKAHEAD_S:
            continue
        while i_seg + 1 < len(segments) and segments[i_seg + 1][0] <= t:
            i_seg += 1
        lab = label_window(t, segments, i_seg)
        if lab is None:
            continue
        a, d = decode_points(blob)
        points = list(zip(a, d))
        out.append((t, mf.features_from_arrays(a, d, n_sectors), segments[i_seg][2], *lab,
                    augment(points, lab[0], lab[1], rng, n_sectors, augment_copies)))
    return out


def samples_from_recording(path, n_sectors=mf.N_SECTORS, pulses=None, augment_copies=None):
    """The same samples straight from a scans/ file (for checking; training uses the database)."""
    pictures, commands, t_end = pictures_from_recording(path)
    if not pictures:
        return []
    if pulses is not None:
        for c, ds in pulse_lengths(commands).items():
            pulses.setdefault(c, []).extend(ds)
    segments = [(t, t2, a) for (t, a), (t2, _) in zip(commands, commands[1:])]
    segments.append((commands[-1][0], t_end, commands[-1][1]))
    rng = np.random.default_rng(zlib.crc32(os.path.basename(path).encode()))
    out, i_seg = [], 0
    for t, points in pictures:
        if t > t_end - LOOKAHEAD_S:
            continue
        while i_seg + 1 < len(segments) and segments[i_seg + 1][0] <= t:
            i_seg += 1
        lab = label_window(t, segments, i_seg)
        if lab is None:
            continue
        out.append((t, mf.features_from_points(points, n_sectors), segments[i_seg][2], *lab,
                    augment(points, lab[0], lab[1], rng, n_sectors, augment_copies)))
    return out


# ══════════════════════════════════════════════════════════════
#  Building the whole dataset (from the database)
# ══════════════════════════════════════════════════════════════

def build(n_sectors=mf.N_SECTORS, out_dir=None, quiet=False, db=None):
    from route_database import RouteDatabase
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = out_dir or os.path.join(here, OUT_DIR)
    os.makedirs(os.path.join(out_dir, "csv"), exist_ok=True)
    db = db or RouteDatabase("car_routes.db")
    import_scans(db, quiet)
    sessions = db.ml_sessions()
    if not quiet:
        print(f"Reading the training data from the database: {len(sessions)} recorded drive(s)")

    X, y, joy, run_id, mirrored, augmented, times, run_names = [], [], [], [], [], [], [], []
    pulses = {}
    for sid, name, _n in sessions:
        rng = np.random.default_rng(zlib.crc32(f"{name}#{sid}".encode()))   # same copies every time
        samples = samples_from_db(db, sid, n_sectors, rng)
        if len(samples) < MIN_SAMPLES:
            if not quiet:
                print(f"  session #{sid} {name}: only {len(samples)} samples - skipped")
            continue
        r = len(run_names)
        run_names.append(f"#{sid} {name}")
        for s0, e0, a0 in db.get_segments(sid):
            c = ACTION_MAP.get(a0, 'S')
            if c != 'S' and 0.02 <= e0 - s0 <= 3.0:
                pulses.setdefault(c, []).append(e0 - s0)
        counts = {}
        # the readable CSV copy is only for looking at the data — if it cannot be written
        # (e.g. the file belongs to root after a 'sudo' run), training goes on without it
        try:
            f = open(os.path.join(out_dir, "csv", f"session_{sid:03d}_{name}.csv"), "w")
            f.write("timestamp," + ",".join(mf.sector_names(n_sectors)) + ",button,joy_x,joy_y,action\n")
        except OSError as e:
            f = None
            if not quiet:
                print(f"  (CSV copy not written: {e.strerror} - training goes on)")
        try:
            t0 = samples[0][0]
            for t, vec, button, jx, jy, lab, augs in samples:
                if f:
                    f.write(f"{t - t0:.2f}," + ",".join(f"{v:.3f}" for v in vec)
                            + f",{button},{jx:.3f},{jy:.3f},{lab}\n")
                counts[lab] = counts.get(lab, 0) + 1
                for k, (v, ax, ay, al) in enumerate([(vec, jx, jy, lab)] + augs):
                    X.append(v); y.append(al); joy.append((ax, ay))
                    run_id.append(r); mirrored.append(0); augmented.append(int(k > 0)); times.append(t)
                    if MIRROR:
                        X.append(mf.mirror_features(v)); y.append(MIRROR_ACTION[al]); joy.append((-ax, ay))
                        run_id.append(r); mirrored.append(1); augmented.append(int(k > 0)); times.append(t)
        finally:
            if f:
                f.close()
        if not quiet:
            c = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            print(f"  Run {r + 1:2d}  session #{sid:3d} {name}: {len(samples):5d} samples   {c}")

    if not X:
        print("No usable recorded drives in the database. Record drives on the RECORDING ROUTE screen first.")
        return None
    # the typical (median) time the driver held each button — ml_driver.py taps the same way
    pulse_s = {c: float(np.median(ds)) for c, ds in pulses.items() if ds}
    for c, m in MIRROR_ACTION.items():
        if c in pulse_s and m not in pulse_s:
            pulse_s[m] = pulse_s[c]
    path = os.path.join(out_dir, "dataset.npz")
    np.savez_compressed(path,
                        X=np.array(X, dtype=np.float32),
                        y=np.array(y),
                        joy=np.array(joy, dtype=np.float32),
                        run_id=np.array(run_id, dtype=np.int32),
                        mirrored=np.array(mirrored, dtype=np.int8),
                        augmented=np.array(augmented, dtype=np.int8),
                        t=np.array(times),
                        run_names=np.array(run_names),
                        n_sectors=n_sectors, max_mm=mf.MAX_MM,
                        pulse_s=json.dumps(pulse_s))
    n_real = sum(1 for m, g in zip(mirrored, augmented) if not m and not g)
    if not quiet:
        print(f"\nDataset: {len(run_names)} runs, {n_real} samples"
              + (f" (+{n_real} mirrored)" if MIRROR else "")
              + (f" (+{n_real * AUGMENT_COPIES * (2 if MIRROR else 1)} recovery copies)" if AUGMENT_COPIES else "")
              + f", {n_sectors} LIDAR inputs each -> {path}")
        print("Typical button hold times: " + "  ".join(f"{c}={v:.2f}s" for c, v in sorted(pulse_s.items())))
    return path


if __name__ == "__main__":
    print("Building the ML dataset from the database car_routes.db ...")
    build()
