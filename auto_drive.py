"""
auto_drive.py
=============
"RUN CAR ML".

NAV_MODE = 'ml' (default): the TRAINED NEURAL NETWORK drives the car
  (ml_driver.py, model file ml_model.npz made by the TRAIN button / ml_train.py).
  No recorded route is needed — the network reacts to what the LIDAR sees.

NAV_MODE = 'lidar': the hand-written (rule-based, no learning) navigation —
  kept for comparison (PDF section 11: rule-based vs. neural network):
  - the plain ring track (a closed ring with a small cluster in the middle)
    is driven by ring_nav.py;
  - any other closed track without junctions (a "0" with a wall in the
    middle, an oval ...) is driven by track_nav.py.
Everything below describes the older TIMED replay, which is still here as
the last fallback (NAV_MODE = 'timed', or when the LIDAR sees no way to drive).

TIMED REPLAY — replays a route that was previously recorded (via
route_database.py) by driving the motors through the exact same F/B/L/R/S sequence.

THE ROUTE ITSELF IS REPLAYED BY TIME ONLY: every recorded action is held for
exactly its recorded duration. Nothing the LIDAR sees changes a segment.

The LIDAR is used for one thing — the FINISH CHECK of a closed route (one
that ends where it starts):

  1. Before the first move, while the car is still standing, the replay
     remembers what the LIDAR sees: the distance profile all around the car.
     That is the START POINT.
  2. The route is driven, by time, exactly as recorded.
  3. After the last move the car stands still and looks again. If it sees
     the start point's surroundings (= the route is a closed one), it
     compares the two profiles and works out where it stands relative to
     the start point: how far short of it / past it, how far to the side,
     and how many degrees it is turned. It then closes the forward/backward
     gap with a few short pulses, measuring after each one, until it stands
     level with the start point, and turns back to the direction it started
     in (FINISH_FIX_HEADING) — so the next run starts from the same pose.

A timed replay can't end exactly on the start point by itself: the last
metres of a route are usually short taps, and a 0.1 s tap moves this car
anything between a few mm and a few cm. The finish check does what the
driver did while recording — look, and tap until the car is there.

The check only moves the car forward/backward along its own direction and
turns it on the spot (it cannot move it sideways). It does nothing, and
says so, when: the place at the end does not look like
the start point (an open route), the gap is implausibly large, or the LIDAR
gives no data. It does not use anything from the recording, so it works
with any recording, old or new. FINISH_CHECK = False turns it off.

Usage:
    auto = AutoDrive(motors, route_db, lidar)   # lidar is optional
    started = auto.start()      # runs in a background thread; False if no route found
    ...
    auto.stop()                 # cancel early (e.g. user pressed BACK/STOP on screen)
"""

import math
import threading
import time
from motor_driver_L298n import MotorDriverL298n
from route_database import RouteDatabase
from ring_nav import RingDriver
from track_nav import TrackDriver
from ml_driver import MLDriver

# ── How RUN CAR ML drives ──────────────────────────────────────
#   'ml'    — the trained neural network (ml_driver.py). THE MACHINE-LEARNING MODE.
#   'lidar' — LIDAR navigation (hand-written rules, no learning): the car drives by what it SEES and where it IS.
#             From the recording it takes only the number of laps (and, on the
#             ring, the direction). The plain ring track is driven by ring_nav.py,
#             every other closed track without junctions by track_nav.py. Only if
#             the LIDAR sees no way to drive at all does it fall back to 'timed'.
#   'timed' — the old replay: every recorded action for its recorded time.
NAV_MODE = 'ml'
LAPS     = 1      # None = as many laps as were recorded; or a number, e.g. 1
# Estimating the recorded laps from the recorded commands (measured on this
# car): a turn pulse of T seconds turns it about 101*T - 7 degrees, and every
# forward pulse drifts about 3 degrees in the direction of travel.
TURN_DEG_PER_S, TURN_DEAD_DEG, FORWARD_DRIFT_DEG = 101.0, 7.0, 3.0

# Fallback hold time, only used if a recorded step has no duration for some
# reason (e.g. data from an old recording).
DEFAULT_STEP_DURATION = 0.1

POLL_INTERVAL = 0.02          # sleep slice while holding a segment (keeps Cancel responsive)

# ── Finish check (closed routes) ───────────────────────────────
FINISH_CHECK        = True    # False = plain timed replay, the LIDAR is not used at all
LIDAR_READY_TIMEOUT = 4.0     # wait at most this long for the LIDAR to spin up before the route
ALIGN_SETTLE_S      = 0.40    # stand still this long before measuring (stop rolling + a full fresh scan)
ALIGN_TOLERANCE_MM  = 15.0    # closer than this to the start point = done
ALIGN_MAX_GAP_MM    = 250.0   # a gap bigger than this is not believed — nothing is corrected
ALIGN_MAX_PULSES    = 8       # at most this many correction pulses in total
ALIGN_SPEED_MM_S    = 280.0   # first guess of how fast a short pulse moves the car; after the
                              # first pulse the length adapts to how far the car really moved
FINISH_FIX_HEADING  = True    # also turn the car back to its starting direction (False = only
                              # forward/backward, the car is left pointing wherever it ended)
ALIGN_TOLERANCE_DEG = 5.0     # turned less than this against the starting direction = done
ALIGN_TURN_DEG_S    = 150.0   # first guess of how fast a short turn pulse turns the car (adapts too)
ALIGN_MIN_PULSE_S   = 0.06
ALIGN_MAX_PULSE_S   = 0.22
# "Is this the start point?" — the two profiles are laid over each other;
# if they differ by more than this on average, it is another place.
MATCH_MAX_COST_MM   = 60.0
MATCH_SEARCH_DEG    = 30.0    # the car may be turned up to this much against its starting direction
MATCH_CAP_MM        = 150.0   # one 3-degree slice can't count for more than this
MATCH_MIN_OVERLAP   = 0.40    # share of the slices that must be seen in both profiles


# ══════════════════════════════════════════════════════════════
#  Comparing two LIDAR distance profiles (pure maths, no hardware)
# ══════════════════════════════════════════════════════════════

def _rotation_cost(live: list, ref: list, shift_deg: float):
    """Average difference (mm) between the live profile and the reference
    profile looked at `shift_deg` further clockwise; None if they overlap
    too little."""
    n     = len(ref)
    shift = shift_deg * n / 360.0
    base  = math.floor(shift)
    frac  = shift - base
    total, count = 0.0, 0
    for i, lv in enumerate(live):
        if lv is None:
            continue
        a = ref[(i + base) % n]
        b = ref[(i + base + 1) % n]
        if a is None or b is None:
            continue
        diff = abs(lv - (a + (b - a) * frac))
        total += diff if diff < MATCH_CAP_MM else MATCH_CAP_MM
        count += 1
    if count < MATCH_MIN_OVERLAP * n:
        return None
    return total / count


def match_heading(live: list, ref: list, search_deg: float = MATCH_SEARCH_DEG, step_deg: float = 1.0):
    """
    How many degrees clockwise the car is turned now compared with how it
    stood when `ref` was taken: tries every heading within +/- search_deg
    and keeps the one where the two profiles fit best.
    Returns (heading_deg, fit_mm) or None.
    """
    costs = []
    steps = int(round(2 * search_deg / step_deg))
    for k in range(steps + 1):
        deg  = -search_deg + k * step_deg
        cost = _rotation_cost(live, ref, deg)
        if cost is not None:
            costs.append((deg, cost))
    if len(costs) < 3:
        return None
    best = min(range(len(costs)), key=lambda k: costs[k][1])
    deg, cost = costs[best]
    if 0 < best < len(costs) - 1:           # fine position: parabola through 3 points
        c0, c1, c2 = costs[best - 1][1], cost, costs[best + 1][1]
        den = c0 - 2 * c1 + c2
        if den > 1e-9:
            deg += 0.5 * (c0 - c2) / den * step_deg
    return deg, cost


def match_position(live: list, ref: list, heading_deg: float):
    """
    Where the car stands now relative to where it stood when `ref` was
    taken, once the heading difference is known: (right_mm, forward_mm) in
    the directions the car had at `ref`. The live wall points are laid over
    the reference walls and shifted until they coincide; a point only has
    to land ON the reference wall line, so a direction the walls say
    nothing about (sliding along a plain corridor) comes out as 0.
    Returns (right_mm, forward_mm) or None.
    """
    n    = len(ref)
    binw = 2.0 * math.pi / n
    ref_pts = [None if r is None else (r * math.sin(i * binw), r * math.cos(i * binw))
               for i, r in enumerate(ref)]
    normals = [None] * n
    for i in range(n):
        p0, p1, p2 = ref_pts[i - 1], ref_pts[i], ref_pts[(i + 1) % n]
        if p0 is None or p1 is None or p2 is None:
            continue
        if abs(ref[(i + 1) % n] - ref[i - 1]) > 0.35 * ref[i] + 30.0:
            continue                               # a jump in distance = an edge, not one wall
        tx, ty = p2[0] - p0[0], p2[1] - p0[1]
        length = math.hypot(tx, ty)
        if length < 1e-6:
            continue
        normals[i] = (ty / length, -tx / length)

    psi = math.radians(heading_deg)
    live_pts = [(r * math.sin(j * binw + psi), r * math.cos(j * binw + psi))
                for j, r in enumerate(live) if r is not None]
    if len(live_pts) < 0.3 * n:
        return None

    tx_e = ty_e = 0.0
    for _ in range(5):
        sxx = sxy = syy = bx = by = 0.0
        used = 0
        for qx, qy in live_pts:
            wx, wy = qx + tx_e, qy + ty_e
            i = int(round((math.atan2(wx, wy) % (2.0 * math.pi)) / binw)) % n
            nrm = normals[i]
            if nrm is None:
                continue
            px, py = ref_pts[i]
            resid = nrm[0] * (px - qx) + nrm[1] * (py - qy)
            if abs(resid - (nrm[0] * tx_e + nrm[1] * ty_e)) > 100.0:
                continue                           # not the same wall
            sxx += nrm[0] * nrm[0]; sxy += nrm[0] * nrm[1]; syy += nrm[1] * nrm[1]
            bx  += nrm[0] * resid;  by  += nrm[1] * resid
            used += 1
        if used < 0.25 * n:
            return None
        lam = 0.02 * used
        sxx += lam; syy += lam
        det = sxx * syy - sxy * sxy
        if det < 1e-9:
            return None
        tx_e = (syy * bx - sxy * by) / det
        ty_e = (sxx * by - sxy * bx) / det
    return tx_e, ty_e


class AutoDrive:
    """Replays the latest (or a specific) recorded route from the database by
    time, then (closed routes) uses the LIDAR to finish on the start point."""

    def __init__(self, motors: MotorDriverL298n, route_db: RouteDatabase, lidar=None):
        """
        :param motors:   initialised MotorDriverL298n instance
        :param route_db: initialised RouteDatabase instance
        :param lidar:    initialised LidarDriver instance (needs get_profile()) —
                         optional; without it there is no finish check.
        """
        self._motors   = motors
        self._route_db = route_db
        self._lidar    = lidar

        self._thread  = None
        self._running = False

        self._on_finish = None  # optional callback(), called when replay ends
        self._event_sink = None # optional callback(text): navigation steps, for the scan log
        self._ml = MLDriver(motors, lidar) if lidar is not None else None
        self.last_problem = ""  # why start() returned False: 'no_model' / 'no_route'

        print("[AutoDrive] Initialised" + (" (timed replay + LIDAR finish check)" if lidar else " (timing only — no LIDAR)"))

    def set_on_finish(self, callback) -> None:
        """
        Optional callback (no args), called when playback ends — either because
        the recorded data ran out, or because stop() was called. Lets the UI
        know driving is done without having to poll is_running constantly.
        """
        self._on_finish = callback

    def set_event_sink(self, callback) -> None:
        """Optional callback(text) that receives every navigation step as text
        (ui.py passes it on to the scan log, so a drive can be analysed later)."""
        self._event_sink = callback

    @staticmethod
    def route_shape(trajectory: list) -> tuple:
        """
        What the recorded drive looked like, from its commands alone:
        (clockwise, laps). The total turn is estimated from the turn pulses.
        """
        turn, forwards = 0.0, 0
        for step in trajectory:
            a, t = step['action'], step.get('duration') or 0.0
            if a in ('R', 'L'):
                deg = max(0.0, TURN_DEG_PER_S * t - TURN_DEAD_DEG)
                turn += deg if a == 'R' else -deg
            elif a == 'F':
                forwards += 1
        clockwise = turn >= 0
        turn = abs(turn) + FORWARD_DRIFT_DEG * forwards
        return clockwise, max(1, int(round(turn / 360.0)))

    @property
    def is_running(self) -> bool:
        """True while the car drives by itself (neural network or replay)."""
        return self._running or (self._ml is not None and self._ml.is_running)

    @property
    def ml_driver(self):
        return self._ml

    # ── Public API ────────────────────────────────────────────

    def start(self, session_id: int = None) -> bool:
        """
        Start replaying a recorded route in the background.

        :param session_id: specific session to replay, or None for the most
                            recently recorded one (default).
        :return: True if a route was found and playback started,
                 False if there's no recorded data to drive from.
        """
        if self.is_running:
            return True  # already driving — nothing to do
        self.last_problem = ""

        if NAV_MODE == 'ml':
            if self._ml is None:
                self.last_problem = "no_lidar"
                return False
            self._ml.set_event_sink(self._event_sink)
            self._ml.set_on_finish(self._on_finish)
            if not self._ml.start():
                self.last_problem = "no_model"
                return False
            return True

        trajectory = self._route_db.get_waypoint_trajectory(session_id)
        if not trajectory:
            print("[AutoDrive] No recorded route found — nothing to drive.")
            self.last_problem = "no_route"
            return False

        self._running = True
        self._thread = threading.Thread(
            target=self._replay_loop, args=(trajectory,), daemon=True
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        """Cancel playback early — motors are stopped immediately."""
        self._running = False
        if self._ml is not None:
            self._ml.stop()
        self._motors.stop()

    # ── Internal ──────────────────────────────────────────────

    def _apply_action(self, action: str) -> None:
        if action == 'F':
            self._motors.forward()
        elif action == 'B':
            self._motors.backward()
        elif action == 'L':
            self._motors.turn_left()
        elif action == 'R':
            self._motors.turn_right()
        elif action in ('G', 'H', 'I', 'J'):   # recorded diagonals
            self._motors.curve(action)
        else:  # 'S' or anything unrecognised
            self._motors.stop()

    def _run_segment(self, action: str, duration: float) -> str:
        """
        Hold `action` for exactly its recorded `duration`.
        Returns 'time', or 'cancel' if the replay was stopped meanwhile.
        """
        self._apply_action(action)

        start = time.time()
        while self._running:
            elapsed = time.time() - start
            if elapsed >= duration:
                return 'time'
            # never sleep past the moment the segment is due to end
            time.sleep(max(0.0, min(POLL_INTERVAL, duration - elapsed)))
        return 'cancel'

    # ── Finish check ──────────────────────────────────────────

    def _sleep(self, seconds: float) -> bool:
        """Sleep; returns False at once if the replay is cancelled."""
        end = time.time() + seconds
        while self._running:
            left = end - time.time()
            if left <= 0:
                return True
            time.sleep(min(POLL_INTERVAL, left))
        return False

    def _remember_start_point(self):
        """
        Car standing, before the first move: wait until the LIDAR delivers a
        steady picture (its motor has only just been started) and return the
        distance profile of the start point. None if there is no usable
        LIDAR data — the replay then runs without the finish check.
        """
        deadline = time.time() + LIDAR_READY_TIMEOUT
        previous = None
        while self._running and time.time() < deadline:
            profile = self._lidar.get_profile()
            seen = sum(1 for v in profile if v is not None)
            if profile and seen >= 0.5 * len(profile):
                if previous is not None:
                    cost = _rotation_cost(profile, previous, 0.0)
                    if cost is not None and cost < 15.0:      # two pictures in a row agree
                        return profile
                previous = profile
            else:
                previous = None
            if not self._sleep(0.25):
                return None
        return None

    def _where_am_i(self, start_profile: list):
        """
        Car standing: where is it relative to the start point?
        Returns ({'gap', 'side', 'heading', 'fit'}, None) or (None, reason):
          gap     — mm the car must still drive FORWARD (along its own
                    direction) to be level with the start point; negative =
                    it is past it and must go back
          side    — mm it stands to the right (+) / left (-) of the start point
          heading — degrees it is turned clockwise (+) against its starting direction
        """
        live = self._lidar.get_profile()
        if not live:
            return None, "no LIDAR data"
        heading = match_heading(live, start_profile)
        if heading is None or heading[1] > MATCH_MAX_COST_MM:
            return None, "this place does not look like the start point (open route?)"
        pos = match_position(live, start_profile, heading[0])
        if pos is None:
            return None, "this place does not look like the start point (open route?)"
        right, forward = pos
        psi = math.radians(heading[0])
        # distance still to go along the direction the car is pointing NOW
        gap = -(right * math.sin(psi) + forward * math.cos(psi))
        return {'gap': gap, 'side': right, 'heading': heading[0], 'fit': heading[1]}, None

    def _finish_on_start_point(self, start_profile: list) -> None:
        """
        After the last recorded step: bring the car onto the start point with
        up to ALIGN_MAX_PULSES short pulses, measuring after each one —
        first forward/backward until it is level with the start point, then
        (FINISH_FIX_HEADING) left/right until it points the way it started.
        The pulse length adapts to what the previous pulse of the same kind
        really did to the car.
        """
        tag = "[AutoDrive] FINISH check:"
        self._motors.stop()
        # per kind of pulse: [length of the last one, error before it]
        move = {'pulse': None, 'before': None, 'guess': ALIGN_SPEED_MM_S}
        turn = {'pulse': None, 'before': None, 'guess': ALIGN_TURN_DEG_S}
        used = 0
        while self._running:
            if not self._sleep(ALIGN_SETTLE_S):
                return
            pose, reason = self._where_am_i(start_profile)
            if pose is None:
                print(f"{tag} {reason} — nothing corrected")
                return
            gap, heading = pose['gap'], pose['heading']
            where = ("level with the start point" if abs(gap) <= ALIGN_TOLERANCE_MM else
                     f"{gap:.0f}mm short of the start point" if gap > 0 else
                     f"{-gap:.0f}mm past the start point")
            side = f"{abs(pose['side']):.0f}mm to the {'right' if pose['side'] > 0 else 'left'}"
            way  = f"turned {abs(heading):.0f}deg {'right' if heading > 0 else 'left'}"
            print(f"{tag} {where}, {side}, {way}" + (f" (after {used} pulses)" if used else ""))
            if abs(gap) > ALIGN_MAX_GAP_MM:
                print(f"{tag} that is too far to believe — nothing corrected")
                return

            if abs(gap) > ALIGN_TOLERANCE_MM:
                kind, error = move, gap
                action = 'F' if gap > 0 else 'B'
            elif FINISH_FIX_HEADING and abs(heading) > ALIGN_TOLERANCE_DEG:
                kind, error = turn, heading
                action = 'L' if heading > 0 else 'R'     # turned right -> turn back left
            else:
                return                                    # on the start point
            if used >= ALIGN_MAX_PULSES:
                return

            if kind['pulse'] is None:
                pulse = 0.8 * abs(error) / kind['guess']
            else:
                changed = abs(kind['before'] - error)
                if changed < 0.15 * abs(kind['before']):
                    pulse = kind['pulse'] * 1.4           # the car hardly reacted — push longer
                else:
                    pulse = kind['pulse'] * max(0.5, min(1.5, abs(error) / changed))
            pulse = max(ALIGN_MIN_PULSE_S, min(ALIGN_MAX_PULSE_S, pulse))
            kind['pulse'], kind['before'] = pulse, error

            print(f"{tag} pulse '{action}' {pulse:.3f}s")
            self._apply_action(action)
            ok = self._sleep(pulse)
            self._motors.stop()
            if not ok:
                return
            used += 1

    def _replay_loop(self, trajectory: list) -> None:
        print(f"[AutoDrive] Route loaded: {len(trajectory)} recorded steps.")

        # Remember what the start point looks like, for the finish check.
        start_profile = None
        try:
            if NAV_MODE in ('lidar', 'ring') and self._lidar is not None and hasattr(self._lidar, 'get_last_revolution'):
                clockwise, laps = self.route_shape(trajectory)
                if LAPS:
                    laps = int(LAPS)
                print(f"[AutoDrive] LIDAR navigation: {'clockwise' if clockwise else 'counter-clockwise'}, "
                      f"{laps} lap(s) (from the recording)")
                result = RingDriver(self._motors, self._lidar, on_event=self._event_sink).drive(
                    lambda: self._running, clockwise, laps)
                print(f"[AutoDrive] LIDAR navigation (ring track) ended: {result}")
                if result != 'no_ring':
                    return
                # not the plain ring: drive it as "any closed track"
                print("[AutoDrive] Not the plain ring track: using the general track navigation.")
                result = TrackDriver(self._motors, self._lidar, on_event=self._event_sink).drive(
                    lambda: self._running, clockwise, laps)
                print(f"[AutoDrive] LIDAR navigation (any track) ended: {result}")
                if result != 'no_track':
                    return
                print("[AutoDrive] Falling back to the timed replay.")
            print(f"[AutoDrive] Replaying {len(trajectory)} recorded steps (by time)...")
            if FINISH_CHECK and self._lidar is not None:
                if not hasattr(self._lidar, 'get_profile'):
                    print("[AutoDrive] lidar_driver.py has no get_profile() (older file) — "
                          "replaying without the finish check.")
                else:
                    self._motors.stop()
                    start_profile = self._remember_start_point()
                    if not self._running:
                        pass
                    elif start_profile is None:
                        print("[AutoDrive] No steady LIDAR data — replaying without the finish check.")
                    else:
                        print("[AutoDrive] Start point remembered (LIDAR).")
            self._drive(trajectory, start_profile)
        except Exception as e:
            # whatever happens, the car must not be left with its motors on
            print(f"[AutoDrive] ERROR during replay: {e!r} — stopping.")
        finally:
            self._motors.stop()
            self._running = False
            if self._on_finish:
                self._on_finish()

    def _drive(self, trajectory: list, start_profile) -> None:
        """The timed replay itself, then the finish check."""
        for i, step in enumerate(trajectory):
            if not self._running:
                print("[AutoDrive] Replay cancelled.")
                break
            action   = step['action']
            duration = step.get('duration', DEFAULT_STEP_DURATION)
            started  = time.time()
            ended_by = self._run_segment(action, duration)
            # One line per DB row, so the replay output can be checked
            # against the DB: what was recorded vs. how long it was held now.
            print(f"[AutoDrive] step {i + 1}/{len(trajectory)}: action='{action}' "
                  f"recorded={duration:.3f}s held={time.time() - started:.3f}s ({ended_by})")
        else:
            print("[AutoDrive] Reached end of recorded data — stopping.")
            self._motors.stop()
            if start_profile is not None:
                self._finish_on_start_point(start_profile)

    # ── Self Test ─────────────────────────────────────────────

    def self_test(self) -> None:
        """Replay the latest recorded session end-to-end, printing progress."""
        print("\n[AutoDrive] --- Self-test START ---")
        done = threading.Event()
        self.set_on_finish(done.set)

        if not self.start():
            print("[AutoDrive] Nothing to replay — record a route first.")
            return

        done.wait(timeout=300)
        print("[AutoDrive] --- Self-test COMPLETE ---\n")


# ── Run directly ──────────────────────────────────────────────
if __name__ == "__main__":
    import RPi.GPIO as GPIO
    from lidar_driver import LidarDriver

    GPIO.setmode(GPIO.BCM)

    motors = MotorDriverL298n(default_speed=70)
    lidar  = LidarDriver(angle_offset=0.0)
    lidar.start()
    lidar.start_motor()

    db     = RouteDatabase("car_routes.db")
    auto   = AutoDrive(motors, db, lidar)

    try:
        auto.self_test()
    except KeyboardInterrupt:
        print("\n[AutoDrive] Interrupted.")
        auto.stop()
    finally:
        lidar.cleanup()
        motors.cleanup()
        GPIO.cleanup()
