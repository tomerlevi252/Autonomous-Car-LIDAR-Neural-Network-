"""
ml_driver.py   (V3 — 100% neural network)
==========================================
STEP 4 of the PDF — "real-time inference": the TRAINED NEURAL NETWORK drives
the car by itself, from what the LIDAR sees right now.

        LIDAR (one revolution, ~6 per second)
            |
        ml_features.py   -> 16 numbers 0..1 (same as in training)
            |
        NEURAL NETWORK (ml_model.npz)  -> joy_x (steering), joy_y (throttle)
            |                             or F / L / R  (classification model)
        motors

V3: EVERY driving decision is the network's. There is no safety layer, no
"stuck" rule and no other hand-written rule that looks at the LIDAR: the
LIDAR picture goes ONLY into the network, and only the network's answer moves
the motors. (The network learned by itself to keep away from walls — see the
"recovery augmentation" in ml_dataset.py.)
The only things that are not the network: the STOP button on the screen,
MAX_RUN_S, and stopping when the LIDAR delivers no data at all (with no
input there is nothing for the network to decide on).

Two ways of driving (MODE):
  'steps'  (default) — drives the way YOU drove when you recorded: look,
           decide, one short tap (as long as your typical tap: ~0.12 s),
           stop, look again. The network's steering joy_x is turned into taps
           the way a driver would: joy_x = +0.33 (turn right a third of the time)
           -> about every third tap is a right turn (like a PWM signal: it only
           uses the network's numbers, never the LIDAR).
  'cruise' — drives without stopping: joy_x / joy_y go straight to the two
           motor sides like a tank joystick (left = y + x, right = y - x).
"""

import collections
import os
import threading
import time

import ml_features as mf
from ml_model import load_model

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "ml_model.npz")

# ── How it drives ─────────────────────────────────────────────
MODE             = 'steps'   # 'steps' or 'cruise'
START_DELAY_S    = 1.5       # after RUN is pressed: time to take your hand away
SETTLE_S         = 0.20      # after a tap: stand this long so the next LIDAR picture is sharp
PULSE_SCALE      = 1.0       # 1.0 = taps as long as yours were in the recordings
DEFAULT_PULSE_S  = {'F': 0.12, 'L': 0.12, 'R': 0.12, 'B': 0.12}   # if the model has none
MAX_RUN_S        = None      # None = drive until STOP; or seconds, e.g. 120
SAMPLE_PERIOD_S  = 0.10      # history for the LSTM: one picture every 0.1 s (as in training)
STEER_GAIN       = 1.5       # joy_x is multiplied by this before it becomes taps (V3: 1.5 — tested in the
                             # simulator: 1.0 reacts too late at an inward bulge of the wall; 1.5 and 2.0
                             # drove every track without touching a wall). Uses only the network's number.
TURN_AT          = 0.75      # 'steps': a turn tap once this much turning has added up (see the top)

# CRUISE
CRUISE_SPEED     = 70        # % — speed with the joystick fully forward
CRUISE_MIN_PWM   = 30        # % — a side below this does not turn at all; it is then set to 0
CRUISE_PERIOD_S  = 0.15      # a new decision about this often (one LIDAR revolution)

# ── LIDAR data (not a driving rule: without input the network cannot decide) ──
MIN_POINTS       = 200       # fewer points in a revolution = the LIDAR is not delivering
NO_DATA_ABORT_S  = 3.0       # no LIDAR data for this long -> stop driving


class MLDriver:
    """Drives the car with the trained neural network. Same interface as AutoDrive."""

    def __init__(self, motors, lidar, model_path: str = MODEL_PATH, on_event=None, log=print):
        self._motors     = motors
        self._lidar      = lidar
        self._model_path = model_path
        self._on_event   = on_event
        self._log        = log
        self._model      = None
        self._thread     = None
        self._running    = False
        self._on_finish  = None
        self._history    = collections.deque(maxlen=64)   # (time, features) for the LSTM
        self._last_sample = 0.0
        self.last_decision = ""
        self.steps = 0

    # ── public ────────────────────────────────────────────────

    def set_on_finish(self, callback) -> None:
        self._on_finish = callback

    def set_event_sink(self, callback) -> None:
        self._on_event = callback

    @property
    def is_running(self) -> bool:
        return self._running

    def is_model_available(self) -> bool:
        return os.path.isfile(self._model_path)

    def load(self) -> bool:
        """(Re)load ml_model.npz — call after training. False if missing / broken."""
        if not self.is_model_available():
            self._log(f"[MLDriver] No trained model ({self._model_path}) - press TRAIN first.")
            return False
        try:
            self._model = load_model(self._model_path)
        except Exception as e:
            self._log(f"[MLDriver] Could not load the model: {e!r}")
            return False
        meta = self._model.meta
        self._log(f"[MLDriver] Model loaded: {self._model.kind.upper()} "
                  f"{getattr(self._model, 'task', 'joystick')}, {self._model.n_inputs} inputs, "
                  f"outputs {self._model.classes}, {self._model.n_params()} parameters, "
                  f"trained {meta.get('trained', '?')} on {meta.get('runs', '?')} runs "
                  f"(test accuracy {meta.get('test_accuracy', 0) * 100:.0f}%)")
        return True

    def start(self) -> bool:
        if self._running:
            return True
        if not self.load():
            return False
        if self._lidar is None or not hasattr(self._lidar, 'get_last_revolution'):
            self._log("[MLDriver] No LIDAR driver - cannot drive.")
            return False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        self._motors.stop()

    # ── helpers ───────────────────────────────────────────────

    def _say(self, text: str) -> None:
        self.last_decision = text
        self._log(f"[MLDriver] {text}")
        if self._on_event:
            try:
                self._on_event(text)
            except Exception:
                pass

    def _features(self, points):
        m = self._model.meta
        return mf.features_from_points(points, int(m.get('n_sectors', self._model.n_inputs)),
                                       float(m.get('max_mm', mf.MAX_MM)),
                                       m.get('percentile'), m.get('min_valid_mm'))

    def _sample_history(self) -> None:
        """Keep a picture every SAMPLE_PERIOD_S (for the LSTM's 2 s of history)."""
        now = time.time()
        if now - self._last_sample < SAMPLE_PERIOD_S:
            return
        pts = self._lidar.get_last_revolution()
        if len(pts) >= MIN_POINTS:
            self._history.append((now, self._features(pts)))
            self._last_sample = now

    def _sleep(self, seconds: float) -> bool:
        end = time.time() + seconds
        while self._running:
            left = end - time.time()
            if left <= 0:
                return True
            self._sample_history()
            time.sleep(min(0.02, left))
        return False

    def _ask_network(self, vec):
        """-> (joy_x, joy_y, action_from_classifier_or_None, text)"""
        m = self._model
        if m.kind == 'lstm':
            seq = [f for _, f in list(self._history)[-m.window:]]
            while len(seq) < m.window:
                seq.insert(0, seq[0] if seq else vec)
            _, _, out = m.predict(seq)
        else:
            _, _, out = m.predict(vec)
        if 'joy_x' in out:
            return out['joy_x'], out['joy_y'], None, f"joy x{out['joy_x']:+.2f} y{out['joy_y']:+.2f}"
        best = max(out, key=out.get)
        probs = " ".join(f"{k}{v:.2f}" for k, v in out.items())
        jx = out.get('R', 0.0) - out.get('L', 0.0)
        jy = out.get('F', 0.0) - out.get('B', 0.0)
        return jx, jy, best, probs

    def _apply(self, action: str) -> None:
        if action == 'F':
            self._motors.forward()
        elif action == 'B':
            self._motors.backward()
        elif action == 'L':
            self._motors.turn_left()
        elif action == 'R':
            self._motors.turn_right()
        else:
            self._motors.stop()

    # ── main loop ─────────────────────────────────────────────

    def _get_points(self, started_no_data):
        """A usable revolution, or None (and whether it is time to give up)."""
        pts = self._lidar.get_last_revolution()
        if len(pts) >= MIN_POINTS:
            return pts, False
        return None, time.time() - started_no_data > NO_DATA_ABORT_S

    def _loop(self) -> None:
        reason = "stopped"
        try:
            self._motors.stop()
            self._say(f"ML start: {MODE} mode, waiting {START_DELAY_S:.1f}s")
            if not self._sleep(START_DELAY_S):
                return
            t_start = time.time()
            if MODE == 'cruise':
                reason = self._cruise(t_start)
            else:
                reason = self._steps(t_start)
        except Exception as e:
            reason = f"ERROR {e!r}"
        finally:
            self._motors.stop()
            self._running = False
            self._say(f"ML FINISHED ({reason}) after {self.steps} decisions")
            if self._on_finish:
                self._on_finish()

    def _steps(self, t_start) -> str:
        pulse_s = dict(DEFAULT_PULSE_S)
        pulse_s.update(self._model.meta.get('pulse_s', {}))
        budget = 0.0           # turning "owed" by the network (see the top of this file)
        no_data_since = time.time()
        while self._running:
            if MAX_RUN_S and time.time() - t_start > MAX_RUN_S:
                return "time limit"
            pts, give_up = self._get_points(no_data_since)
            if pts is None:
                self._motors.stop()
                if give_up:
                    return "no LIDAR data"
                if not self._sleep(0.05):
                    return "stopped"
                continue
            no_data_since = time.time()
            self._sample_history()
            vec = self._features(pts)
            jx, jy, cls_action, text = self._ask_network(vec)

            # network output -> one tap
            if cls_action is not None:                      # classification model
                action = cls_action
            elif jy < -0.3:
                action = 'B'
            else:                                           # joystick model: spread the turns over the taps
                jx = max(-1.0, min(1.0, jx * STEER_GAIN))
                if budget * jx < 0:
                    budget = 0.0                            # the network changed its mind: start again
                budget += jx
                if budget >= TURN_AT:
                    action, budget = 'R', budget - 1.0
                elif budget <= -TURN_AT:
                    action, budget = 'L', budget + 1.0
                else:
                    action = 'F'
            pulse = pulse_s.get(action, 0.12) * PULSE_SCALE
            self.steps += 1
            self._say(f"ML #{self.steps} {text} -> net {action} | do {action} {pulse:.2f}s")
            if action != 'S' and pulse > 0:
                self._apply(action)
                ok = self._sleep(pulse)
                self._motors.stop()
                if not ok:
                    return "stopped"
            t_stop = time.time()
            if not self._sleep(SETTLE_S):
                return "stopped"
            if not self._wait_fresh(t_stop):
                return "stopped"
        return "stopped"

    def _wait_fresh(self, t_stop: float, limit_s: float = 0.4) -> bool:
        """Wait until the last LIDAR revolution was measured completely after the car
        stopped (every point younger than the stop), so the next decision is made on a
        sharp picture of where the car really stands. False if cancelled."""
        end = time.time() + limit_s
        while self._running and time.time() < end:
            pts = self._lidar.get_last_revolution()
            if pts and max(p[2] for p in pts) <= time.time() - t_stop:
                return True
            if not self._sleep(0.02):
                return False
        return self._running

    def _cruise(self, t_start) -> str:
        no_data_since = time.time()
        while self._running:
            if MAX_RUN_S and time.time() - t_start > MAX_RUN_S:
                return "time limit"
            pts, give_up = self._get_points(no_data_since)
            if pts is None:
                self._motors.stop()
                if give_up:
                    return "no LIDAR data"
                if not self._sleep(0.05):
                    return "stopped"
                continue
            no_data_since = time.time()
            self._sample_history()
            jx, jy, cls_action, text = self._ask_network(self._features(pts))
            if cls_action is not None:
                jx, jy = {'F': (0, 1), 'L': (-1, 0), 'R': (1, 0), 'B': (0, -1)}.get(cls_action, (0, 0))
            left, right = jy + jx, jy - jx                     # tank mixing
            biggest = max(1.0, abs(left), abs(right))
            left, right = CRUISE_SPEED * left / biggest, CRUISE_SPEED * right / biggest
            left = 0.0 if abs(left) < CRUISE_MIN_PWM else left
            right = 0.0 if abs(right) < CRUISE_MIN_PWM else right
            self._motors.drive(left, right)
            self.steps += 1
            self._say(f"ML #{self.steps} {text} -> motors L{left:+.0f}% R{right:+.0f}%")
            if not self._sleep(CRUISE_PERIOD_S):
                return "stopped"
        return "stopped"


# ── Run directly: drive with the model, without the screen ────
if __name__ == "__main__":
    import RPi.GPIO as GPIO
    from lidar_driver import LidarDriver
    from motor_driver_L298n import MotorDriverL298n

    GPIO.setmode(GPIO.BCM)
    motors = MotorDriverL298n(default_speed=100)
    lidar = LidarDriver()
    lidar.start()
    lidar.start_motor()
    ml = MLDriver(motors, lidar)
    try:
        if ml.start():
            while ml.is_running:
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[MLDriver] Interrupted.")
    finally:
        ml.stop()
        lidar.stop_motor()
        lidar.cleanup()
        motors.cleanup()
