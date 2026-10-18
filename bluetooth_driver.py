"""
bluetooth_driver.py
===================
Bluetooth RFCOMM driver for car control.
Receives commands from "Arduino Bluetooth Control" app
and translates them to motor movements.

Commands from app:
  F = forward        B = backward
  R = turn right     L = turn left
  G = forward-left   H = forward-right
  I = backward-left  J = backward-right
  S = stop
  V / Y = horn pressed   W = lights pressed   (v/w = released, ignored)

Speed is fixed: set once via default_speed in main.py
(MotorDriverL298n(default_speed=...)). Digits 0-9 from the app are ignored.

Wiring:
  Raspberry Pi 4 onboard Bluetooth
"""

import bluetooth
import threading
import time
from motor_driver_L298n import MotorDriverL298n


class BluetoothDriver:
    """
    Bluetooth RFCOMM server.
    Listens for incoming connections and translates
    app commands to motor movements.
    """

    def __init__(self, motors: MotorDriverL298n):
        """
        :param motors: initialised MotorDriverL298n instance
        """
        self._motors        = motors
        self._running       = False
        self._client        = None
        self._on_connect    = None  # callback when connected
        self._on_disconnect = None  # callback when disconnected
        self._on_horn       = None  # callback when 'HORN' pressed on app (cmd 'V')
        self._on_lights     = None  # callback when 'LIGHTS' pressed on app (cmd 'W')

        # Last driving command (F/B/L/R/S) + when it was pressed — read by
        # ui.py so it can log "what the driver did".
        self._current_action   = 'S'
        self._action_time      = time.time()
        self._on_action_change = None  # callback(new_action) on every F/B/L/R/S transition

        print("[BluetoothDriver] Initialised")

    def set_on_connect(self, callback) -> None:
        """Set callback function called when phone connects."""
        self._on_connect = callback

    def set_on_disconnect(self, callback) -> None:
        """Set callback function called when phone disconnects."""
        self._on_disconnect = callback

    def set_on_horn(self, callback) -> None:
        """Set callback function called when 'HORN' is pressed on the app."""
        self._on_horn = callback

    def set_on_lights(self, callback) -> None:
        """Set callback function called when 'LIGHTS' is pressed on the app."""
        self._on_lights = callback

    @property
    def is_running(self) -> bool:
        """True while the RFCOMM server (and any active connection) is up."""
        return self._running

    def get_action_state(self):
        """
        Return (action, action_time) — the last F/B/L/R/S command received
        and the timestamp (time.time()) it was pressed.
        """
        return self._current_action, self._action_time

    def set_on_action_change(self, callback) -> None:
        """
        Set callback(new_action, timestamp) — called the instant a NEW
        driving command (F/B/L/R/S or the diagonals G/H/I/J) arrives, i.e. a
        real change, not a repeat. `timestamp` is the time.time() at which
        the command was received (taken BEFORE the motors are switched), so
        the same instant ends the previous segment and starts the new one.
        Used by the RECORDING ROUTE flow in ui.py to time exactly how long
        each action was held, event by event, instead of sampling on a timer.
        Pass None to unregister.
        """
        self._on_action_change = callback

    def _set_action(self, action: str, now: float = None) -> None:
        # Only a REAL transition should notify on_action_change. Some apps
        # (incl. ours) keep sending the same command byte repeatedly while a
        # button is held, not just once on press — without this check, every
        # repeat would be treated as a new segment by the RECORDING ROUTE
        # flow in ui.py, chopping one long hold into many tiny ones (some
        # short enough to get filtered out entirely), which is why recorded
        # replays were coming out shorter than the real drive.
        if now is None:
            now = time.time()
        changed = (action != self._current_action)
        self._current_action = action
        self._action_time    = now
        if changed and self._on_action_change:
            self._on_action_change(action, now)

    # ── Command handler ───────────────────────────────────────

    def _handle_command(self, cmd: str) -> None:
        """Translate a single command character to motor action."""
        cmd = cmd.strip()
        if not cmd:
            return

        # One timestamp per command, taken before anything else happens —
        # this is the moment the motors change, so it is what the recording
        # must use (not "whenever the callback got around to running").
        now = time.time()

        print(f"[BluetoothDriver] CMD: '{cmd}'")

        if cmd == 'F':
            self._motors.forward()
            self._set_action('F', now)
        elif cmd == 'B':
            self._motors.backward()
            self._set_action('B', now)
        elif cmd == 'R':
            self._motors.turn_right()
            self._set_action('R', now)
        elif cmd == 'L':
            self._motors.turn_left()
            self._set_action('L', now)
        elif cmd in ('G', 'H', 'I', 'J'):
            # Diagonals (G fwd-left, H fwd-right, I back-left, J back-right).
            # These change what the motors do, so they are driving actions
            # like any other and MUST be reported — otherwise the recording
            # stores a curve as plain "F"/"R" and the replay drives straight.
            self._motors.curve(cmd)
            self._set_action(cmd, now)
        elif cmd == 'S':
            self._motors.stop()
            self._set_action('S', now)
        elif cmd == 'V':
            # HORN pressed (standard protocol)
            if self._on_horn:
                self._on_horn()
        elif cmd == 'v':
            pass  # HORN released — no action needed
        elif cmd == 'Y':
            # HORN pressed — this is what our actual app sends for HORN
            if self._on_horn:
                self._on_horn()
        elif cmd == 'W':
            # LIGHTS pressed
            if self._on_lights:
                self._on_lights()
        elif cmd == 'w':
            pass  # LIGHTS released — no action needed
        else:
            print(f"[BluetoothDriver] Unknown command: '{cmd}'")

    # ── Server ────────────────────────────────────────────────

    def start(self) -> None:
        """Start Bluetooth RFCOMM server and wait for connection."""
        server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        server_sock.bind(('', 1))          # ערוץ קבוע 1 — האפליקציה מצפה לזה
        server_sock.listen(1)
        port = server_sock.getsockname()[1]

        # רישום שירות SDP — Serial Port Profile
        bluetooth.advertise_service(
            server_sock,
            "CarML-BT",
            service_id="00001101-0000-1000-8000-00805F9B34FB",
            service_classes=[bluetooth.SERIAL_PORT_CLASS],
            profiles=[bluetooth.SERIAL_PORT_PROFILE],
        )

        print(f"[BluetoothDriver] Waiting for connection on port {port}...")

        self._running = True

        while self._running:
            try:
                client_sock, address = server_sock.accept()
                self._client = client_sock
                print(f"[BluetoothDriver] Connected: {address}")

                # קריאה ל-callback כשמתחבר
                if self._on_connect:
                    self._on_connect()

                # Handle commands in loop
                while self._running:
                    try:
                        data = client_sock.recv(1024)
                        if not data:
                            break
                        for byte in data.decode('utf-8', errors='ignore'):
                            self._handle_command(byte)
                    except bluetooth.BluetoothError:
                        break

                print("[BluetoothDriver] Disconnected.")

                # קריאה ל-callback כשמתנתק
                if self._on_disconnect:
                    self._on_disconnect()

                self._motors.stop()
                client_sock.close()

            except bluetooth.BluetoothError as e:
                if self._running:
                    print(f"[BluetoothDriver] Error: {e}")

        server_sock.close()
        print("[BluetoothDriver] Server stopped.")

    def start_async(self) -> None:
        """Start Bluetooth server in a background thread."""
        t = threading.Thread(target=self.start, daemon=True)
        t.start()
        print("[BluetoothDriver] Server running in background thread.")

    def stop(self) -> None:
        """Stop the Bluetooth server."""
        self._running = False
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
        print("[BluetoothDriver] Stopped.")

    def cleanup(self) -> None:
        """Stop server and motors."""
        self.stop()
        print("[BluetoothDriver] Cleaned up.")

    # ── Self Test ─────────────────────────────────────────────

    def self_test(self) -> None:
        """Start server and wait for app connection."""
        print("\n[BluetoothDriver] --- Self-test START ---")
        print("[BluetoothDriver] פתח את האפליקציה, התחבר ל-raspberrypi")
        print("[BluetoothDriver] לחץ על כפתורי התנועה — המנועים יזוזו!")
        print("[BluetoothDriver] Ctrl+C לעצירה")
        self.start()


# ── Run directly ──────────────────────────────────────────────
if __name__ == "__main__":
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)

    motors = MotorDriverL298n(default_speed=70)
    bt     = BluetoothDriver(motors)

    try:
        bt.self_test()
    except KeyboardInterrupt:
        print("\n[BluetoothDriver] Interrupted.")
    finally:
        motors.cleanup()
        GPIO.cleanup()
