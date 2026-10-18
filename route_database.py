"""
route_database.py
=================
SQLite database car_routes.db — V4: THE TRAINING DATA OF THE NEURAL NETWORK.

Tables:
  route_sessions  one row per recorded drive ("run")
  action_events   every button press of that drive: action (F/B/L/R/S/G/H/I/J),
                  when it started (timestamp) and how long it was held (duration)
                  -> the LABELS of the machine learning ("what the driver did")
  ml_samples      (V4) what the LIDAR saw during that drive, 10 times a second:
                    t        the moment (same clock as action_events.timestamp)
                    s00..s15 the 16 network inputs (distance per direction, 0..1),
                             readable in any SQLite viewer
                    points   the whole LIDAR revolution (compressed), so the
                             training can compute the inputs again with other
                             settings and make the "recovery copies"
                  -> the INPUTS of the machine learning ("what the car saw")
  ml_imports      (V4) which scans/ files were already copied into the database

The neural network is trained ONLY from this database (ml_dataset.py reads
ml_samples + action_events). The files in scans/ stay as a raw backup.
"""

import os
import sqlite3
import time
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

DB_FILE = "car_routes.db"
ML_COLUMNS = 16          # s00..s15 in ml_samples (the 16 network inputs, for reading the table)


class RouteDatabase:
    """SQLite database manager for driving-event logging, inspection, and route replay."""

    def __init__(self, db_path: str = DB_FILE):
        # A relative name is resolved next to this file, NOT against the
        # current working directory — so the program always reads/writes the
        # same car_routes.db no matter which folder it was started from.
        if not os.path.isabs(db_path):
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), db_path)
        self.db_path = db_path
        print(f"[RouteDatabase] Using {self.db_path}")
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection."""
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """Create route_sessions and action_events tables if missing; add the
        zone_0..zone_7 columns to an already-existing action_events table
        from before (ALTER TABLE) so older databases upgrade in place."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS route_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Event-based recording: one row per button press, holding the
            # exact duration it was held, plus the 8-zone LIDAR snapshot
            # taken the moment the action started.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS action_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    action TEXT NOT NULL,
                    duration REAL NOT NULL,
                    zone_0 REAL,
                    zone_1 REAL,
                    zone_2 REAL,
                    zone_3 REAL,
                    zone_4 REAL,
                    zone_5 REAL,
                    zone_6 REAL,
                    zone_7 REAL,
                    FOREIGN KEY (session_id) REFERENCES route_sessions (id) ON DELETE CASCADE
                )
            """)

            # Upgrade path: a DB created before the zone_N columns existed
            # has action_events without them — add them in place instead of
            # losing the table (CREATE TABLE IF NOT EXISTS won't add columns
            # to an existing table).
            existing_cols = {row[1] for row in cursor.execute("PRAGMA table_info(action_events)")}
            for i in range(8):
                col = f"zone_{i}"
                if col not in existing_cols:
                    cursor.execute(f"ALTER TABLE action_events ADD COLUMN {col} REAL")

            # V4 — machine-learning data: what the LIDAR saw, 10 times a second
            cols = ", ".join(f"s{i:02d} REAL" for i in range(ML_COLUMNS))
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS ml_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    t REAL NOT NULL,
                    {cols},
                    points BLOB NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES route_sessions (id) ON DELETE CASCADE
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS ml_samples_session ON ml_samples (session_id, t)")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_imports (
                    file TEXT PRIMARY KEY,
                    session_id INTEGER,
                    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

    def start_session(self, session_name: Optional[str] = None) -> int:
        """Start a new route recording session and return its ID."""
        if not session_name:
            session_name = f"Route_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO route_sessions (name) VALUES (?)", (session_name,))
            conn.commit()
            session_id = cursor.lastrowid
            print(f"[RouteDatabase] Started new session '{session_name}' (ID: {session_id})")
            return session_id

    def log_action_event(self, session_id: int, action: str, duration: float,
                          zones: Optional[List[float]] = None) -> None:
        """
        Log one driving segment — action (F/B/L/R/S) held for exactly
        `duration` seconds, as it actually happened, plus the 8-zone LIDAR
        snapshot (`zones`) read at the moment the action started. One row
        per button press/release pair, not one row per fixed time tick —
        this is what RECORDING ROUTE uses (see ui.py); replay itself only
        needs "do X for Y seconds", the LIDAR snapshot is stored alongside
        for inspection/analysis.
        """
        zone_values = (list(zones) + [None] * 8)[:8] if zones is not None else [None] * 8
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO action_events (
                    session_id, timestamp, action, duration,
                    zone_0, zone_1, zone_2, zone_3, zone_4, zone_5, zone_6, zone_7
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, time.time(), action, duration, *zone_values))
            conn.commit()

    def log_action_events(self, session_id: int, events: List[Dict[str, Any]]) -> int:
        """
        Save a whole recording in ONE transaction. Each event is a dict:
            {'action': 'F', 'start': <time.time() when it began>,
             'duration': <seconds>, 'zones': [8 values or None]}
        `timestamp` in the table is the real moment the action started.

        This is what RECORDING ROUTE uses: the route is timed in memory while
        driving and written here once, when recording stops. Writing row by
        row during the drive (a commit to the SD card per button press, tens
        of ms each) blocked the Bluetooth thread and that time went missing
        from the recorded durations. Returns the number of rows written.
        """
        rows = []
        for ev in events:
            zones = ev.get('zones')
            zone_values = (list(zones) + [None] * 8)[:8] if zones is not None else [None] * 8
            rows.append((session_id, ev['start'], ev['action'], round(ev['duration'], 3), *zone_values))
        with self._get_connection() as conn:
            conn.executemany("""
                INSERT INTO action_events (
                    session_id, timestamp, action, duration,
                    zone_0, zone_1, zone_2, zone_3, zone_4, zone_5, zone_6, zone_7
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
        return len(rows)

    def get_action_events(self, session_id: int) -> List[Tuple]:
        """Retrieve all (id, timestamp, action, duration, zone_0..zone_7) event rows, in order."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, action, duration,
                       zone_0, zone_1, zone_2, zone_3, zone_4, zone_5, zone_6, zone_7
                FROM action_events
                WHERE session_id = ?
                ORDER BY id ASC
            """, (session_id,))
            return cursor.fetchall()

    def list_sessions(self) -> List[Tuple[int, str, str, int]]:
        """List all saved sessions with their recorded action-event counts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.id, s.name, s.started_at, COUNT(r.id) as sample_count
                FROM route_sessions s
                LEFT JOIN action_events r ON s.id = r.session_id
                GROUP BY s.id
                ORDER BY s.id DESC
            """)
            return cursor.fetchall()

    def get_waypoint_trajectory(self, session_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves the recorded driving sequence for replay — one entry per
        button press, each meaning "hold this action for this many seconds":
        Returns: [{'action': 'F', 'duration': 3.3}, ...]
        If session_id is None, automatically loads the most recent session.
        """
        if session_id is None:
            sessions = self.list_sessions()
            if not sessions:
                print("[RouteDatabase] No sessions available for trajectory replay.")
                return []
            session_id = sessions[0][0]  # Select latest recorded session

        events = self.get_action_events(session_id)
        if not events:
            print(f"[RouteDatabase] Session #{session_id} contains no recorded actions.")
            return []

        trajectory = [
            {'action': action, 'duration': duration, 'zones': list(zones)}
            for (_event_id, _timestamp, action, duration, *zones) in events
        ]

        print(f"[RouteDatabase] Loaded {len(trajectory)} recorded actions from Session #{session_id}")
        return trajectory

    # ── V4: machine-learning data ─────────────────────────────

    def add_ml_samples(self, session_id: int, rows) -> int:
        """rows: [(t, [16 inputs], points_blob), ...] — one recorded drive's LIDAR pictures."""
        cols = ", ".join(f"s{i:02d}" for i in range(ML_COLUMNS))
        marks = ", ".join("?" for _ in range(ML_COLUMNS))
        data = [(session_id, t, *((list(f) + [None] * ML_COLUMNS)[:ML_COLUMNS]), blob)
                for t, f, blob in rows]
        with self._get_connection() as conn:
            conn.executemany(f"INSERT INTO ml_samples (session_id, t, {cols}, points) "
                             f"VALUES (?, ?, {marks}, ?)", data)
            conn.commit()
        return len(data)

    def ml_sessions(self):
        """[(session_id, name, number of LIDAR pictures), ...] oldest first — the training runs."""
        with self._get_connection() as conn:
            return conn.execute("""
                SELECT s.id, s.name, COUNT(m.id) FROM route_sessions s
                JOIN ml_samples m ON m.session_id = s.id
                GROUP BY s.id ORDER BY s.id ASC
            """).fetchall()

    def get_ml_pictures(self, session_id: int):
        """[(t, points_blob), ...] of one drive, in time order."""
        with self._get_connection() as conn:
            return conn.execute("SELECT t, points FROM ml_samples WHERE session_id = ? ORDER BY t",
                                (session_id,)).fetchall()

    def get_segments(self, session_id: int):
        """The button presses of one drive as [(start, end, action), ...] (from action_events)."""
        return [(ts, ts + (dur or 0.0), action)
                for (_id, ts, action, dur, *_z) in self.get_action_events(session_id)]

    def is_imported(self, file_name: str) -> bool:
        with self._get_connection() as conn:
            return conn.execute("SELECT 1 FROM ml_imports WHERE file = ?", (file_name,)).fetchone() is not None

    def mark_imported(self, file_name: str, session_id) -> None:
        with self._get_connection() as conn:
            conn.execute("INSERT OR REPLACE INTO ml_imports (file, session_id) VALUES (?, ?)",
                         (file_name, session_id))
            conn.commit()

    def delete_session(self, session_id: int) -> None:
        """Delete ONE recorded drive (its presses and its LIDAR pictures) — e.g. a bad one."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM ml_samples WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM action_events WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM route_sessions WHERE id = ?", (session_id,))
            conn.commit()
        print(f"[RouteDatabase] Session #{session_id} deleted.")

    def clear_all_data(self) -> None:
        """Delete every recorded session, event and LIDAR picture — the UI's CLEAR button.
        (ml_imports is kept, so the scans/ backup files are NOT imported again by themselves.)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM ml_samples")
            cursor.execute("DELETE FROM action_events")
            cursor.execute("DELETE FROM route_sessions")
            try:
                cursor.execute(
                    "DELETE FROM sqlite_sequence WHERE name IN "
                    "('action_events', 'route_sessions', 'ml_samples')"
                )
            except sqlite3.OperationalError:
                pass  # sqlite_sequence doesn't exist yet if nothing was ever inserted
            conn.commit()
        print("[RouteDatabase] All sessions & events deleted.")

    def inspect_database(self) -> None:
        """Print stored sessions and action-event entries to console."""
        sessions = self.list_sessions()
        print("\n" + "=" * 60)
        print("                RECORDED ROUTE SESSIONS                ")
        print("=" * 60)
        if not sessions:
            print("No recorded sessions found.")
            return

        for sess_id, name, created, count in sessions:
            print(f"\nSession ID [{sess_id}] | Name: {name} | Created: {created} | Events: {count}")
            events = self.get_action_events(sess_id)
            if events:
                print("  Event Preview (First 3 entries):")
                for event in events[:3]:
                    e_id, ts, action, duration, *zones = event
                    zone_str = ", ".join(f"{z:.2f}m" if z is not None else "  --  " for z in zones)
                    print(f"    - Event #{e_id} | Action: {action:5s} | Duration: {duration:.2f}s | "
                          f"LIDAR @start: [{zone_str}]")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    import sys
    db = RouteDatabase()
    if len(sys.argv) >= 3 and sys.argv[1] == "delete":
        db.delete_session(int(sys.argv[2]))          # python3 route_database.py delete 7
    else:
        db.inspect_database()
        runs = db.ml_sessions()
        print(f"Machine-learning runs in the database: {len(runs)}")
        for sid, name, n in runs:
            print(f"  session #{sid:3d}  {name:32s} {n:5d} LIDAR pictures")
        print("To delete one bad run:  python3 route_database.py delete <session number>")