"""
track_nav.py
============
Autonomous driving on ANY closed track without junctions (a ring, an "8"
shaped loop, an oval ...) using only the LIDAR.

How it works (plain Python, no extra libraries):

1. WHERE TO GO — from one LIDAR revolution: of all the points a short way
   ahead of the car, pick the one that is farthest from every wall (= the
   middle of the passage) and drive toward it. Nothing about the shape of
   the track is assumed.

2. WHERE AM I — "LIDAR odometry": every new revolution is laid over an
   earlier one (a "keyframe"); how it has to be shifted/turned to fit tells
   how the car has moved. Adding these up gives the car's position and
   heading relative to the start point.

3. WHEN TO STOP — the very first look (standing on the start point) is kept.
   When the car has come around (it has turned a full circle and its
   estimated position is near the start again) every look is compared with
   that first look; when they fit, the car knows exactly where it is
   relative to the start point. That is how laps are counted and how the
   car stops on the start point.

Driving is done in STEPS: look (standing still) - decide - one short move -
look again. After every move the car measures what the move really did (how
far it went, how much it turned) and uses that for the next ones, so it does
not depend on the battery or the floor. A CRUISE mode (driving without
stopping, like in ring_nav) is in here too, but switched off (CRUISE = False)
until it has been tried on the real car.

It never drives forward when something is too close ahead. It stops by
itself if it cannot find the start point again.
"""

import math
import time

# ── Reading a LIDAR revolution ─────────────────────────────────
MIN_VALID_MM   = 80        # closer than this is the car itself
MAX_VALID_MM   = 4000      # farther than this is not part of the track
MIN_POINTS     = 250       # a usable revolution has at least this many points

# ── Laying one look over another ───────────────────────────────
GRID_CELL_MM   = 20.0
GRID_HALF_MM   = 2600.0    # the grid covers this far to every side of the keyframe
GRID_REACH_MM  = 70.0      # a point this far from every keyframe point counts as "not fitting"
MATCH_POINTS   = 180       # about this many points of the new look are used
MATCH_MIN_SHARE = 0.55     # this share of the points must fit, else the look is not trusted
KEY_MOVE_MM    = 140.0     # take a new keyframe after moving this far ...
KEY_TURN_DEG   = 22.0      # ... or turning this much
KEY_MIN_SHARE  = 0.70      # ... or when the fit gets this weak
MOVE_TRY_MM    = 22.0      # possible amounts of movement are tried this far apart (a quarter of a post)
START_CELL_MM  = 50.0      # the blurred copy of the first look, for finding the start point again
START_REACH_MM = 180.0
START_STEP_MM  = 120.0     # it is searched in steps of this ...
START_SEARCH_MM = 480.0    # ... this far around where the car thinks it is ...
START_SEARCH_DEG = 30.0    # ... and this far around its heading


def _wrap(deg: float) -> float:
    """angle to -180..+180"""
    return (deg + 180.0) % 360.0 - 180.0


def to_xy(points, yaw_deg_s=0.0, speed_mm_s=0.0):
    """
    One LIDAR revolution [(angle_deg, distance_mm, age_s), ...] -> ([x], [y])
    in the car's own coordinates (x = right, y = ahead), as things stand NOW:
    while the car moves, every point is shifted by how the car has moved
    since the point was measured (yaw_deg_s clockwise +, speed forward +).
    """
    xs, ys = [], []
    moving = (yaw_deg_s != 0.0 or speed_mm_s != 0.0)
    for p in points:
        d = p[1]
        if d < MIN_VALID_MM or d > MAX_VALID_MM:
            continue
        if moving:
            age = p[2]
            r = math.radians(p[0] - yaw_deg_s * age)
            xs.append(d * math.sin(r)); ys.append(d * math.cos(r) - speed_mm_s * age)
        else:
            r = math.radians(p[0])
            xs.append(d * math.sin(r)); ys.append(d * math.cos(r))
    return xs, ys


class ScanGrid:
    """One look, drawn into a grid so that other looks can be laid over it quickly."""

    _kernels = {}

    def __init__(self, xs, ys, cell=GRID_CELL_MM, reach=GRID_REACH_MM):
        self.cell, self.n = cell, int(2 * GRID_HALF_MM / cell)
        n = self.n
        kernel = ScanGrid._kernels.get((cell, reach))
        if kernel is None:
            r = int(reach // cell)
            kernel = [(dx, dy, int(255 * (1.0 - math.hypot(dx, dy) * cell / reach)))
                      for dx in range(-r, r + 1) for dy in range(-r, r + 1) if math.hypot(dx, dy) * cell < reach]
            ScanGrid._kernels[(cell, reach)] = kernel
        g = bytearray(n * n)
        inv, half = 1.0 / cell, GRID_HALF_MM
        lo, hi = 4, n - 5
        for x, y in zip(xs, ys):
            ix = int((x + half) * inv); iy = int((y + half) * inv)
            if lo <= ix <= hi and lo <= iy <= hi:
                base = iy * n + ix
                for dx, dy, v in kernel:
                    k = base + dy * n + dx
                    if g[k] < v:
                        g[k] = v
        self.g = g

    def score(self, xs, ys, th_deg, x, y):
        """how well the points (car coordinates) fit when the car is at (th, x, y): 0..255 per point"""
        t = math.radians(th_deg)
        c, s = math.cos(t), math.sin(t)
        g, n, inv, half = self.g, self.n, 1.0 / self.cell, GRID_HALF_MM
        total = 0
        for i in range(len(xs)):
            px = xs[i]; py = ys[i]
            ix = int((x + px * c + py * s + half) * inv)
            iy = int((y - px * s + py * c + half) * inv)
            if 0 <= ix < n and 0 <= iy < n:
                total += g[iy * n + ix]
        return total

    def share(self, xs, ys, th_deg, x, y):
        """share of the points that lie on something in this grid"""
        t = math.radians(th_deg)
        c, s = math.cos(t), math.sin(t)
        g, n, inv, half = self.g, self.n, 1.0 / self.cell, GRID_HALF_MM
        hit = 0
        for i in range(len(xs)):
            px = xs[i]; py = ys[i]
            ix = int((x + px * c + py * s + half) * inv)
            iy = int((y - px * s + py * c + half) * inv)
            if 0 <= ix < n and 0 <= iy < n and g[iy * n + ix] > 0:
                hit += 1
        return hit / max(1, len(xs))

    def fit(self, xs, ys, th, x, y, th_step=3.0, xy_step=32.0, th_end=0.2, xy_end=3.0):
        """
        Starting from the guess (th, x, y): shift/turn the look in ever smaller
        steps as long as it fits better. Returns (th, x, y, score).
        """
        best = self.score(xs, ys, th, x, y)
        while th_step >= th_end or xy_step >= xy_end:
            improved = False
            for dth, dx, dy in ((th_step, 0.0, 0.0), (-th_step, 0.0, 0.0), (0.0, xy_step, 0.0),
                                (0.0, -xy_step, 0.0), (0.0, 0.0, xy_step), (0.0, 0.0, -xy_step)):
                sc = self.score(xs, ys, th + dth, x + dx, y + dy)
                if sc > best:
                    best = sc; th += dth; x += dx; y += dy
                    improved = True
            if not improved:
                th_step *= 0.5; xy_step *= 0.5
        return th, x, y, best


def thin(xs, ys, want=MATCH_POINTS):
    """every k-th point, so that about `want` are left"""
    k = max(1, len(xs) // want)
    return xs[::k], ys[::k]


class LidarOdometry:
    """Position and heading of the car relative to where it started, from the LIDAR alone."""

    def __init__(self):
        self.th = 0.0            # heading, deg, clockwise, NOT wrapped (counts full turns)
        self.x = 0.0             # mm, start frame: x = to the right of the start heading
        self.y = 0.0             #                  y = ahead of the start point
        self.start = None        # the first look (ScanGrid)
        self._key = None         # current keyframe
        self._key_pose = (0.0, 0.0, 0.0)
        self._rel = (0.0, 0.0, 0.0)     # pose relative to the keyframe
        self.share = 1.0

    def begin(self, xs, ys):
        self.start = ScanGrid(xs, ys)
        self._start_wide = ScanGrid(xs, ys, cell=START_CELL_MM, reach=START_REACH_MM)   # the same, blurred
        self._key, self._key_pose, self._rel = self.start, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        self.th = self.x = self.y = 0.0

    def _compose(self, key_pose, rel):
        kt, kx, ky = key_pose
        t = math.radians(kt)
        c, s = math.cos(t), math.sin(t)
        return kt + rel[0], kx + rel[1] * c + rel[2] * s, ky - rel[1] * s + rel[2] * c

    def update(self, xs, ys, turned_deg=0.0, moved_mm=0.0, unsure_deg=0.0, move_range=None):
        """
        A new look. turned_deg / moved_mm: about how the car has moved since
        the last look (a guess is enough). unsure_deg: how wrong the turn
        guess may be; move_range = (least, most) mm the car may have gone
        forward. Returns True if the look could be laid over the keyframe.

        The walls are rows of equal posts, so a look also fits "quite well"
        when it is laid one post too far along a wall. That is why not only
        the guess is followed up: every possible amount of movement is tried
        roughly first, the best few are worked out exactly, and the one that
        fits best of all wins (only the true one also fits the ends of the
        walls and everything else that is not a row of posts).
        """
        sx, sy = thin(xs, ys)
        th0, x0, y0 = self._rel
        t = math.radians(th0)
        ux, uy = math.sin(t), math.cos(t)
        turns = [turned_deg]
        k = 4.0
        while k <= unsure_deg + 0.01:
            turns += [turned_deg - k, turned_deg + k]
            k += 4.0
        moves = [moved_mm]
        if move_range is not None:
            d = move_range[0]
            while d <= move_range[1]:
                if abs(d - moved_mm) > MOVE_TRY_MM / 2:
                    moves.append(d)
                d += MOVE_TRY_MM
            if len(turns) == 1:
                turns = [turned_deg - 4.0, turned_deg, turned_deg + 4.0]
        score = self._key.score
        cands = [(score(sx, sy, th0 + a, x0 + d * ux, y0 + d * uy), th0 + a, x0 + d * ux, y0 + d * uy)
                 for d in moves for a in turns]
        cands.sort(reverse=True)
        best, done = None, []
        for _, th, x, y in cands:
            if any(abs(th - q[0]) < 6.0 and abs(x - q[1]) < 30.0 and abs(y - q[2]) < 30.0 for q in done):
                continue
            done.append((th, x, y))
            sol = self._key.fit(sx, sy, th, x, y)
            if best is None or sol[3] > best[3]:
                best = sol
            if len(done) >= 3:
                break
        th, x, y, _ = best
        self.share = self._key.share(sx, sy, th, x, y)
        if self.share < MATCH_MIN_SHARE:
            # not trusted: keep the guess (better than nothing) and start from a fresh keyframe
            self._rel = (th0 + turned_deg, x0 + moved_mm * ux, y0 + moved_mm * uy)
            self.th, self.x, self.y = self._compose(self._key_pose, self._rel)
            self._new_key(xs, ys)
            return False
        self._rel = (th, x, y)
        self.th, self.x, self.y = self._compose(self._key_pose, self._rel)
        if (math.hypot(x, y) > KEY_MOVE_MM or abs(th) > KEY_TURN_DEG or self.share < KEY_MIN_SHARE):
            self._new_key(xs, ys)
        return True

    def _new_key(self, xs, ys):
        self._key = ScanGrid(xs, ys)
        self._key_pose = (self.th, self.x, self.y)
        self._rel = (0.0, 0.0, 0.0)

    def see_start(self, xs, ys, reach_mm=START_SEARCH_MM, reach_deg=START_SEARCH_DEG):
        """
        Compare this look with the first look, around where the odometry
        thinks the car is (first roughly, on a blurred copy of the first
        look, then exactly). Returns (th, x, y, share) relative to the start
        point (th wrapped to -180..180); share = how much of the look fits.
        """
        sx, sy = thin(xs, ys)
        cx, cy = sx[::2], sy[::2]
        th0, x0, y0 = _wrap(self.th), self.x, self.y
        wide = self._start_wide
        cands = []
        steps = int(reach_mm // START_STEP_MM)
        turns = int(reach_deg // 10)
        for i in range(-steps, steps + 1):
            for j in range(-steps, steps + 1):
                for k in range(-turns, turns + 1):
                    th, x, y = th0 + 10.0 * k, x0 + START_STEP_MM * i, y0 + START_STEP_MM * j
                    cands.append((wide.score(cx, cy, th, x, y), th, x, y))
        cands.sort(reverse=True)
        best, tried = None, []
        for _, th, x, y in cands[:12]:
            if any(abs(th - t[0]) < 11.0 and abs(x - t[1]) < 130.0 and abs(y - t[2]) < 130.0 for t in tried):
                continue                          # next to one that was already followed up
            tried.append((th, x, y))
            th, x, y, _ = wide.fit(cx, cy, th, x, y, th_step=5.0, xy_step=60.0, th_end=1.0, xy_end=12.0)
            # around the rough answer, the exact grid may have several "nearly right" places one post
            # apart: look at all of them and follow up the best
            near = max(((self.start.score(cx, cy, th, x + 22.0 * a, y + 22.0 * b), a, b)
                        for a in range(-4, 5) for b in range(-4, 5)))
            sol = self.start.fit(sx, sy, th, x + 22.0 * near[1], y + 22.0 * near[2], th_step=2.0, xy_step=10.0)
            if best is None or sol[3] > best[3]:
                best = sol
            if len(tried) >= 4:
                break
        th, x, y, _ = best
        return _wrap(th), x, y, self.start.share(sx, sy, th, x, y)

    def set_pose(self, th_wrapped, x, y, xs, ys):
        """the car's pose is known exactly (it saw the start): continue from there"""
        self.th += _wrap(th_wrapped - self.th)
        self.x, self.y = x, y
        self._new_key(xs, ys)


# ═══════════════════════════════════════════════════════════════
#  DRIVING
# ═══════════════════════════════════════════════════════════════

# ── Where to go ────────────────────────────────────────────────
LOOK_AHEAD_MM    = (320.0, 240.0, 170.0)   # how far ahead to look for the middle of the passage
MIN_CLEAR_MM     = 110.0   # the LIDAR is never sent closer than this to anything
CLEAR_CAP_MM     = 330.0   # more room than this is not "better" (wide open places: just go on)
SEARCH_DEG       = 96      # the way on is looked for this far to each side of the direction of travel
SEARCH_STEP_DEG  = 6
MID_WEIGHT       = 1.25    # the room half-way to the point counts this much against the room at the point
REACH_COST       = 0.25    # a shorter hop must have this much more room per mm it is shorter
DIR_COST_MM_DEG  = 0.4     # a way that needs turning must be this much roomier per degree
DIR_FOLLOW       = 0.4     # how fast the remembered direction of travel follows the chosen way
# Keeping away from a wall that is close beside the car (the end of the inner wall in a U-turn):
SIDE_COMFORT_MM  = 150.0   # a wall beside the car closer than this, with more room on the other side ...
SIDE_DIFF_MM     = 50.0    # ... (at least this much more) ...
SIDE_PUSH_DEG_MM = 0.30    # ... turns the aim away from it by this many degrees per mm it is too close ...
SIDE_PUSH_MAX_DEG = 25.0   # ... but never by more than this
FRONT_CONE_DEG   = 30.0    # "ahead" = this far to each side of straight ahead
NARROW_CONE_DEG  = 12.0    # "dead ahead"

# ── General ────────────────────────────────────────────────────
START_DELAY_S    = 1.5     # after RUN is pressed: time to take the hand away from the car

# ── STEPS ──────────────────────────────────────────────────────
FORWARD_PULSE_S  = 0.20    # one forward step (about 5 cm on this car)
LONG_PULSE_S     = 0.30    # a longer step where the way ahead is straight and open
LONG_FRONT_MM    = 430.0   # ... nothing dead ahead closer than this
LONG_GUESS       = 3.5     # a long step goes about this many times as far as a normal one (until measured)
LONG_AIM_DEG     = 8.0     # ... and the car pointing within this of the way to go
SETTLE_S         = 0.30    # wait after a pulse before looking
RECHECK_S        = 0.09    # two looks this far apart must agree = the car really stands still
STILL_MM         = 6.0
TURN_TOL_DEG     = 12.0    # pointing more than this off the way to go -> turn before driving on
TURN_SHARE       = 0.85    # turn this share of the error (a little short rather than too far)
# How fast this car turns on the spot changes a lot from day to day (battery, floor): measured
# 100 deg/s on one day and 250 deg/s on another — but always "nothing for the first 0.065 s, then
# at a steady rate". So the rate is not fixed here: it is measured after every turn.
TURN_DEAD_S      = 0.065   # a turn pulse does nothing for this long
TURN_RATE_GUESS  = 170.0   # deg/s — only for the very first turn of a drive
TURN_MIN_S, TURN_MAX_S = 0.08, 0.45
STEP_GUESS_MM    = 70.0    # how far one forward step goes — only a first guess, measured after every step
FRONT_CLEAR_MM   = 190.0   # nothing may be closer than this dead ahead before a forward step
SIDE_CLEAR_MM    = 150.0   # ... and nothing closer than this anywhere ahead
STEP_MARGIN_MM   = 130.0   # ... and after the step this much must still be free dead ahead
MAX_TURNS_IN_ROW = 4       # after this many turns without a forward step: take a small step anyway
STUCK_STEPS      = 3       # forward steps in a row without movement -> back off
MAX_STEPS_PER_LAP = 500

# ── CRUISE ─────────────────────────────────────────────────────
CRUISE           = False   # False = STEPS only (stops after every move). True = also drive without
                           # stopping where the way is clear — NOT yet tried on the real car on this kind of track.
CRUISE_SPEED     = 100     # % — the faster side
CRUISE_SLOW_SIDE_MIN = 25  # % of CRUISE_SPEED — the slowest the inner side goes (do not go lower)
CRUISE_STRAIGHT_S = 0.20   # at the start of a cruise: this long straight, so both sides get rolling ...
CRUISE_RAMP_S    = 0.30    # ... then the steering is let in gradually over this long
CRUISE_FULL_DEG  = 28.0    # the way to go this far to the side = full steering
CRUISE_LOOK_AHEAD_MM = (380.0, 300.0, 220.0)   # look a little farther ahead than in STEPS
CRUISE_ENTER_DEG = 10.0    # start a cruise only when pointing within this of the way to go
CRUISE_ENTER_FRONT_MM = 320.0   # ... and nothing dead ahead closer than this
CRUISE_FRONT_MM  = 240.0   # leave CRUISE if something is closer than this dead ahead
CRUISE_SIDE_FRONT_MM = 170.0    # ... or closer than this anywhere ahead
CRUISE_EDGE_MM   = 95.0    # ... or closer than this anywhere beside the car
CRUISE_MAX_AIM_DEG = 55.0  # ... or the way to go is this far to the side
CRUISE_EXIT_MM   = 650.0   # on the last lap: this close to the start point, switch to STEPS

# ── Laps and the finish ────────────────────────────────────────
AWAY_MM          = 500.0   # the car has "left the start" when it is this far from it
NEAR_MM          = 900.0   # the start is looked for when the car thinks it is this close to it ...
GATE_HEAD_DEG    = 70.0    # ... heading about as at the start ...
GATE_TURN_DEG    = 120.0   # ... and has turned (nearly) a full circle since it was last there
START_MIN_SHARE  = 0.90    # this share of a look must fit the first look to say "this is the start"
FINISH_TOL_MM    = 20.0    # closer than this to the start line = arrived
FINISH_SLOW_MM   = 130.0   # within this of the finish: shorter forward steps
FINISH_AIM_MM    = 160.0   # farther than this from the finish: aim at the start point itself; closer: straight on
FINISH_HEADING_TOL_DEG = 5.0


def pick_way(xs, ys, prefer_deg, reaches=LOOK_AHEAD_MM):
    """
    The way to go: of the points `reach` mm from the car, within SEARCH_DEG
    of the direction of travel (prefer_deg, in the car's coordinates), the
    one with the most room around it AND around the straight line to it
    (so that corners are not cut). Returns (bearing_deg (+ = to the right),
    room_mm, reach_mm) or None.
    """
    lim = (reaches[0] + CLEAR_CAP_MM) ** 2
    near = [(x, y) for x, y in zip(xs[::2], ys[::2]) if x * x + y * y < lim]
    if not near:
        return prefer_deg, CLEAR_CAP_MM, reaches[0]
    floor2 = MIN_CLEAR_MM * MIN_CLEAR_MM

    def room2(cx, cy):
        best = 1e12
        for x, y in near:
            d = (x - cx) * (x - cx) + (y - cy) * (y - cy)
            if d < best:
                best = d
                if d < floor2:
                    break
        return best

    def rate(reach, k):
        """room (mm) for going `reach` mm in direction prefer + k; 0 if not possible"""
        b = math.radians(prefer_deg + k)
        sx, sy = reach * math.sin(b), reach * math.cos(b)
        end = room2(sx, sy)
        if end < floor2:
            return 0.0
        mid = room2(0.5 * sx, 0.5 * sy)
        if mid < floor2:
            return 0.0
        return min(math.sqrt(end), MID_WEIGHT * math.sqrt(mid), CLEAR_CAP_MM)

    # Every direction at every reach is rated; the best one wins. A shorter hop is taken when
    # it has clearly more room around it than the long one — that is what makes the car swing
    # wide around the end of a wall instead of cutting the corner right next to it.
    best = None
    for reach in reaches:
        k = -SEARCH_DEG
        while k <= SEARCH_DEG:
            room = rate(reach, k)
            if room > 0.0:
                score = room - DIR_COST_MM_DEG * abs(k) - REACH_COST * (reaches[0] - reach)
                if best is None or score > best[0]:
                    best = (score, k, room, reach)
            k += SEARCH_STEP_DEG
    if best is None:
        return None
    score, k, room, reach = best
    for fine in (-SEARCH_STEP_DEG / 2.0, SEARCH_STEP_DEG / 2.0):             # a little finer around it
        r = rate(reach, k + fine)
        sc = r - DIR_COST_MM_DEG * abs(k + fine) - REACH_COST * (reaches[0] - reach)
        if r > 0.0 and sc > score:
            score, k, room = sc, k + fine, r
    return _wrap(prefer_deg + k), room, reach


def room_along(xs, ys, bearing_deg, dist_mm):
    """the least room (mm) around the straight line from the car, `dist_mm` long, in that direction"""
    b = math.radians(bearing_deg)
    sx, sy = math.sin(b), math.cos(b)
    lim = (dist_mm + 400.0) ** 2
    near = [(x, y) for x, y in zip(xs[::2], ys[::2]) if x * x + y * y < lim]
    best = 1e12
    n = max(1, int(dist_mm // 60))
    for k in range(1, n + 1):
        cx, cy = sx * dist_mm * k / n, sy * dist_mm * k / n
        for x, y in near:
            d = (x - cx) * (x - cx) + (y - cy) * (y - cy)
            if d < best:
                best = d
    return math.sqrt(best)


def cones(xs, ys):
    """closest thing (mm): dead ahead, anywhere ahead, ahead-left, ahead-right, beside/ahead of the car"""
    narrow = front = left = right = beside = 1e9
    tn_n = math.tan(math.radians(NARROW_CONE_DEG)); tn_f = math.tan(math.radians(FRONT_CONE_DEG))
    for i in range(len(xs)):
        x = xs[i]; y = ys[i]
        d = math.hypot(x, y)
        if y > 0:
            ax = abs(x)
            if ax <= y * tn_f:
                if d < front:
                    front = d
                if ax <= y * tn_n and d < narrow:
                    narrow = d
            if ax > y * 0.27 and ax < y * 3.7:          # 15..75 deg to the side
                if x < 0:
                    if d < left:
                        left = d
                elif d < right:
                    right = d
        if y > -0.36 * abs(x) and d < beside:            # everything except what is behind the car
            beside = d
    return narrow, front, left, right, beside


def sides(xs, ys):
    """closest thing (mm) beside the car: (on the right, on the left) — from 25 to 155 degrees off straight ahead"""
    right = left = 1e9
    for i in range(len(xs)):
        x = xs[i]; y = ys[i]
        ax = abs(x)
        if ax > 0.47 * abs(y) or (y < 0 and ax > 0.47 * -y):       # more than 25 deg off the car's own line
            if y > -2.14 * ax:                                         # ... and not behind it (within 155 deg)
                d = math.hypot(x, y)
                if x > 0:
                    if d < right:
                        right = d
                elif d < left:
                    left = d
    return right, left


def _signature(rev):
    """mean distance in each 10-degree sector — to tell whether two looks are the same"""
    tot = [0.0] * 36; cnt = [0] * 36
    for p in rev:
        if MIN_VALID_MM <= p[1] <= MAX_VALID_MM:
            k = int(p[0] % 360.0 // 10)
            tot[k] += p[1]; cnt[k] += 1
    return [tot[k] / cnt[k] if cnt[k] >= 4 else None for k in range(36)]


def _same(a, b):
    diffs = sorted(abs(x - y) for x, y in zip(a, b) if x is not None and y is not None)
    return len(diffs) >= 12 and diffs[len(diffs) // 2] <= STILL_MM


class TrackDriver:
    """Drives the car around any closed track without junctions. See the module text."""

    def __init__(self, motors, lidar, log=print, on_event=None):
        self._motors = motors
        self._lidar  = lidar
        self._log    = log
        self._event  = on_event
        self.odo     = LidarOdometry()
        self._turn_rate = TURN_RATE_GUESS   # measured: deg/s while turning on the spot
        self._step_mm   = STEP_GUESS_MM     # measured: how far one FORWARD_PULSE_S step moves the car
        self._long_mm   = None              # measured: the same for a LONG_PULSE_S step (None = not yet)

    # ── helpers ───────────────────────────────────────────────

    def _sleep(self, seconds, running) -> bool:
        end = time.time() + seconds
        while running():
            left = end - time.time()
            if left <= 0:
                return True
            time.sleep(min(0.02, left))
        return False

    def _say(self, text, quiet=False):
        if not quiet:
            self._log(f"[TrackDrive] {text}")
        if self._event:
            try:
                self._event(f"NAV {text}")
            except Exception:
                pass

    def _look(self, running, tries=6):
        """One LIDAR revolution taken while the car really stands still -> (xs, ys) or None."""
        last, rev = None, None
        for _ in range(tries):
            if not running():
                return None
            rev = self._lidar.get_last_revolution()
            sig = _signature(rev) if len(rev) >= MIN_POINTS else None
            if sig is not None and last is not None and _same(sig, last):
                break
            last = sig
            if not self._sleep(RECHECK_S, running):
                return None
        if rev is None or len(rev) < MIN_POINTS:
            return None
        return to_xy(rev)

    def _pulse(self, action, seconds, running) -> bool:
        if action == 'F':
            self._motors.forward()
        elif action == 'B':
            self._motors.backward()
        elif action == 'L':
            self._motors.turn_left()
        else:
            self._motors.turn_right()
        ok = self._sleep(seconds, running)
        self._motors.stop()
        return ok and self._sleep(SETTLE_S, running)

    def _turn_pulse(self, angle_deg):
        """how long to turn on the spot for this many degrees"""
        return max(TURN_MIN_S, min(TURN_MAX_S, angle_deg / self._turn_rate + TURN_DEAD_S))

    def _steer(self, u):
        """Forward, steering by u: 0 = straight, +1 = as hard as allowed to the right, -1 = to the left."""
        u = max(-1.0, min(1.0, u))
        fast = CRUISE_SPEED
        slow = fast * (1.0 - abs(u) * (1.0 - CRUISE_SLOW_SIDE_MIN / 100.0))
        self._motors._set_direction(False, True, False, True)
        if u > 0:
            self._motors._set_speed(slow, fast)          # (right, left): right side slower = turns right
        else:
            self._motors._set_speed(fast, slow)

    # ── laps / the start point ────────────────────────────────

    def _progress(self, xs, ys, try_start=True):
        """
        Keep track of the laps. Returns (near, exact): near = the car has come
        around and is close to the start point; exact = its pose relative to
        the start point was just found from the first look.
        """
        odo = self.odo
        dist = math.hypot(odo.x, odo.y)
        if dist > AWAY_MM:
            self._away = True
        came_round = abs(odo.th) > 360.0 * (self._laps_done + 1) - GATE_TURN_DEG
        near = self._away and came_round and dist < NEAR_MM and abs(_wrap(odo.th)) < GATE_HEAD_DEG
        exact = False
        if near and try_start:
            th, x, y, share = odo.see_start(xs, ys)
            if share >= START_MIN_SHARE and abs(th) < GATE_HEAD_DEG and math.hypot(x, y) < NEAR_MM:
                odo.set_pose(th, x, y, xs, ys)
                exact = True
                self._start_seen = True
        # crossing the start line (the line through the start point, across the start heading)
        if near and self._prev_y is not None and self._prev_y < 0.0 <= odo.y and self._laps_done < self._laps - 1:
            self._laps_done += 1
            self._away = False
            self._say(f"lap {self._laps_done}/{self._laps} done" + ("" if self._start_seen else " (start point estimated)"))
            self._start_seen = False
            near = False
        self._prev_y = odo.y
        return near, exact

    # ── CRUISE ────────────────────────────────────────────────

    def _cruise(self, running) -> str:
        """
        Drive without stopping until something ends it. Returns why:
        'finish', 'blocked', 'edge', 'sharp', 'no way', 'lost' or 'cancelled'.
        The car is stopped on return.
        """
        odo = self.odo
        t_on = t_prev = time.time()
        yaw = speed = 0.0                    # how the car is moving (deg/s clockwise, mm/s), from the odometry
        prev = (odo.th, odo.x, odo.y)
        lost, n, reason = 0, 0, 'cancelled'
        self._steer(0.0)                     # start straight: both sides must get rolling first
        self._say("CRUISE on")
        while running():
            if not self._sleep(max(0.02, loc_period(self._lidar) - (time.time() - t_prev)), running):
                break
            now = time.time()
            dt, t_prev = now - t_prev, now
            n += 1
            rev = self._lidar.get_last_revolution()
            if len(rev) < MIN_POINTS:
                lost += 1
                if lost >= 2:
                    reason = 'lost'
                    break
                continue
            xs, ys = to_xy(rev, yaw, speed)
            if odo.update(xs, ys, yaw * dt, speed * dt):
                lost = 0
            else:
                lost += 1
                if lost >= 2:
                    reason = 'lost'
                    break
            # how the car is moving now (for the next look)
            t = math.radians(odo.th)
            fwd = (odo.x - prev[1]) * math.sin(t) + (odo.y - prev[2]) * math.cos(t)
            new_yaw, new_speed = (odo.th - prev[0]) / dt, fwd / dt
            if abs(new_yaw) < 250.0 and -100.0 < new_speed < 700.0:
                yaw = 0.5 * yaw + 0.5 * new_yaw
                speed = 0.5 * speed + 0.5 * new_speed
            prev = (odo.th, odo.x, odo.y)

            near, _ = self._progress(xs, ys, try_start=(n % 3 == 0))
            final = self._laps_done >= self._laps - 1
            narrow, front, _, _, beside = cones(xs, ys)
            way = pick_way(xs, ys, _wrap(self._dir - odo.th), CRUISE_LOOK_AHEAD_MM)

            if final and near and math.hypot(odo.x, odo.y) < CRUISE_EXIT_MM:
                reason = 'finish'
            elif narrow < CRUISE_FRONT_MM or front < CRUISE_SIDE_FRONT_MM:
                reason = 'blocked'
            elif beside < CRUISE_EDGE_MM:
                reason = 'edge'
            elif way is None:
                reason = 'no way'
            elif abs(way[0]) > CRUISE_MAX_AIM_DEG:
                reason = 'sharp'
            else:
                reason = None
            if reason:
                break

            aim = way[0]
            self._dir += 0.6 * DIR_FOLLOW * _wrap(odo.th + aim - self._dir)
            limit = max(0.0, min(1.0, (now - t_on - CRUISE_STRAIGHT_S) / CRUISE_RAMP_S))
            u = max(-limit, min(limit, aim / CRUISE_FULL_DEG))
            self._steer(u)
            self._say(f"cruise lap {self._laps_done + 1}/{self._laps} | way {aim:+.0f}deg, room {way[1]:.0f}mm | steer {u:+.2f} | "
                      f"{speed:.0f}mm/s, turning {yaw:+.0f}deg/s | ahead {narrow:.0f}mm | "
                      f"from start: {odo.x:+.0f},{odo.y:+.0f}mm, turned {odo.th:+.0f}deg", quiet=(n % 3 != 0))
        self._motors.stop()
        self._say(f"CRUISE off: {reason}")
        self._coast = (yaw * 0.12, speed * 0.15)        # about how the car still moves while it stops
        if reason != 'cancelled' and not self._sleep(SETTLE_S, running):
            return 'cancelled'
        return reason

    # ── the drive ─────────────────────────────────────────────

    def drive(self, running, clockwise: bool = True, laps: int = 1) -> str:
        """
        Drive `laps` laps and stop on the start point.
        running: function returning False when the drive must be cancelled.
        (clockwise is not needed: the car simply goes on in the direction it is placed in.)
        Returns 'done', 'cancelled', 'no_track' (the LIDAR sees no way to
        drive from here — nothing was driven) or 'stuck'.
        """
        self._motors.stop()
        laps = max(1, int(laps))
        self._laps, self._laps_done = laps, 0
        self._away, self._start_seen, self._prev_y = False, False, None
        self._dir = 0.0                            # direction of travel, in the start frame
        self._coast = (0.0, 0.0)
        odo = self.odo = LidarOdometry()

        if not self._sleep(START_DELAY_S, running):
            return 'cancelled'
        look, deadline = None, time.time() + 6.0
        while running() and time.time() < deadline and look is None:
            look = self._look(running)
            if look is None and not self._sleep(0.25, running):
                return 'cancelled'
        if not running():
            return 'cancelled'
        if look is None or pick_way(look[0], look[1], 0.0) is None:
            self._say("the LIDAR sees no way to drive from here")
            return 'no_track'
        odo.begin(look[0], look[1])
        self._say(f"track: {len(look[0])} LIDAR points in the first look | {laps} lap(s), stopping on this start point")

        last_action, last_pulse, last_move = None, 0.0, 0.0
        before = (0.0, 0.0, 0.0)
        no_move = turns_in_row = step = bad = 0
        xs, ys = look
        while running():
            step += 1
            if step > MAX_STEPS_PER_LAP * laps:
                self._say("too many steps, stopping")
                return 'stuck'

            # ── where am I now? ──
            if step > 1:
                look = self._look(running)
                if look is None:
                    if not running():
                        return 'cancelled'
                    self._say("the LIDAR gives no usable look, stopping")
                    return 'stuck'
                xs, ys = look
                if last_action == 'F':
                    ok = odo.update(xs, ys, 0.0, last_move, move_range=(-20.0, max(220.0, 2.5 * last_move)))
                elif last_action == 'B':
                    ok = odo.update(xs, ys, 0.0, -self._step_mm, move_range=(-max(220.0, 2.5 * self._step_mm), 20.0))
                elif last_action in ('L', 'R'):
                    expect = self._turn_rate * max(0.0, last_pulse - TURN_DEAD_S)
                    ok = odo.update(xs, ys, expect if last_action == 'R' else -expect, 0.0,
                                    unsure_deg=max(16.0, 0.7 * expect))
                else:                                   # after a cruise
                    ok = odo.update(xs, ys, self._coast[0], self._coast[1], unsure_deg=12.0,
                                    move_range=(-20.0, max(200.0, 2.5 * self._coast[1])))
                bad = 0 if ok else bad + 1
                # measure what the last pulse really did, for the next ones
                moved = math.hypot(odo.x - before[1], odo.y - before[2])
                turned = abs(odo.th - before[0])
                if ok and last_action == 'F':
                    no_move = no_move + 1 if moved < 8.0 else 0
                    if moved >= 8.0 and abs(last_pulse - FORWARD_PULSE_S) < 0.01:
                        self._step_mm = max(25.0, min(220.0, 0.5 * self._step_mm + 0.5 * moved))
                    elif moved >= 8.0 and abs(last_pulse - LONG_PULSE_S) < 0.01:
                        self._long_mm = max(50.0, min(600.0, moved if self._long_mm is None else 0.5 * self._long_mm + 0.5 * moved))
                elif ok and last_action in ('L', 'R') and last_pulse > TURN_DEAD_S + 0.02 and turned > 2.0:
                    rate = turned / (last_pulse - TURN_DEAD_S)
                    if 40.0 < rate < 500.0:
                        self._turn_rate = 0.5 * self._turn_rate + 0.5 * rate

            near, exact = self._progress(xs, ys)
            final = self._laps_done >= laps - 1
            to_go = -odo.y                                  # mm to the start line (when near)
            narrow, front, left, right, beside = cones(xs, ys)
            way = pick_way(xs, ys, _wrap(self._dir - odo.th))

            status = (f"#{step} lap {self._laps_done + 1}/{laps} | from start: {odo.x:+.0f},{odo.y:+.0f}mm, turned {odo.th:+.0f}deg | "
                      + (f"way {way[0]:+.0f}deg, room {way[1]:.0f}mm" if way else "no way seen")
                      + f" | ahead {narrow:.0f}mm" + (" | START SEEN" if exact else "") + ("" if bad == 0 else " | position estimated"))

            # ── arrived? ──
            if final and near and to_go <= FINISH_TOL_MM:
                if not exact:
                    # by the odometry alone the car is there, but the first look does not fit here:
                    # look for the start point in a much wider area before believing that
                    th, x, y, share = odo.see_start(xs, ys, reach_mm=2 * START_SEARCH_MM, reach_deg=40.0)
                    if share >= START_MIN_SHARE and abs(th) < GATE_HEAD_DEG:
                        odo.set_pose(th, x, y, xs, ys)
                        self._prev_y = odo.y
                        self._say(status + f" -> the start point is really {math.hypot(x, y):.0f}mm from here: going on")
                        last_action = 'look'
                        before = (odo.th, odo.x, odo.y)
                        self._coast = (0.0, 0.0)
                        continue
                    self._say(status + " -> at the start point (by the estimate only: the LIDAR could not confirm it)")
                    break
                self._say(status + " -> at the start point")
                break
            # gone a whole lap too far without finding the start: do not drive on for ever
            if abs(odo.th) > 360.0 * laps + 300.0:
                self._say(status + " -> the start point was not found again, stopping")
                self._motors.stop()
                return 'stuck'

            # ── which way? ──
            aim = way[0] if way else None
            finishing = final and near and exact
            if finishing and to_go > FINISH_AIM_MM:
                # the last bit: aim at the start point itself (if the way to it is free)
                to_start = _wrap(math.degrees(math.atan2(-odo.x, -odo.y)) - odo.th)
                if abs(to_start) < 50.0 and aim is not None and \
                        room_along(xs, ys, to_start, math.hypot(odo.x, odo.y)) >= MIN_CLEAR_MM + 30.0:
                    aim = to_start
            elif finishing:
                aim = 0.0                                   # nearly there: straight on
            if way:
                self._dir += DIR_FOLLOW * _wrap(odo.th + way[0] - self._dir)
            # A wall close beside the car while the other side has room (the end of the inner wall in
            # a U-turn): do not turn toward it yet — the turn toward that side is made smaller, so the
            # car first drives on a little, away from the wall, and turns when it has room. (Only a
            # turn TOWARD the wall is reduced; this never makes the car turn the other way.)
            side_r, side_l = sides(xs, ys)
            if aim is not None and not (finishing and to_go <= FINISH_AIM_MM) \
                    and min(side_r, side_l) < SIDE_COMFORT_MM and abs(side_r - side_l) > SIDE_DIFF_MM:
                push = min(SIDE_PUSH_MAX_DEG, SIDE_PUSH_DEG_MM * (SIDE_COMFORT_MM - min(side_r, side_l)))
                if side_r < side_l and aim > 0.0:           # wall on the right, way to the right
                    aim = max(0.0, aim - push)
                    status += f" | wall {side_r:.0f}mm on the right: aim {aim:+.0f}deg"
                elif side_l < side_r and aim < 0.0:         # wall on the left, way to the left
                    aim = min(0.0, aim + push)
                    status += f" | wall {side_l:.0f}mm on the left: aim {aim:+.0f}deg"

            # ── CRUISE when everything is in order ──
            if (CRUISE and aim is not None and abs(aim) <= CRUISE_ENTER_DEG and no_move == 0 and bad == 0
                    and narrow >= CRUISE_ENTER_FRONT_MM and front >= FRONT_CLEAR_MM and beside >= CRUISE_EDGE_MM + 15.0
                    and not (final and near)):
                self._say(status + " -> cruise")
                why = self._cruise(running)
                if why == 'cancelled':
                    return 'cancelled'
                last_action, turns_in_row = None, 0
                before = (odo.th, odo.x, odo.y)
                continue

            # ── one step ──
            room_ahead = min(narrow - STEP_MARGIN_MM, front - STEP_MARGIN_MM + 40.0)    # how far a step may go
            blocked = narrow < FRONT_CLEAR_MM or front < SIDE_CLEAR_MM or room_ahead < 0.6 * self._step_mm
            last_move = 0.0
            # a long step: until one was measured, expect it to go (at least) LONG_GUESS x a normal step
            long_mm = self._long_mm if self._long_mm is not None else LONG_GUESS * self._step_mm
            if no_move >= STUCK_STEPS:
                self._say(status + " -> not moving: backing off")
                action, pulse = 'B', FORWARD_PULSE_S
                no_move = 0
            elif aim is None:
                # no way seen: turn toward the roomier side and look again
                action = 'R' if right > left else 'L'
                pulse = self._turn_pulse(20.0)
                self._say(status + f" -> turn '{action}' {pulse:.2f}s (looking for the way)")
            elif turns_in_row >= MAX_TURNS_IN_ROW and not blocked:
                action, pulse = 'F', 0.6 * FORWARD_PULSE_S
                last_move = 0.5 * self._step_mm
                self._say(status + f" -> forward {pulse:.2f}s (enough turning)")
            elif abs(aim) > TURN_TOL_DEG or blocked:
                if blocked and abs(aim) <= TURN_TOL_DEG:
                    turn_right, angle = (right > left), 15.0     # something ahead: toward the roomier side
                else:
                    turn_right, angle = (aim > 0), min(TURN_SHARE * abs(aim), 40.0)
                action = 'R' if turn_right else 'L'
                pulse = self._turn_pulse(angle)
                self._say(status + f" -> turn '{action}' {pulse:.2f}s" + (" (blocked ahead)" if blocked else ""))
            else:
                action, pulse, last_move = 'F', FORWARD_PULSE_S, self._step_mm
                if final and near and to_go < max(FINISH_SLOW_MM, 1.2 * self._step_mm):   # close to the finish: shorter steps
                    pulse = max(0.07, min(FORWARD_PULSE_S, FORWARD_PULSE_S * to_go / max(self._step_mm, 20.0)))
                    last_move = min(self._step_mm, to_go)
                elif abs(aim) <= LONG_AIM_DEG and narrow >= LONG_FRONT_MM and room_ahead >= 1.5 * long_mm \
                        and way and way[1] >= MIN_CLEAR_MM + 40.0 and not (final and near and to_go < 2.0 * long_mm):
                    pulse, last_move = LONG_PULSE_S, long_mm
                elif room_ahead < self._step_mm:                # little room: a shorter step
                    pulse = max(0.07, FORWARD_PULSE_S * room_ahead / self._step_mm)
                    last_move = room_ahead
                self._say(status + f" -> forward {pulse:.2f}s")

            turns_in_row = turns_in_row + 1 if action in ('L', 'R') else 0
            last_action, last_pulse = action, pulse
            before = (odo.th, odo.x, odo.y)
            if not self._pulse(action, pulse, running):
                return 'cancelled'

        if not running():
            return 'cancelled'

        # ── on the start point: turn back to the starting direction ──
        pose, flips, last_sign = None, 0, 0
        for _ in range(5):
            look = self._look(running)
            if look is None:
                break
            th, x, y, share = odo.see_start(look[0], look[1], reach_mm=240.0)
            if share < START_MIN_SHARE:
                break
            if pose is not None and last_pulse > TURN_DEAD_S + 0.02 and abs(th - pose[0]) > 2.0:
                rate = abs(th - pose[0]) / (last_pulse - TURN_DEAD_S)          # what the last turn really did
                if 40.0 < rate < 500.0:
                    self._turn_rate = 0.5 * self._turn_rate + 0.5 * rate
            pose = (th, x, y)
            sign = 1 if th > 0 else -1
            flips += 1 if sign == -last_sign else 0
            last_sign = sign
            if abs(th) <= FINISH_HEADING_TOL_DEG or (flips >= 2 and abs(th) <= 3 * FINISH_HEADING_TOL_DEG):
                break                                   # there — or as close as this car's shortest turn gets
            action = 'L' if th > 0 else 'R'
            last_pulse = self._turn_pulse((TURN_SHARE if flips == 0 else 0.5) * abs(th))
            self._say(f"finish: turned {th:+.0f}deg against the start direction -> turn '{action}' {last_pulse:.2f}s")
            if not self._pulse(action, last_pulse, running):
                return 'cancelled'
        if pose is not None:
            self._say(f"FINISHED: {laps} lap(s). Standing {math.hypot(pose[1], pose[2]):.0f}mm from the start point "
                      f"({pose[1]:+.0f} to the side, {pose[2]:+.0f} ahead), heading {pose[0]:+.0f}deg from the start direction")
        else:
            self._say(f"FINISHED: {laps} lap(s) (final position not confirmed by the LIDAR)")
        return 'done'


def loc_period(lidar) -> float:
    """seconds per LIDAR revolution (how often a new look is available)"""
    period = getattr(lidar, 'rev_time', None)
    return period if period and 0.08 <= period <= 0.4 else 1.0 / 6.0
