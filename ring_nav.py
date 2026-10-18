"""
ring_nav.py
===========
Autonomous driving on the RING TRACK using the LIDAR — the car drives by
WHERE IT IS, not by how long each button was held.

The track (as mapped from the recorded scans): an outer ring of about 45 round
posts (about 125 cm across) and a cluster of posts in the middle; the car
drives in the lane between them.

Two parts, both plain Python (no extra libraries):

1. RingLocalizer — "where am I?", from ONE LIDAR revolution:
     rho    distance of the car from the centre of the ring (mm)
     eps    how many degrees the car points OUTWARD (+) or INWARD (-) of the
            driving direction along the lane
     pose   its exact place on the ring and its heading
   rho and eps come from fitting a circle to the outer ring — that works on
   every revolution, also while moving. The pose comes from the POSTS: each
   revolution the posts in view are found one by one (every post is a small
   arc of points) and laid over a map of all the posts. The posts do not
   stand on a perfect circle — each is a little nearer or farther — and that
   makes every part of the ring different, from wherever it is looked at.
   The map of the posts is built by the car itself while it drives.

2. RingDriver — "what now?", in two ways of driving:
     CRUISE  the normal one: drives forward without stopping and steers by
             slowing one side (like the app's forward-left / forward-right
             buttons). About 6 times a second it compares how fast the car
             is turning with how fast it should be, and adjusts.
     STEPS   look (standing still) - decide - one short move - look again.
             Used to line the car up at the start, to get back into the lane
             if something went wrong, and for the last bit to the start
             point, where it has to stop exactly.
   It never drives forward when something is too close ahead, and it leaves
   CRUISE the moment it is too far from the lane or loses sight of the ring.
   After the recorded number of laps it stops on the start point and turns
   back to the direction it started in.

All the numbers that may need tuning on the real car are the constants below.
The measured behaviour of this car (from its own recordings):
   forward, both sides 100%          : about 350 mm/s, straight
   forward-right, right side at 25%  : about 310 mm/s, turning 58-70 deg/s once rolling,
                                       but spinning at 100 deg/s or more from a standing start
   turn on the spot                  : about 100 deg/s x pulse - 7 deg
"""

import bisect
import json
import math
import os
import time

# ── Reading a LIDAR revolution ─────────────────────────────────
MIN_VALID_MM  = 80        # closer than this is the car itself
MIN_POINTS    = 250       # a usable revolution has at least this many points
BAND_MM       = 60.0      # points within this of the ring circle are "outer ring"
MIN_RING_PTS  = 150       # the outer ring must be seen by at least this many points
MAX_RING_RMS  = 30.0      # mm — worse than this is not our ring
RING_R_MIN, RING_R_MAX = 300.0, 2000.0   # plausible ring radius, mm
INNER_ZONE    = 0.40      # only things closer to the centre than this share of the radius are the
                          # "inner cluster"; anything else inside the lane (a hand, an object) is
                          # just an obstacle

# ── Is this the plain ring track at all? ───────────────────────
# This module only knows a closed ring with a small cluster in the middle. Anything
# else (an "8", an oval, a ring with a wall in it ...) is driven by track_nav instead.
PLAIN_OUTSIDE_MM  = 120.0  # points farther than this beyond the ring: the ring is open there
PLAIN_LANE_MM     = 110.0  # points closer than this to the ring belong to it; farther in = in the lane
PLAIN_INNER_MAX   = 0.36   # the cluster in the middle reaches at most this share of the radius from the centre
PLAIN_MAX_OUTSIDE = 0.02   # at most this share of the points may lie beyond the ring ...
PLAIN_MAX_LANE    = 0.025  # ... or in the lane (farther from the centre than the cluster may reach) ...
PLAIN_MIN_SECTORS = 18     # ... and the ring must be seen in at least this many of 36 directions

# ── The posts ──────────────────────────────────────────────────
POST_RADIUS_MM   = 48.0   # the posts of the outer ring are about 96 mm across (measured from the scans)
POST_MIN_POINTS  = 4      # a post must be seen by at least this many points
POST_MATCH_MM    = 30.0   # a seen post within this of a map post is that post
POST_NEW_MM      = 60.0   # a seen post farther than this from every map post is a new one
POSE_MAX_RES_MM  = 9.0    # average distance seen-post to map-post above this = pose not trusted
POSE_MIN_SHARE   = 0.75   # of the seen posts that are near map posts, this share must lie ON them
POSE_MIN_MARGIN  = 1.5    # the best way of laying the posts over the map must beat the runner-up (mm)
POSE_MIN_POSTS   = 8
POSE_WINDOW_DEG  = 36     # look for the heading this far around the expected one
POSE_LOST_LOOKS  = 2      # after this many looks without a pose: search all the way around
POSE_GLOBAL_MARGIN = 3.0  # ... and such a search must be this clearly the best (mm)

# ── The map of the track (kept between drives) ─────────────────
MAP_FILE         = "ring_map.json"   # next to this file
MAP_MIN_POSTS    = 36     # the map is complete when it has at least this many posts ...
MAP_MIN_SEEN     = 2      # ... each seen at least this often ...
MAP_MAX_GAP_DEG  = 30.0   # ... and no gap between neighbouring posts wider than this

# ── Driving: general ───────────────────────────────────────────
START_DELAY_S    = 1.5    # after RUN is pressed: time to take the hand away from the car
WALL_MARGIN_MM   = 110.0  # keep the LIDAR at least this far from the inner cluster / outer ring
FRONT_CONE_DEG   = 30.0   # "ahead" = this far to each side of straight ahead
NARROW_CONE_DEG  = 12.0   # "dead ahead" (used while cruising)
EPS_MAX_DEG      = 25.0   # never aim more than this across the lane

# ── Driving: CRUISE ────────────────────────────────────────────
# What the real cruise runs showed: with one side slowed the car turns about
# 58-70 deg/s once it is rolling, but from a standing start the slowed side
# does not start at all and the car spins at 100 deg/s or more. So CRUISE
# (1) starts straight and lets the steering in gradually, and (2) does not
# trust any fixed "this much steering = this much turn": it measures how fast
# the car is really turning and adjusts the steering until that matches.
CRUISE              = True    # False = STEPS only (slow, stops after every move)
CRUISE_SPEED        = 100     # % — the faster side
CRUISE_SLOW_SIDE_MIN = 25     # % of CRUISE_SPEED — the slowest the inner side goes (the app uses 25).
                              # Do not go lower: a side that stops completely does not start again
                              # at low power, and the car then spins around it.
CRUISE_SPEED_MM_S   = 350.0   # measured: speed once rolling
CRUISE_ACCEL_MM_S2  = 650.0   # measured: how fast it picks up speed from standing
CRUISE_STRAIGHT_S   = 0.20    # at the start of a cruise: this long straight, so both sides get rolling ...
CRUISE_RAMP_S       = 0.30    # ... then the steering is let in gradually over this long
CRUISE_ENTER_LEAD_DEG = 14.0  # measured: the steering takes ~0.4 s to bite, and meanwhile the lane keeps
                              # curving. So a cruise is started with the car pointing this much further
                              # toward the centre than it should end up (a driver does the same).
CRUISE_K_RHO        = 0.12    # deg of aim per mm off the lane centre (gentler than in STEPS)
CRUISE_EPS_MAX_DEG  = 12.0    # never aim more than this across the lane while cruising
CRUISE_K_HEADING    = 1.5     # wanted extra turn rate per degree of heading error (deg/s per deg)
CRUISE_YAW_MAX      = 85.0    # never ask for more turn rate than this toward the centre (deg/s) ...
CRUISE_YAW_MIN      = -25.0   # ... or this away from it
CRUISE_YAW_GUESS    = 70.0    # rough guess: full steering turns about this fast. Half of the steering
                              # is set straight from this guess, the rest is found by measuring.
CRUISE_KP           = 0.004   # steering per deg/s of missing turn rate, at once
CRUISE_KI           = 0.035   # ... and per second while it stays missing
CRUISE_FRONT_MM     = 240.0   # leave CRUISE if something is closer than this dead ahead
CRUISE_SIDE_FRONT_MM = 170.0  # ... or closer than this anywhere ahead
CRUISE_EDGE_MM      = 40.0    # ... or the car is (about to be) this close to the allowed edge of the lane
CRUISE_MAX_ERR_DEG  = 30.0    # ... or it points this far off
CRUISE_LOOKAHEAD_S  = 0.35    # steer by where the car will be this much later, not where it was seen
CRUISE_ENTER_ERR_DEG = 7.0    # start/resume CRUISE only when pointing within this of that lead
CRUISE_ENTER_EDGE_MM = 45.0   # ... and this far inside the lane
CRUISE_ENTER_FRONT_MM = 280.0 # ... and nothing dead ahead closer than this
CRUISE_EXIT_DEG     = 45.0    # this far (around the ring) before the finish: switch to STEPS
CRUISE_POSE_EVERY   = 2       # find the exact pose on every N-th revolution (it costs time)

# ── Driving: STEPS ─────────────────────────────────────────────
FORWARD_PULSE_S  = 0.20   # one forward step (about 5 cm on this car)
SETTLE_S         = 0.30   # wait after a pulse before looking (car stops rolling + fresh revolution)
RECHECK_S        = 0.09   # two looks this far apart must agree = the car really stands still
STILL_MM         = 5.0
K_RHO_DEG_PER_MM = 0.20   # 50 mm off the lane centre -> aim 10 degrees back toward it
TURN_TOL_DEG     = 10.0   # heading error bigger than this -> turn before driving on
TURN_SHARE       = 0.85   # turn this share of the error (a little short rather than too far)
TURN_RATE_DEG_S  = 100.0  # measured: turn = 100 deg/s x pulse - 7 deg
TURN_DEAD_DEG    = 7.0
TURN_MIN_S, TURN_MAX_S = 0.10, 0.45
FRONT_CLEAR_MM   = 190.0  # nothing may be closer than this dead ahead before a forward step
SIDE_CLEAR_MM    = 150.0  # ... and nothing closer than this anywhere ahead
MAX_TURNS_IN_ROW = 4      # after this many turns without a forward step: take a small step anyway
                          # (if there is room), so the car can never just wiggle on the spot
FINISH_TOL_DEG   = 2.0    # closer than this (around the ring) to the start point = arrived
FINISH_SLOW_MM   = 90.0   # within this distance of the finish: shorter forward steps
FINISH_HEADING_TOL_DEG = 5.0
BLEND_DEG        = 220.0  # over the last part of the route, drift from the lane centre back to the
BLEND_END_DEG    = 70.0   # start point's distance from the centre; be there this long before the finish
MAX_STEPS_PER_LAP = 300
STUCK_STEPS      = 3      # forward steps in a row without movement -> back off


def _wrap(deg: float) -> float:
    """angle to -180..+180"""
    return (deg + 180.0) % 360.0 - 180.0


def _solve3(m, v):
    """solve a 3x3 linear system (Cramer); None if singular"""
    (a, b, c), (d, e, f), (g, h, i) = m
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-9:
        return None
    x = (v[0] * (e * i - f * h) - b * (v[1] * i - f * v[2]) + c * (v[1] * h - e * v[2])) / det
    y = (a * (v[1] * i - f * v[2]) - v[0] * (d * i - f * g) + c * (d * v[2] - v[1] * g)) / det
    z = (a * (e * v[2] - v[1] * h) - b * (d * v[2] - v[1] * g) + v[0] * (d * h - e * g)) / det
    return x, y, z


def _fit_circle(xs, ys):
    """least-squares circle through points: (cx, cy, r) or None"""
    n = len(xs)
    if n < 10:
        return None
    sx = sy = sxx = syy = sxy = sxz = syz = sz = 0.0
    for x, y in zip(xs, ys):
        z = x * x + y * y
        sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y
        sxz += x * z; syz += y * z; sz += z
    sol = _solve3(((sxx, sxy, sx), (sxy, syy, sy), (sx, sy, n)), (sxz, syz, sz))
    if sol is None:
        return None
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 0:
        return None
    return cx, cy, math.sqrt(r2)


def find_posts(xs, ys, cx, cy, radius):
    """
    The posts of the outer ring seen in one revolution, as their centres in
    the car's own coordinates: [(x, y), ...].
    A round post is closest to the LIDAR at exactly one point, so every post
    shows up as a dip in the distance when going around; a circle of the
    posts' size is then fitted through the points of each dip.
    """
    ring = []
    for x, y in zip(xs, ys):
        if abs(math.hypot(x - cx, y - cy) - radius) < BAND_MM + 10.0:
            ring.append((math.degrees(math.atan2(x, y)) % 360.0, math.hypot(x, y), x, y))
    if len(ring) < 20:
        return []
    ring.sort()
    # start the walk at the biggest gap in bearing, so no post is cut in two at 0/360
    gap_at, gap = 0, -1.0
    for i in range(len(ring)):
        g = (ring[i][0] - ring[i - 1][0]) % 360.0
        if g > gap:
            gap_at, gap = i, g
    ring = ring[gap_at:] + ring[:gap_at]

    posts = []
    n = len(ring)
    k = 0
    while k < n:
        j = k
        while j + 1 < n and (ring[j + 1][0] - ring[j][0]) % 360.0 < 2.5:
            j += 1
        seg = ring[k:j + 1]
        k = j + 1
        m = len(seg)
        if m < 5:
            continue
        d = [p[1] for p in seg]
        sm = [d[0]] + [(d[i - 1] + d[i] + d[i + 1]) / 3.0 for i in range(1, m - 1)] + [d[-1]]
        for q in range(2, m - 2):
            if not (sm[q] <= sm[q - 1] and sm[q] < sm[q + 1] and sm[q] <= sm[q - 2] and sm[q] < sm[q + 2]):
                continue
            lo = q
            while lo > 0 and sm[lo - 1] >= sm[lo]:
                lo -= 1
            hi = q
            while hi < m - 1 and sm[hi + 1] >= sm[hi]:
                hi += 1
            if sm[lo] - sm[q] < 4.0 or sm[hi] - sm[q] < 4.0:
                continue                                   # too shallow to be a post
            # first guess: the closest point, pushed one post radius further along its ray
            a = math.radians(seg[q][0])
            ex, ey = (d[q] + POST_RADIUS_MM) * math.sin(a), (d[q] + POST_RADIUS_MM) * math.cos(a)
            used = 0
            for _ in range(4):                             # fit a post-sized circle to the dip
                suu = suv = svv = bu = bv = 0.0
                used = 0
                for i in range(lo, hi + 1):
                    dx, dy = seg[i][2] - ex, seg[i][3] - ey
                    dist = math.hypot(dx, dy)
                    res = dist - POST_RADIUS_MM
                    if abs(res) < 10.0 and dist > 1.0:
                        ux, uy = dx / dist, dy / dist
                        suu += ux * ux; suv += ux * uy; svv += uy * uy
                        bu += ux * res; bv += uy * res
                        used += 1
                det = suu * svv - suv * suv
                if used < 3 or abs(det) < 1e-6:
                    break
                sx = (svv * bu - suv * bv) / det
                sy = (suu * bv - suv * bu) / det
                if abs(sx) > 25.0 or abs(sy) > 25.0:       # a dip that is not a post
                    used = 0
                    break
                ex += sx; ey += sy
            if used >= POST_MIN_POINTS:
                posts.append((ex, ey))
    return posts


class PostMap:
    """The map: where every post of the outer ring stands (origin = ring centre
    as first seen, y axis = the car's heading at the start)."""

    def __init__(self):
        self.posts = []          # [x, y, times seen]
        self._index = None       # (sorted angles, post indices)

    def __len__(self):
        return len(self.posts)

    def complete(self) -> bool:
        """True when the whole ring is on the map (see MAP_MIN_POSTS / MAP_MAX_GAP_DEG)."""
        angles = sorted(math.degrees(math.atan2(p[0], p[1])) % 360.0 for p in self.posts if p[2] >= MAP_MIN_SEEN)
        if len(angles) < MAP_MIN_POSTS:
            return False
        gaps = [(angles[(i + 1) % len(angles)] - angles[i]) % 360.0 for i in range(len(angles))]
        return max(gaps) <= MAP_MAX_GAP_DEG

    def _build_index(self):
        order = sorted(range(len(self.posts)), key=lambda i: math.atan2(self.posts[i][0], self.posts[i][1]))
        self._index = ([math.atan2(self.posts[i][0], self.posts[i][1]) for i in order], order)

    def nearest(self, wx, wy):
        """(index of the closest map post, its distance) — looks only at the posts in that direction"""
        if self._index is None:
            self._build_index()
        angles, order = self._index
        n = len(order)
        if n == 0:
            return None, 1e9
        k = bisect.bisect_left(angles, math.atan2(wx, wy))
        best_i, best = None, 1e9
        for j in (k - 2, k - 1, k, k + 1):
            i = order[j % n]
            dist = math.hypot(wx - self.posts[i][0], wy - self.posts[i][1])
            if dist < best:
                best_i, best = i, dist
        return best_i, best

    @staticmethod
    def to_world(obs, th_deg, x, y):
        t = math.radians(th_deg)
        c, s = math.cos(t), math.sin(t)
        return [(x + ox * c + oy * s, y - ox * s + oy * c) for ox, oy in obs]

    def _cost(self, obs, th, x, y):
        total = 0.0
        for wx, wy in self.to_world(obs, th, x, y):
            dist = self.nearest(wx, wy)[1]
            total += dist if dist < 40.0 else 40.0
        return total / len(obs)

    def _refine(self, obs, th, x, y):
        """slide/turn the seen posts onto their map posts (least squares, a few rounds)"""
        for _ in range(5):
            pairs = []
            for (ox, oy), (wx, wy) in zip(obs, self.to_world(obs, th, x, y)):
                i, dist = self.nearest(wx, wy)
                if dist < 35.0:
                    pairs.append((ox, oy, self.posts[i][0], self.posts[i][1]))
            if len(pairs) < 5:
                break
            n = float(len(pairs))
            cax = sum(p[0] for p in pairs) / n; cay = sum(p[1] for p in pairs) / n
            cbx = sum(p[2] for p in pairs) / n; cby = sum(p[3] for p in pairs) / n
            num = den = 0.0
            for ox, oy, mx, my in pairs:
                ax, ay, bx, by = ox - cax, oy - cay, mx - cbx, my - cby
                num += ay * bx - ax * by
                den += ax * bx + ay * by
            t = math.atan2(num, den)
            th = math.degrees(t)
            c, s = math.cos(t), math.sin(t)
            x = cbx - (cax * c + cay * s)
            y = cby - (-cax * s + cay * c)
        res, matched, clash = 0.0, 0, 0
        for wx, wy in self.to_world(obs, th, x, y):
            dist = self.nearest(wx, wy)[1]
            if dist < POST_MATCH_MM:
                res += dist; matched += 1
            elif dist < POST_NEW_MM:
                clash += 1                 # neither on a map post nor clear of them: speaks against this pose
        return th, x, y, self._cost(obs, th, x, y), (res / matched if matched else 99.0), matched, clash

    def locate(self, obs, centre, th_pred, half_window):
        """
        Lay the seen posts over the map. `centre` is the ring centre in the
        car's coordinates (a first idea of where the car is), th_pred the
        expected heading. Every heading at which the posts roughly fall on map
        posts is tried and refined; the best one wins.
        Returns (theta_deg, x, y, residual_mm, matched, clashing, margin_mm) or None.
        (matched = seen posts lying on a map post; clashing = seen posts that are
        close to a map post but not on it. Seen posts far from every map post are
        posts the map does not know yet — they say nothing either way.)
        """
        cands = []
        steps = int(half_window // 2)
        for k in range(-steps, steps + 1):
            th = th_pred + 2.0 * k
            t = math.radians(th)
            x = -(centre[0] * math.cos(t) + centre[1] * math.sin(t))
            y = -(-centre[0] * math.sin(t) + centre[1] * math.cos(t))
            cands.append((self._cost(obs, th, x, y), th, x, y))
        sols = []
        for k in range(len(cands)):
            c = cands[k][0]
            if (k == 0 or c <= cands[k - 1][0]) and (k == len(cands) - 1 or c < cands[k + 1][0]):
                sols.append(self._refine(obs, cands[k][1], cands[k][2], cands[k][3]))
        if not sols:
            return None
        sols.sort(key=lambda s: s[3])
        best = sols[0]
        second = 99.0
        for s in sols[1:]:
            if abs(_wrap(s[0] - best[0])) > 3.0:
                second = s[3]
                break
        return best[0] % 360.0, best[1], best[2], best[4], best[5], best[6], second - best[3]

    def learn(self, world_posts):
        for wx, wy in world_posts:
            i, dist = self.nearest(wx, wy)
            if i is not None and dist < POST_MATCH_MM:
                p = self.posts[i]
                p[2] += 1
                w = 1.0 / min(p[2], 10)
                p[0] += (wx - p[0]) * w
                p[1] += (wy - p[1]) * w
            elif dist > POST_NEW_MM:
                self.posts.append([wx, wy, 1])
                self._index = None
        self._index = None


class RingLocalizer:
    """Where is the car on the ring? See the module text."""

    def __init__(self):
        self.radius = None            # radius of the outer ring (to the posts' faces), mm
        self.inner  = None            # how far the inner cluster reaches from the centre, mm
        self.map    = PostMap()
        self._theta = None            # last heading (deg, clockwise, 0 = heading at the start)
        self._beta  = None            # last bearing of the ring centre
        self._misses = 0              # looks in a row without a pose

    def save_map(self, path) -> bool:
        """write the map of the track to a file, so the next drive starts with it"""
        try:
            data = {'version': 1, 'radius': self.radius, 'inner': self.inner, 'post_radius': POST_RADIUS_MM,
                    'posts': [[round(p[0], 1), round(p[1], 1), int(p[2])] for p in self.map.posts]}
            with open(path + '.tmp', 'w') as f:
                json.dump(data, f)
            os.replace(path + '.tmp', path)
            return True
        except Exception:
            return False

    def load_map(self, path) -> bool:
        """read a map written by save_map(); False if there is none / it is unusable"""
        try:
            with open(path) as f:
                data = json.load(f)
            posts = [[float(p[0]), float(p[1]), int(p[2])] for p in data['posts']]
            radius = float(data['radius'])
            if not (RING_R_MIN <= radius <= RING_R_MAX) or len(posts) < POSE_MIN_POSTS:
                return False
            self.map = PostMap()
            self.map.posts = posts
            self.radius = radius
            self.inner = float(data['inner']) if data.get('inner') else None
            return True
        except Exception:
            return False

    def _ring_centre(self, xs, ys):
        """centre of the outer ring in the car's own coordinates -> (cx, cy, rms, n) or None"""
        fit = _fit_circle(xs, ys)
        if fit is None:
            return None
        cx, cy, r = fit
        if self.radius is None:
            # first time: learn the radius — fit, keep the points near the circle, fit again
            for _ in range(6):
                bx, by = [], []
                for x, y in zip(xs, ys):
                    if abs(math.hypot(x - cx, y - cy) - r) < 80.0:
                        bx.append(x); by.append(y)
                fit = _fit_circle(bx, by)
                if fit is None:
                    return None
                cx, cy, r = fit
            if not (RING_R_MIN <= r <= RING_R_MAX):
                return None
            radius = r
        else:
            radius = self.radius
            band = 90.0
            for _ in range(6):                      # the radius is known: only move the centre
                suu = suv = svv = bu = bv = 0.0
                n = 0
                for x, y in zip(xs, ys):
                    dx, dy = x - cx, y - cy
                    dist = math.hypot(dx, dy)
                    res = dist - radius
                    if abs(res) < band and dist > 1.0:
                        ux, uy = dx / dist, dy / dist
                        suu += ux * ux; suv += ux * uy; svv += uy * uy
                        bu += ux * res; bv += uy * res
                        n += 1
                det = suu * svv - suv * suv
                if n < 40 or abs(det) < 1e-9:
                    return None
                sx = (svv * bu - suv * bv) / det
                sy = (suu * bv - suv * bu) / det
                cx += sx; cy += sy
                band = BAND_MM
                if abs(sx) + abs(sy) < 0.05:
                    break
        se, n = 0.0, 0
        for x, y in zip(xs, ys):
            res = math.hypot(x - cx, y - cy) - radius
            if abs(res) < BAND_MM:
                se += res * res; n += 1
        if n < MIN_RING_PTS:
            return None
        rms = math.sqrt(se / n)
        if rms > MAX_RING_RMS:
            return None
        if self.radius is None:
            self.radius = radius
        return cx, cy, rms, n

    def measure(self, points, yaw_deg_s=0.0, speed_mm_s=0.0, want_pose=True, learn=True, moved_deg=0.0):
        """
        points: one LIDAR revolution as [(angle_deg, distance_mm, age_s), ...]
        (age = how long ago the point was measured). yaw_deg_s / speed_mm_s:
        how the car has been moving meanwhile (clockwise +, forward +) — the
        points are shifted to where they are NOW; 0, 0 when standing still.
        moved_deg: about how far around the ring (deg, clockwise +) the car has
        gone since the last call — helps to know where to look for the pose.

        Returns None if the ring is not seen, or a dict:
          rho, beta     distance from the ring centre (mm); bearing of the centre (deg)
          front, left, right   closest thing ahead / ahead-left / ahead-right (mm)
          pose_ok       True if the exact pose was found from the posts, then also:
          theta, x, y, phi   heading (deg, 0 = at the start), position, place around the ring (deg)
          res, posts    how well the posts fit (mm), how many were seen
        """
        xs, ys = [], []
        front = left = right = narrow = 1e9
        moving = (yaw_deg_s != 0.0 or speed_mm_s != 0.0)
        for p in points:
            a, d = p[0], p[1]
            if d < MIN_VALID_MM:
                continue
            if moving:
                age = p[2]
                r = math.radians(a - yaw_deg_s * age)
                x, y = d * math.sin(r), d * math.cos(r) - speed_mm_s * age
                d = math.hypot(x, y)
                aw = math.degrees(math.atan2(x, y))
            else:
                r = math.radians(a)
                x, y = d * math.sin(r), d * math.cos(r)
                aw = _wrap(a)
            xs.append(x); ys.append(y)
            if abs(aw) <= FRONT_CONE_DEG and d < front:
                front = d
            if abs(aw) <= NARROW_CONE_DEG and d < narrow:
                narrow = d
            if -75.0 <= aw < -FRONT_CONE_DEG / 2 and d < left:
                left = d
            if FRONT_CONE_DEG / 2 < aw <= 75.0 and d < right:
                right = d
        if len(xs) < MIN_POINTS:
            return None
        ring = self._ring_centre(xs, ys)
        if ring is None:
            return None
        cx, cy, rms, n_ring = ring
        radius = self.radius

        inner = []
        n_outside = n_lane = 0
        sectors = set()
        for x, y in zip(xs, ys):
            dist = math.hypot(x - cx, y - cy)
            if dist < radius * INNER_ZONE:
                inner.append(dist)
                if dist > radius * PLAIN_INNER_MAX:
                    n_lane += 1                  # the thing in the middle reaches far out: not a small cluster
            elif dist < radius - PLAIN_LANE_MM:
                n_lane += 1                      # something standing in the lane
            elif dist > radius + PLAIN_OUTSIDE_MM:
                n_outside += 1                   # something seen beyond the ring: the ring is open
            elif abs(dist - radius) < BAND_MM:
                sectors.add(int(math.degrees(math.atan2(x - cx, y - cy)) % 360.0 // 10))
        if len(inner) >= 15:
            inner.sort()
            reach = inner[int(0.9 * (len(inner) - 1))]
            self.inner = reach if self.inner is None else 0.5 * max(self.inner, reach) + 0.5 * self.inner

        beta = math.degrees(math.atan2(cx, cy))
        out = {'rho': math.hypot(cx, cy), 'beta': beta, 'front': front, 'left': left, 'right': right, 'narrow': narrow,
               'rms': rms, 'cx': cx, 'cy': cy, 'pose_ok': False, 'res': None, 'posts': 0,
               'outside': n_outside / len(xs), 'lane': n_lane / len(xs), 'sectors': len(sectors)}

        # the heading to expect: the bearing of the ring centre changes by the car's own
        # turning minus its way around the ring, so heading = last - change of bearing + way
        predicted = None if self._theta is None else self._theta - _wrap(beta - self._beta) + moved_deg

        if want_pose:
            obs = find_posts(xs, ys, cx, cy, radius)
            out['posts'] = len(obs)
            if len(obs) >= POSE_MIN_POSTS:
                if len(self.map) == 0:
                    # the very first look defines the map: heading 0, origin = ring centre
                    self.map.learn(PostMap.to_world(obs, 0.0, -cx, -cy))
                    out.update(pose_ok=True, theta=0.0, x=-cx, y=-cy, res=0.0)
                else:
                    need = POSE_MIN_MARGIN
                    if predicted is None or (self._misses >= POSE_LOST_LOOKS and not moving):
                        sol = self.map.locate(obs, (cx, cy), predicted or 0.0, 180)     # all the way around
                        need = POSE_GLOBAL_MARGIN
                    else:
                        sol = self.map.locate(obs, (cx, cy), predicted, POSE_WINDOW_DEG)
                    if sol is not None:
                        th, x, y, res, matched, clash, margin = sol
                        out['res'] = res
                        if (res <= POSE_MAX_RES_MM and matched >= POSE_MIN_POSTS
                                and matched >= POSE_MIN_SHARE * (matched + clash) and margin >= need):
                            out.update(pose_ok=True, theta=th, x=x, y=y)
                            if learn:
                                self.map.learn(PostMap.to_world(obs, th, x, y))

        if out['pose_ok']:
            self._theta = out['theta']
            self._misses = 0
            out['phi'] = math.degrees(math.atan2(out['x'], out['y'])) % 360.0
        elif predicted is not None:
            self._theta = predicted % 360.0
            if want_pose:
                self._misses += 1
        if self._theta is not None:
            self._beta = beta
        return out


class RingDriver:
    """Drives the car around the ring. See the module text."""

    def __init__(self, motors, lidar, log=print, on_event=None):
        self._motors = motors
        self._lidar  = lidar
        self._log    = log
        self._event  = on_event
        self.loc     = RingLocalizer()
        self._turn_gain = 1.0          # learned: how much longer/shorter turn pulses must be
        self._step_mm   = 50.0         # learned: how far one forward pulse moves the car

    # ── helpers ───────────────────────────────────────────────

    def _sleep(self, seconds, running) -> bool:
        end = time.time() + seconds
        while running():
            left = end - time.time()
            if left <= 0:
                return True
            time.sleep(min(0.02, left))
        return False

    def _look(self, running, tries=6, moved_deg=0.0):
        """A measurement taken while the car really stands still, or None.
        moved_deg: about how far around the ring the car went since the last look."""
        last = None
        for _ in range(tries):
            if not running():
                return None
            m = self.loc.measure(self._lidar.get_last_revolution(), moved_deg=moved_deg)
            moved_deg = 0.0
            if m is not None and last is not None and \
                    math.hypot(m['cx'] - last['cx'], m['cy'] - last['cy']) <= STILL_MM:
                return m
            last = m
            if not self._sleep(RECHECK_S, running):
                return None
        return last

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

    def _say(self, text, quiet=False):
        if not quiet:
            self._log(f"[RingDrive] {text}")
        if self._event:
            try:
                self._event(f"NAV {text}")
            except Exception:
                pass

    def _steer(self, u):
        """
        Forward, steering by u: 0 = straight, +1 = as hard as allowed toward
        the ring centre, -1 = away from it. The side on the inside of the
        turn is slowed down (to CRUISE_SLOW_SIDE_MIN at the most).
        """
        u = max(-1.0, min(1.0, u))
        fast = CRUISE_SPEED
        slow = fast * (1.0 - abs(u) * (1.0 - CRUISE_SLOW_SIDE_MIN / 100.0))
        turn_right = (u > 0) == self._clockwise          # toward the centre = right when clockwise
        self._motors._set_direction(False, True, False, True)
        if turn_right:
            self._motors._set_speed(slow, fast)          # (right, left)
        else:
            self._motors._set_speed(fast, slow)

    # ── geometry of the route ─────────────────────────────────

    def _aim(self, m, remaining, k_rho, eps_max=EPS_MAX_DEG):
        """(lane target mm, eps = pointing outward deg, want deg, err deg)"""
        target = self._lane
        if remaining < BLEND_DEG:              # end of the route: come back to the start point's radius
            share = (remaining - BLEND_END_DEG) / (BLEND_DEG - BLEND_END_DEG)
            target = self._start_rho + (self._lane - self._start_rho) * max(0.0, min(1.0, share))
        target = max(self._rho_lo, min(self._rho_hi, target))
        d = 1.0 if self._clockwise else -1.0
        eps = _wrap(d * (m['beta'] - d * 90.0))    # + = pointing outward, - = inward (0 = along the lane)
        want = max(-eps_max, min(eps_max, k_rho * (target - m['rho'])))
        return target, eps, want, eps - want

    def _advance(self, m, guess_deg):
        """count the way around the ring: from the pose if there is one, else by the guess"""
        d = 1.0 if self._clockwise else -1.0
        if m['pose_ok']:
            step = d * _wrap(m['phi'] - self._last_phi)
            self._last_phi = m['phi']
        else:
            step = guess_deg
            self._last_phi = (self._last_phi + d * step) % 360.0
        self._done += step

    # ── CRUISE ────────────────────────────────────────────────

    def _cruise(self, running) -> str:
        """
        Drive without stopping until something ends it. Returns why:
        'finish' (close to the end of the route), 'blocked', 'edge',
        'heading', 'lost' or 'cancelled'. The car is stopped on return.

        About 6 times a second: look, work out how fast the car SHOULD be
        turning (the curve of the lane, plus a bit to remove the heading
        error), compare with how fast it IS turning (seen from how the ring
        centre moves across the view), and move the steering accordingly.
        """
        loc = self.loc
        sign = 1.0 if self._clockwise else -1.0
        t_on = time.time()
        u, u_i = 0.0, 0.0                    # steering now; its slowly learned part
        yaw_in = 0.0                         # measured turn rate toward the centre (deg/s)
        speed = 0.0                          # estimated speed (mm/s)
        beta_prev, pose_prev = None, None
        self._steer(0.0)                     # start straight: both sides must get rolling first
        t_prev = t_on
        lost, n, reason = 0, 0, 'cancelled'
        self._say("CRUISE on")
        while running():
            if not self._sleep(max(0.02, loc_period(self._lidar) - (time.time() - t_prev)), running):
                break
            now = time.time()
            dt, t_prev = now - t_prev, now
            n += 1
            # speed: builds up from standing; corrected below whenever two poses are close in time
            speed = min(CRUISE_SPEED_MM_S, speed + CRUISE_ACCEL_MM_S2 * dt * (1.0 - 0.4 * abs(u)))
            way = math.degrees(speed * dt / max(self._last_rho, 1.0))        # expected way around the ring
            m = loc.measure(self._lidar.get_last_revolution(), sign * yaw_in, speed,
                            want_pose=(n % CRUISE_POSE_EVERY == 0), learn=False, moved_deg=sign * way)
            if m is None:
                lost += 1
                if lost >= 2:
                    reason = 'lost'
                    break
                continue
            lost = 0
            self._last_rho = m['rho']
            self._advance(m, way)
            if m['pose_ok']:
                if pose_prev is not None and 0.05 < now - pose_prev[0] < 0.6:
                    seen = math.hypot(m['x'] - pose_prev[1], m['y'] - pose_prev[2]) / (now - pose_prev[0])
                    if seen < 700.0:
                        speed = 0.5 * speed + 0.5 * seen
                pose_prev = (now, m['x'], m['y'])

            # how fast is the car really turning? heading = place on the ring - bearing of the centre,
            # so turn rate = (rate of going around) - (rate of change of that bearing)
            if beta_prev is not None and dt > 0.05:
                rate = math.degrees(speed / max(m['rho'], 1.0)) - sign * _wrap(m['beta'] - beta_prev) / dt
                if -200.0 < rate < 250.0:
                    yaw_in = 0.5 * yaw_in + 0.5 * rate
            beta_prev = m['beta']

            remaining = self._goal - self._done
            # the look is already a little old and the car keeps moving: judge the lane
            # position by where it is heading (distance from centre grows when pointing outward)
            _, eps_now, _, _ = self._aim(m, remaining, CRUISE_K_RHO, CRUISE_EPS_MAX_DEG)
            ahead = dict(m)
            ahead['rho'] = m['rho'] + CRUISE_LOOKAHEAD_S * speed * math.sin(math.radians(eps_now))
            target, eps, want, err = self._aim(ahead, remaining, CRUISE_K_RHO, CRUISE_EPS_MAX_DEG)

            if remaining <= CRUISE_EXIT_DEG:
                reason = 'finish'
            elif m['narrow'] < CRUISE_FRONT_MM or m['front'] < CRUISE_SIDE_FRONT_MM:
                reason = 'blocked'
            elif ahead['rho'] < self._rho_lo + CRUISE_EDGE_MM or ahead['rho'] > self._rho_hi - CRUISE_EDGE_MM:
                reason = 'edge'
            elif abs(err) > CRUISE_MAX_ERR_DEG:
                reason = 'heading'
            else:
                reason = None
            if reason:
                break

            # wanted turn rate toward the centre: the curve of the lane at this speed + heading correction
            want_yaw = math.degrees(speed / max(m['rho'], 1.0)) + CRUISE_K_HEADING * err
            want_yaw = max(CRUISE_YAW_MIN, min(CRUISE_YAW_MAX, want_yaw))
            limit = max(0.0, min(1.0, (now - t_on - CRUISE_STRAIGHT_S) / CRUISE_RAMP_S))
            miss = want_yaw - yaw_in
            if abs(u) < limit or miss * u < 0:        # don't wind up while the steering is at its limit
                u_i += CRUISE_KI * miss * dt
            u_i = max(-0.6, min(1.0, u_i))
            u = 0.5 * want_yaw / CRUISE_YAW_GUESS + u_i + CRUISE_KP * miss
            u = max(-0.6, min(limit, u))
            self._steer(u)
            self._say(f"cruise lap {self._done / 360.0:.2f}/{self._laps} | {m['rho']:.0f}mm from centre (want {target:.0f}) | "
                      f"pointing {eps:+.0f}deg (want {want:+.0f}) | turning {yaw_in:+.0f}deg/s (want {want_yaw:+.0f}) | "
                      f"steer {u:+.2f} | {speed:.0f}mm/s | ahead {m['narrow']:.0f}mm"
                      + ("" if m['pose_ok'] or n % CRUISE_POSE_EVERY else " | no pose"), quiet=(n % 3 != 0))
        self._motors.stop()
        self._say(f"CRUISE off: {reason}")
        if reason != 'cancelled' and not self._sleep(SETTLE_S, running):
            return 'cancelled'
        return reason

    # ── the drive ─────────────────────────────────────────────

    def drive(self, running, clockwise: bool = True, laps: int = 1) -> str:
        """
        Drive `laps` laps and stop on the start point.
        running: function returning False when the drive must be cancelled.
        Returns 'done', 'cancelled', 'no_ring' (the LIDAR does not see the
        ring track here — nothing was driven) or 'stuck'.
        """
        self._motors.stop()
        self._clockwise, self._laps = clockwise, laps
        d = 1.0 if clockwise else -1.0            # clockwise: the ring centre is on the car's RIGHT

        # hands off the car, LIDAR spinning (its motor was only just started), then a first steady look
        if not self._sleep(START_DELAY_S, running):
            return 'cancelled'
        # is this the plain ring track at all? two looks in a row must both say so
        plain, m0 = 0, None
        for _ in range(8):
            m0 = RingLocalizer().measure(self._lidar.get_last_revolution(), want_pose=False)
            if (m0 is not None and m0['outside'] <= PLAIN_MAX_OUTSIDE and m0['lane'] <= PLAIN_MAX_LANE
                    and m0['sectors'] >= PLAIN_MIN_SECTORS):
                plain += 1
                if plain >= 2:
                    break
            else:
                plain = 0
            if not self._sleep(0.2, running):
                return 'cancelled'
        if plain < 2:
            self._say("this is not the plain ring track"
                      + ("" if m0 is None else f" ({100 * m0['outside']:.0f}% of the LIDAR points lie beyond the ring, "
                                               f"{100 * m0['lane']:.0f}% in the lane, ring seen in {m0['sectors']}/36 directions)"))
            return 'no_ring'

        map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MAP_FILE)
        self.loc = RingLocalizer()
        from_file = self.loc.load_map(map_path)    # the track as mapped on earlier drives
        m, deadline, tries = None, time.time() + 6.0, 0
        while running() and time.time() < deadline and (m is None or not m['pose_ok']):
            m = self._look(running, tries=3)
            if m is None or not m['pose_ok']:
                tries += 1
                if from_file and tries < 4:
                    self.loc._theta = None         # look again against the saved map
                else:
                    if from_file:
                        self._say("this does not look like the saved map of the track: mapping it again")
                    from_file = False
                    self.loc = RingLocalizer()     # nothing usable yet: start clean on the next try
                if not self._sleep(0.25, running):
                    return 'cancelled'
        if not running():
            return 'cancelled'
        if m is None or not m['pose_ok']:
            self._say("the LIDAR does not see the ring track from here")
            return 'no_ring'

        loc = self.loc
        inner = loc.inner if loc.inner is not None else loc.radius * 0.28
        self._lane = (inner + loc.radius) / 2.0
        self._rho_lo, self._rho_hi = inner + WALL_MARGIN_MM, loc.radius - WALL_MARGIN_MM
        # where to finish: the start point — but not closer to a wall than the car may drive
        self._start_rho = max(self._rho_lo + 30.0, min(self._rho_hi - 40.0, m['rho']))
        start_phi = m['phi']
        self._goal = 360.0 * laps
        self._done = 0.0                           # how far around the ring so far (in driving direction)
        self._last_phi = start_phi
        self._last_rho = m['rho']
        start_theta = m['theta']
        mapping = not loc.map.complete()           # the track is not fully on the map yet
        self._say(f"ring: radius {loc.radius:.0f}mm, {m['posts']} posts in view, inner cluster reaches {inner:.0f}mm, "
                  f"lane centre {self._lane:.0f}mm | start: {m['rho']:.0f}mm from the centre | "
                  f"{'clockwise' if clockwise else 'counter-clockwise'}, {laps} lap(s) | "
                  + (f"map of the track: {len(loc.map)} posts" + (" (from file)" if from_file else "")))
        if mapping:
            self._say("the track is not fully mapped yet: driving in STEPS until it is (about one lap), then CRUISE")

        last = m
        last_action = None
        last_pulse = last_want = 0.0
        no_move = 0
        turns_in_row = 0
        step = 0
        while running():
            step += 1
            if step > MAX_STEPS_PER_LAP * laps:
                self._say("too many steps, stopping")
                return 'stuck'

            # ── where am I now? ──
            if step > 1:
                guess = math.degrees(self._step_mm / max(last['rho'], 1.0)) if last_action == 'F' else 0.0
                m = self._look(running, moved_deg=d * guess)
                if m is None:
                    if not running():
                        return 'cancelled'
                    self._say("lost sight of the ring, stopping")
                    return 'stuck'
                self._advance(m, guess)
                # learn from what the last pulse really did
                if last_action == 'F' and m['pose_ok'] and last['pose_ok']:
                    moved = math.hypot(m['x'] - last['x'], m['y'] - last['y'])
                    no_move = no_move + 1 if moved < 8.0 else 0
                    if moved >= 8.0 and last_pulse >= FORWARD_PULSE_S - 0.01:
                        self._step_mm = 0.7 * self._step_mm + 0.3 * moved
                elif last_action in ('L', 'R') and last_want > 3.0:
                    turned = abs(_wrap(m['theta'] - last['theta'])) if (m['pose_ok'] and last['pose_ok']) \
                        else abs(_wrap(m['beta'] - last['beta']))
                    ratio = max(0.5, min(2.0, last_want / max(turned, 1.0)))
                    self._turn_gain = max(0.3, min(2.0, self._turn_gain * (0.5 + 0.5 * ratio)))   # (this car turned 2.5x faster on some days)

            if mapping and loc.map.complete():
                mapping = False
                saved = loc.save_map(map_path)
                self._say(f"the track is mapped: {len(loc.map)} posts" + (" (saved)" if saved else ""))

            remaining = self._goal - self._done
            arc_left = math.radians(max(remaining, 0.0)) * m['rho']
            target, eps, want, err = self._aim(m, remaining, K_RHO_DEG_PER_MM)

            status = (f"#{step} lap {self._done / 360.0:.2f}/{laps} | {m['rho']:.0f}mm from centre (want {target:.0f}) | "
                      f"pointing {eps:+.0f}deg (want {want:+.0f}) | ahead {m['front']:.0f}mm"
                      + ("" if m['pose_ok'] else " | place on the ring estimated"))

            # ── arrived? ──
            if remaining <= FINISH_TOL_DEG:
                self._say(status + " -> at the start point")
                break

            # ── CRUISE when everything is in order ──
            inside = self._rho_lo + CRUISE_ENTER_EDGE_MM <= m['rho'] <= self._rho_hi - CRUISE_ENTER_EDGE_MM
            cruise_next = (CRUISE and not mapping and remaining > CRUISE_EXIT_DEG + 15.0 and inside and no_move == 0
                           and m['narrow'] >= CRUISE_ENTER_FRONT_MM and m['front'] >= FRONT_CLEAR_MM)
            if cruise_next:
                err += CRUISE_ENTER_LEAD_DEG       # line up for the cruise: a bit further toward the centre
            if cruise_next and abs(err) <= CRUISE_ENTER_ERR_DEG:
                self._say(status + " -> cruise")
                why = self._cruise(running)
                if why == 'cancelled':
                    return 'cancelled'
                last_action = None
                turns_in_row = 0
                last = m
                continue

            # ── one step ──
            blocked = m['narrow'] < FRONT_CLEAR_MM or m['front'] < SIDE_CLEAR_MM
            if no_move >= STUCK_STEPS:
                self._say(status + " -> not moving: backing off")
                action, pulse, last_want = 'B', FORWARD_PULSE_S, 0.0
                no_move = 0
            elif turns_in_row >= MAX_TURNS_IN_ROW and m['narrow'] >= SIDE_CLEAR_MM:
                action, pulse, last_want = 'F', 0.6 * FORWARD_PULSE_S, 0.0
                self._say(status + f" -> forward {pulse:.2f}s (enough turning)")
            elif abs(err) > (CRUISE_ENTER_ERR_DEG if cruise_next else TURN_TOL_DEG) or blocked:
                if blocked and abs(err) <= TURN_TOL_DEG:
                    # something ahead although the heading is right: turn toward the freer side
                    turn_inward = (m['right'] > m['left']) if clockwise else (m['left'] > m['right'])
                    angle = 15.0
                else:
                    turn_inward = err > 0
                    angle = TURN_SHARE * abs(err)
                # inward = toward the ring centre = RIGHT when driving clockwise
                action = ('R' if turn_inward else 'L') if clockwise else ('L' if turn_inward else 'R')
                angle = min(angle, 40.0)
                pulse = self._turn_gain * (angle + TURN_DEAD_DEG) / TURN_RATE_DEG_S
                pulse = max(TURN_MIN_S, min(TURN_MAX_S, pulse))
                last_want = angle
                self._say(status + f" -> turn '{action}' {pulse:.2f}s" + (" (blocked ahead)" if blocked else ""))
            else:
                pulse = FORWARD_PULSE_S
                if arc_left < FINISH_SLOW_MM:      # close to the finish: shorter steps
                    pulse = max(0.08, min(FORWARD_PULSE_S, FORWARD_PULSE_S * arc_left / max(self._step_mm, 20.0)))
                action, last_want = 'F', 0.0
                self._say(status + f" -> forward {pulse:.2f}s")

            turns_in_row = turns_in_row + 1 if action in ('L', 'R') else 0
            last, last_action, last_pulse = m, action, pulse
            if not self._pulse(action, pulse, running):
                return 'cancelled'

        if not running():
            return 'cancelled'

        # ── on the start point: turn back to the starting direction ──
        for _ in range(5):
            m = self._look(running)
            if m is None or not m['pose_ok']:
                break
            off = _wrap(m['theta'] - start_theta)
            if abs(off) <= FINISH_HEADING_TOL_DEG:
                break
            action = 'L' if off > 0 else 'R'
            pulse = max(TURN_MIN_S, min(TURN_MAX_S, self._turn_gain * (TURN_SHARE * abs(off) + TURN_DEAD_DEG) / TURN_RATE_DEG_S))
            self._say(f"finish: turned {off:+.0f}deg against the start direction -> turn '{action}' {pulse:.2f}s")
            if not self._pulse(action, pulse, running):
                return 'cancelled'
        if m is not None and m['pose_ok']:
            self._say(f"FINISHED: {laps} lap(s). Standing {m['rho']:.0f}mm from the centre (start was {self._start_rho:.0f}), "
                      f"{_wrap(m['phi'] - start_phi):+.1f}deg around the ring from the start point, "
                      f"heading {_wrap(m['theta'] - start_theta):+.0f}deg from the start direction | map: {len(loc.map)} posts")
            if loc.map.complete():
                loc.save_map(map_path)
        else:
            self._say(f"FINISHED: {laps} lap(s) (final position not confirmed)")
        return 'done'


def loc_period(lidar) -> float:
    """seconds per LIDAR revolution (how often a new look is available)"""
    period = getattr(lidar, 'rev_time', None)
    return period if period and 0.08 <= period <= 0.4 else 1.0 / 6.0
