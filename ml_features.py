"""
ml_features.py
==============
ONE LIDAR revolution  ->  the list of numbers the neural network gets as input.

This file is shared by ALL the ML parts, on purpose:
    ml_dataset.py   (building the training data from the recorded scans/)
    ml_driver.py    (live driving on the car)
If training and driving computed the input even slightly differently, the
network would see on the road something different from what it learned on.

What it does (PDF sections 1, 6):

  The PDF talks about 4 distances  L, F, R, B  (left, front, right, rear).
  Our D200 is a 360-degree LIDAR (about 700 points per revolution), so we can
  give the network more directions than 4. The circle around the car is cut
  into N_SECTORS equal "slices" (sectors):

        sector 0 = straight ahead, then clockwise
        N_SECTORS = 4  -> exactly the PDF:  F, R, B, L
        N_SECTORS = 16 -> one value every 22.5 degrees (our default)

  For every sector we take the distance to the NEAREST thing in it
  (a "robust minimum": the 10th percentile, so one bad point can't fool it),
  then CLEAN it (PDF section 6):
        - points closer than MIN_VALID_MM are the car itself -> thrown away
        - distance = max(0, min(distance, MAX_MM))
        - a sector with no points at all = "nothing there" = MAX_MM
  and NORMALIZE it:   value = distance / MAX_MM      (0.0 .. 1.0)

  Example (MAX_MM = 1000):  a wall 240 mm in front  ->  0.24
"""

import math

import numpy as np

# ── Settings (change them here; ml_dataset.py and ml_driver.py both use them) ──
N_SECTORS    = 16       # how many directions the network sees (4 = exactly as in the PDF)
MAX_MM       = 1000.0   # distances are clipped to this and divided by it (PDF: 500 cm; our track is small)
MIN_VALID_MM = 80.0     # closer than this = the car itself / an invalid reading
PERCENTILE   = 10       # "nearest" = this percentile of the sector's distances (robust minimum)


def sector_of(angle_deg: float, n: int = N_SECTORS) -> int:
    """Which sector an angle falls in. Sector 0 is centred straight ahead."""
    width = 360.0 / n
    return int(((angle_deg + width / 2.0) % 360.0) // width) % n


def sector_names(n: int = N_SECTORS) -> list:
    """Readable names for the CSV header: s00_0deg, s01_22deg, ..."""
    return [f"s{i:02d}_{int(round(i * 360.0 / n))}deg" for i in range(n)]


def features_from_points(points, n: int = N_SECTORS, max_mm: float = MAX_MM,
                         percentile: float = None, min_valid_mm: float = None) -> list:
    """
    points: [(angle_deg, distance_mm, ...), ...]  (anything after the first two is ignored)
            angle 0 = straight ahead, clockwise — exactly what lidar_driver.py gives.
    returns: list of n numbers in 0.0 .. 1.0 (1.0 = far / nothing seen)
    percentile / min_valid_mm: None = the settings above (the driver passes the
    values the model was trained with, so they always match).
    """
    if not points:
        return [1.0] * n
    a = np.fromiter((p[0] for p in points), dtype=np.float64, count=len(points))
    d = np.fromiter((p[1] for p in points), dtype=np.float64, count=len(points))
    return features_from_arrays(a, d, n, max_mm, percentile, min_valid_mm)


def features_from_arrays(angles, dists, n: int = N_SECTORS, max_mm: float = MAX_MM,
                         percentile: float = None, min_valid_mm: float = None) -> list:
    """The same as features_from_points(), for NumPy arrays of angles (deg) and distances (mm)."""
    pct = PERCENTILE if percentile is None else percentile
    min_valid = MIN_VALID_MM if min_valid_mm is None else min_valid_mm
    keep = dists >= min_valid
    a, d = angles[keep], np.minimum(dists[keep], max_mm)
    width = 360.0 / n
    sec = (((a + width / 2.0) % 360.0) // width).astype(np.int64) % n
    order = np.lexsort((d, sec))                        # by sector, then nearest first
    d_sorted = d[order]
    counts = np.bincount(sec, minlength=n)
    starts = np.cumsum(counts) - counts
    k = (counts * pct / 100.0).astype(np.int64)         # the PERCENTILE-th nearest point of each sector
    out = []
    for i in range(n):
        if counts[i] == 0:
            out.append(1.0)                             # nothing seen = far away
        else:
            v = d_sorted[starts[i] + min(k[i], counts[i] - 1)]
            out.append(float(max(0.0, min(v, max_mm)) / max_mm))
    return out


def shift_points(angles, dists, rotate_deg: float, shift_right_mm: float):
    """
    The LIDAR picture as it would look if the car stood `shift_right_mm` further to
    the right and turned `rotate_deg` further clockwise (used for recovery
    augmentation in ml_dataset.py). Returns (angles, dists) as NumPy arrays.
    """
    r = np.radians(angles)
    x = dists * np.sin(r) - shift_right_mm
    y = dists * np.cos(r)
    return (np.degrees(np.arctan2(x, y)) - rotate_deg) % 360.0, np.hypot(x, y)


def mirror_features(vec, n: int = None) -> list:
    """
    The same picture seen in a mirror (left <-> right). Sector k becomes
    sector (n - k) % n. Used to double the training data: a left turn in the
    mirror is a right turn (see ml_dataset.MIRROR_ACTION).
    """
    n = n or len(vec)
    return [vec[(n - k) % n] for k in range(n)]


def front_distance_mm(points, cone_deg: float = 20.0) -> float:
    """Nearest thing within +/- cone_deg of straight ahead (mm). For the safety layer."""
    best = float('inf')
    for p in points:
        a = (p[0] + 180.0) % 360.0 - 180.0
        if abs(a) <= cone_deg and p[1] >= MIN_VALID_MM:
            best = min(best, p[1])
    return best


def distance_in_cone_mm(points, centre_deg: float, cone_deg: float) -> float:
    """Nearest thing within +/- cone_deg around centre_deg (0 = ahead, 90 = right ...)."""
    best = float('inf')
    for p in points:
        a = (p[0] - centre_deg + 180.0) % 360.0 - 180.0
        if abs(a) <= cone_deg and p[1] >= MIN_VALID_MM:
            best = min(best, p[1])
    return best
