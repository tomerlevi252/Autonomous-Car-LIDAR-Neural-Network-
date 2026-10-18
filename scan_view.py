"""
scan_view.py
============
Turns a LIDAR log from the folder  scans/  into a PICTURE: the walls of the
track as the LIDAR saw them, and the way the car drove.

How to use (in the car_ml folder):

    python3 scan_view.py                 the newest file in scans/
    python3 scan_view.py all             every file in scans/ that has no picture yet
    python3 scan_view.py scans/scan_RUN_20261006_110103.csv.gz      one file

The picture is saved next to the scan file, with the same name and  .png
(scans/scan_RUN_20261006_110103.png). Copy it to a computer/phone like the
scan files, and open it there.

In the picture:  dark dots = walls (posts);  the coloured line = the way the
car drove (each lap a different colour);  green dot = start;  red square =
end;  the bar at the bottom shows 50 cm.

It needs only Pillow (the same library the screen driver uses) and track_nav.py.
It does not touch the car: it only reads the file.
"""

import glob
import gzip
import math
import os
import sys
import time

from PIL import Image, ImageDraw, ImageFont

import track_nav as tn

SCAN_DIR   = "scans"
SIZE_PX    = 1100          # the picture is this many pixels wide and high
REFIND_SHARE = 0.85        # a look fitting worse than this is searched for again, much wider
MIN_HITS   = 2             # a spot is drawn as "wall" when the LIDAR hit it at least this often
LAP_COLOURS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (148, 103, 189), (140, 86, 75), (23, 190, 207)]


# ── reading the file ───────────────────────────────────────────

def read_scan(path):
    """-> (points [(time, angle_deg, dist_mm)], events [(time, text)], header text)"""
    packets, events, header = [], [], []
    last, base = None, 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("P,"):
                a = line.rstrip().split(",")
                try:
                    ts = int(a[3]); v = a[4:]
                    pts = [(float(v[i]), int(v[i + 1])) for i in range(0, len(v) - 2, 3)]
                    t_pi = float(a[1])
                except ValueError:
                    continue
                if last is not None and ts < last - 15000:
                    base += 30000                      # the LIDAR's own clock starts again every 30 s
                last = ts
                packets.append((t_pi, (base + ts) / 1000.0, pts))
            elif line.startswith("E,"):
                p = line.rstrip().split(",", 2)
                try:
                    events.append((float(p[1]), p[2] if len(p) > 2 else ""))
                except ValueError:
                    pass
            elif line.startswith("#"):
                header.append(line[1:].strip())
    if not packets:
        return [], events, " | ".join(header)
    # the LIDAR's clock says WHEN each packet was measured much better than its arrival time does
    off = min(p[0] - p[1] for p in packets)
    points = []
    for _, t_lidar, pts in packets:
        t = t_lidar + off
        n = len(pts)
        for k, (ang, dist) in enumerate(pts):
            points.append((t - (n - 1 - k) * 0.00025, ang, dist))
    return points, events, " | ".join(header)


class _Looks:
    """hands out "the LIDAR revolution that ended at time T" """

    def __init__(self, points):
        self.p = points
        self.times = [q[0] for q in points]

    def at(self, t_end, length=1.0 / 6.0):
        import bisect
        i0 = bisect.bisect_left(self.times, t_end - length)
        i1 = bisect.bisect_right(self.times, t_end)
        return [(a, d, t_end - t) for t, a, d in self.p[i0:i1]]


# ── building the map ───────────────────────────────────────────

class _Map:
    """all looks drawn into one grid, so that each new look can be laid over everything seen so far"""

    def __init__(self):
        self.grid = None
        self.cloud = []            # (x, y) of wall points, in the start frame

    def add(self, xs, ys, th, x, y):
        t = math.radians(th)
        c, s = math.cos(t), math.sin(t)
        wx = [x + px * c + py * s for px, py in zip(xs, ys)]
        wy = [y - px * s + py * c for px, py in zip(xs, ys)]
        if self.grid is None:
            self.grid = tn.ScanGrid(wx, wy)
        else:
            g = self.grid
            n, inv, half = g.n, 1.0 / g.cell, tn.GRID_HALF_MM
            kernel = tn.ScanGrid._kernels[(tn.GRID_CELL_MM, tn.GRID_REACH_MM)]
            lo, hi = 4, n - 5
            for px, py in zip(wx[::2], wy[::2]):
                ix = int((px + half) * inv); iy = int((py + half) * inv)
                if lo <= ix <= hi and lo <= iy <= hi:
                    base = iy * n + ix
                    for dx, dy, v in kernel:
                        k = base + dy * n + dx
                        if g.g[k] < v:
                            g.g[k] = v
        self.cloud += list(zip(wx, wy))

    def place(self, xs, ys, th, x, y, turns, moves):
        """the best pose for this look near the guess (th, x, y); turns/moves = what else to try"""
        sx, sy = tn.thin(xs, ys)
        t = math.radians(th)
        ux, uy = math.sin(t), math.cos(t)
        cands = [(self.grid.score(sx, sy, th + a, x + d * ux, y + d * uy), th + a, x + d * ux, y + d * uy)
                 for d in moves for a in turns]
        cands.sort(reverse=True)
        best, done = None, []
        for _, cth, cx, cy in cands:
            if any(abs(cth - q[0]) < 6.0 and abs(cx - q[1]) < 30.0 and abs(cy - q[2]) < 30.0 for q in done):
                continue
            done.append((cth, cx, cy))
            sol = self.grid.fit(sx, sy, cth, cx, cy)
            if best is None or sol[3] > best[3]:
                best = sol
            if len(done) >= 3:
                break
        return best[0], best[1], best[2], self.grid.share(sx, sy, best[0], best[1], best[2])


def build_from_scans(looks, events, t0, t1, progress=None):
    """work out the map and the car's way from the scans alone (any drive: by hand or autonomous)"""
    m, path = _Map(), []
    rev = 1.0 / 6.0
    th = x = y = 0.0
    yaw = speed = 0.0
    turned_total, laps, away, prev_y = 0.0, 0, False, None
    start_th = 0.0
    t, n = t0 + 2 * rev, 0
    while t < t1:
        scan = looks.at(t, rev)
        t += rev
        xs, ys = tn.to_xy(scan, yaw, speed)
        if len(xs) < 250:
            continue
        n += 1
        if m.grid is None:
            m.add(xs, ys, 0.0, 0.0, 0.0)
            path.append((0.0, 0.0, 0.0, 0))
            continue
        gth = th + yaw * rev
        r = math.radians(th)
        gx, gy = x + speed * rev * math.sin(r), y + speed * rev * math.cos(r)
        nth, nx, ny, share = m.place(xs, ys, gth, gx, gy, (-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0),
                                     [-44.0 + 22.0 * k for k in range(10)])
        if share < REFIND_SHARE:
            # the car turned or jumped more than expected (a fast turn on the spot): look much wider,
            # around where it was last known for sure
            wth, wx, wy, wshare = m.place(xs, ys, th, x, y, [4.0 * k for k in range(-22, 23)],
                                          [-66.0 + 22.0 * k for k in range(14)])
            if wshare > share + 0.03:
                nth, nx, ny, share = wth, wx, wy, wshare
        if share >= tn.MATCH_MIN_SHARE:
            r = math.radians(nth)
            fwd = (nx - x) * math.sin(r) + (ny - y) * math.cos(r)
            yaw = 0.5 * yaw + 0.5 * (nth - th) / rev
            speed = 0.5 * speed + 0.5 * fwd / rev
            if abs(yaw) > 320.0 or abs(speed) > 1000.0:
                yaw = speed = 0.0
            th, x, y = nth, nx, ny
            if n % 2 == 0 or share < 0.9:
                m.add(xs, ys, th, x, y)
        else:
            th, x, y = gth, gx, gy                      # not trusted: go on with the guess
        # count laps: around once, and over the start line again
        dist = math.hypot(x, y)
        if dist > tn.AWAY_MM:
            away = True
        if away and dist < tn.NEAR_MM and abs(th) > 360.0 * (laps + 1) - tn.GATE_TURN_DEG \
                and prev_y is not None and prev_y < 0.0 <= y and abs(tn._wrap(th)) < 90.0:
            laps += 1
            away = False
        prev_y = y
        path.append((x, y, th, laps))
        if progress and n % 60 == 0:
            progress(n)
    return m, path


# ── drawing ────────────────────────────────────────────────────

def _font(size):
    try:
        return ImageFont.load_default(size=size)       # newer Pillow
    except TypeError:
        return ImageFont.load_default()


def draw(m, path, title, lines, out_path):
    if not m.cloud:
        return False
    xs = [p[0] for p in m.cloud] + [p[0] for p in path]
    ys = [p[1] for p in m.cloud] + [p[1] for p in path]
    xs.sort(); ys.sort()
    k = max(0, int(0.002 * len(xs)))                    # ignore the few farthest stray points
    x0, x1, y0, y1 = xs[k], xs[-1 - k], ys[k], ys[-1 - k]
    top, margin = 90, 50
    scale = min((SIZE_PX - 2 * margin) / max(x1 - x0, 1.0), (SIZE_PX - top - 2 * margin) / max(y1 - y0, 1.0))
    ox = margin + ((SIZE_PX - 2 * margin) - (x1 - x0) * scale) / 2.0
    oy = top + margin + ((SIZE_PX - top - 2 * margin) - (y1 - y0) * scale) / 2.0

    def px(x, y):                                       # y = "ahead at the start" points UP in the picture
        return ox + (x - x0) * scale, oy + (y1 - y) * scale

    img = Image.new("RGB", (SIZE_PX, SIZE_PX), (255, 255, 255))
    d = ImageDraw.Draw(img)
    # walls: count the hits per pixel, draw the pixels that were hit often enough
    hits = {}
    for wx, wy in m.cloud:
        u, v = px(wx, wy)
        key = (int(u) // 2, int(v) // 2)
        hits[key] = hits.get(key, 0) + 1
    for (u, v), cnt in hits.items():
        if cnt >= MIN_HITS:
            shade = 40 if cnt >= 3 * MIN_HITS else 120
            d.rectangle([2 * u, 2 * v, 2 * u + 2, 2 * v + 2], fill=(shade, shade, shade))
    # the way driven, one colour per lap
    for a, b in zip(path, path[1:]):
        d.line([px(a[0], a[1]), px(b[0], b[1])], fill=LAP_COLOURS[b[3] % len(LAP_COLOURS)], width=4)
    for i in range(0, len(path), max(1, len(path) // 40)):      # small arrows: which way the car pointed
        x, y, th, lap = path[i]
        r = math.radians(th)
        u, v = px(x, y)
        d.line([(u, v), (u + 16 * math.sin(r), v - 16 * math.cos(r))], fill=(0, 0, 0), width=2)
    if path:
        u, v = px(path[0][0], path[0][1])
        d.ellipse([u - 11, v - 11, u + 11, v + 11], fill=(0, 170, 0), outline=(0, 0, 0))
        u, v = px(path[-1][0], path[-1][1])
        d.rectangle([u - 8, v - 8, u + 8, v + 8], fill=(220, 0, 0), outline=(0, 0, 0))
    # texts
    d.text((20, 12), title, fill=(0, 0, 0), font=_font(26))
    for i, text in enumerate(lines[:2]):
        d.text((20, 46 + 22 * i), text, fill=(60, 60, 60), font=_font(17))
    bar = 500.0 * scale
    d.line([(30, SIZE_PX - 30), (30 + bar, SIZE_PX - 30)], fill=(0, 0, 0), width=5)
    d.text((30, SIZE_PX - 58), "50 cm", fill=(0, 0, 0), font=_font(17))
    d.text((SIZE_PX - 330, SIZE_PX - 34), "green = start, red = end, colour = lap", fill=(90, 90, 90), font=_font(15))
    img.save(out_path)
    return True


# ── one file ───────────────────────────────────────────────────

def make_picture(path, quiet=False):
    t_begin = time.time()
    points, events, header = read_scan(path)
    name = os.path.basename(path)
    if len(points) < 2000:
        print(f"[scan_view] {name}: no usable LIDAR data in this file")
        return None
    looks = _Looks(points)
    if not quiet:
        print(f"[scan_view] {name}: {points[-1][0] - points[0][0]:.0f} s of LIDAR data, working out the map "
              f"(on the Pi this takes a minute or two) ...")
    m, way = build_from_scans(looks, events, points[0][0], points[-1][0])
    how = "worked out from the scans"
    seconds = points[-1][0] - points[0][0]
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(way, way[1:]))
    turned = way[-1][2] if way else 0.0
    end = math.hypot(way[-1][0] - way[0][0], way[-1][1] - way[0][1]) if way else 0.0
    kind = "autonomous drive (RUN)" if "_RUN_" in name else "manual drive (REC)" if "_REC_" in name else "drive"
    lines = [f"{kind}: {seconds:.0f} s, {length / 1000.0:.1f} m driven, turned {turned:+.0f} deg = {abs(turned) / 360.0:.1f} laps, "
             f"ended {end / 10.0:.0f} cm from the start",
             f"way: {how} | {len(way)} positions | {len(m.cloud)} wall points"]
    finished = [text for _, text in events if "FINISHED" in text]
    if finished:
        lines[1] = finished[-1].replace("NAV ", "")[:125]
    out = path[:-len(".csv.gz")] + ".png" if path.endswith(".csv.gz") else path + ".png"
    if not draw(m, way, name, lines, out):
        print(f"[scan_view] {name}: nothing to draw")
        return None
    print(f"[scan_view] {name} -> {out}   ({time.time() - t_begin:.0f} s)")
    return out


def main(argv):
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), SCAN_DIR)
    if len(argv) >= 2 and argv[1] == "all":
        files = [f for f in sorted(glob.glob(os.path.join(folder, "scan_*.csv.gz")))
                 if not os.path.exists(f[:-len(".csv.gz")] + ".png")]
        if not files:
            print("[scan_view] every scan in scans/ already has a picture")
    elif len(argv) >= 2:
        files = argv[1:]
    else:
        files = sorted(glob.glob(os.path.join(folder, "scan_*.csv.gz")), key=os.path.getmtime)[-1:]
        if not files:
            print(f"[scan_view] no scan files in {folder}")
    for f in files:
        try:
            make_picture(f)
        except Exception as e:
            print(f"[scan_view] {os.path.basename(f)}: could not make a picture: {e!r}")


if __name__ == "__main__":
    main(sys.argv)
