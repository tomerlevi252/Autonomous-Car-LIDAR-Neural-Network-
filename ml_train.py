"""
ml_train.py
===========
STEPS 3-4 of the PDF — train the neural network and check how good it is.

    python3 ml_train.py              build the dataset from scans/ + train + evaluate + save
    python3 ml_train.py --no-build   use the existing ml_data/dataset.npz
    python3 ml_train.py --cv         also "leave one run out" cross-validation (slower, for the report)
    python3 ml_train.py --lstm       also train the LSTM (PDF section 10) and compare  (slow on the Pi)
    python3 ml_train.py --fast       only the network the car drives with (what the TRAIN button runs)

It runs on the Raspberry Pi itself (only NumPy + Pillow, both already there)
in about a minute; it also runs on any computer with Python + NumPy.
The TRAIN button on the car's screen runs exactly this.

What it does, in the order of the PDF:
  1. (ml_dataset.py) recordings -> table of  LIDAR inputs -> what the driver did
  2. SPLIT BY RUNS, not randomly (PDF section 8):
         the last TEST_RUNS recordings      = test       (never seen in training)
         the TEST_RUNS before them          = validation (to know when to stop training)
         all the others                     = training
  3. Train the neural networks (ml_model.py):
         MLP "classes"  : LIDAR -> F / L / R           (PDF section 7)
         MLP "joystick" : LIDAR -> joy_x, joy_y         (PDF sections 12-13, "even better")
  4. Evaluate on the test runs (PDF section 9): accuracy, CONFUSION MATRIX,
     precision / recall / F1, and the MODEL COMPARISON of PDF section 11:
         always-F  vs  rule-based  vs  MLP classes  vs  MLP joystick  (vs LSTM)
  5. Train the chosen network (MODEL_TYPE) on ALL the runs and save it:
         ml_model.npz   + a report:  ml_data/train_report.txt / .png
"""

import json
import os
import sys
import time

import numpy as np

import ml_dataset
import ml_features as mf
from ml_model import MLP, Ensemble, joystick_to_action, joystick_to_action_array

HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE        = "ml_model.npz"
MODEL_TYPE        = "joystick"    # the network the car drives with: "joystick" (PDF 12-13) or "classes" (PDF 7)
HIDDEN            = (32, 16)      # V3: 32 -> 16 neurons (the PDF's 16 -> 8 is too small to also learn the
                                  # recovery copies; tested in the simulator). (16, 8) = exactly the PDF.
EPOCHS            = 40            # V3: always this many passes over the data (with the recovery copies the
EARLY_STOPPING    = False         # network does not over-learn; early stopping stopped too soon)
BATCH_SIZE        = 64            # PDF: batch_size=64
LEARNING_RATE     = 0.003
TEST_RUNS         = 2             # recordings kept for testing (and as many for validation)
MIN_CLASS_SAMPLES = 30            # an action with fewer real examples than this is left out of the classifier
USE_MIRRORED      = True          # train also on the left<->right mirrored samples
ENSEMBLE_SIZE     = 3             # V3: the car drives with this many networks, their answers averaged
                                  # (1 = a single network)
USE_AUGMENTED     = True          # train also on the recovery copies (V3, see ml_dataset.py)
SEED              = 1


# ══════════════════════════════════════════════════════════════
#  The rule-based baseline (PDF section 11) — no learning at all
# ══════════════════════════════════════════════════════════════

def rule_based(vec):
    """
    The simplest possible driver, written by hand, to compare the network with:
    if there is room straight ahead go forward, otherwise turn to the side
    with more room. Works on the same inputs as the network.
    """
    n = len(vec)
    front = min(vec[0], vec[1 % n], vec[-1])
    right = np.mean([vec[k] for k in range(1, n // 2)])
    left  = np.mean([vec[n - k] for k in range(1, n // 2)])
    if front * mf.MAX_MM > 300:
        return 'F'
    return 'R' if right > left else 'L'


# ══════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════

def confusion(true_actions, pred_actions, classes):
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(true_actions, pred_actions):
        if t in idx and p in idx:
            m[idx[t], idx[p]] += 1
    return m


def metrics_text(cm, classes, title):
    lines = [title]
    n = len(classes)
    total = cm.sum()
    acc = np.trace(cm) / total if total else 0.0
    lines.append(f"  accuracy: {acc * 100:.1f}%   ({np.trace(cm)} of {total} correct)")
    lines.append("  confusion matrix (rows = what the DRIVER did, columns = what the MODEL said)")
    lines.append("             " + "".join(f"{c:>7s}" for c in classes))
    for i, c in enumerate(classes):
        lines.append(f"    driver {c:>2s}" + "".join(f"{cm[i, j]:7d}" for j in range(n)))
    lines.append("    action   precision  recall    F1    examples")
    f1s = []
    for i, c in enumerate(classes):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec  = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        lines.append(f"      {c:>2s}     {prec * 100:6.1f}%   {rec * 100:6.1f}%  {f1:5.2f}   {cm[i, :].sum():6d}")
    lines.append(f"    macro-F1 (average over the actions): {np.mean(f1s):.3f}")
    return "\n".join(lines), acc, float(np.mean(f1s))


def r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def balanced_weights(y, n):
    counts = np.bincount(y, minlength=n).astype(float)
    w = counts.sum() / (n * np.maximum(counts, 1.0))
    return {i: float(w[i]) for i in range(n)}


# ══════════════════════════════════════════════════════════════
#  Data
# ══════════════════════════════════════════════════════════════

def load_dataset(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d['pulse_s'] = json.loads(str(d['pulse_s'])) if 'pulse_s' in d else {}
    if 'augmented' not in d:
        d['augmented'] = np.zeros(len(d['X']), dtype=np.int8)
    d['real'] = (d['mirrored'] == 0) & (d['augmented'] == 0)    # what really happened
    return d


def pick_classes(d):
    real = d['real']
    labels, counts = np.unique(d['y'][real], return_counts=True)
    cnt = dict(zip(labels, counts))
    order = [c for c in ['F', 'L', 'R', 'B', 'S'] if c in cnt]
    classes = [c for c in order if cnt[c] >= MIN_CLASS_SAMPLES]
    dropped = [f"{c} ({cnt[c]})" for c in order if cnt[c] < MIN_CLASS_SAMPLES]
    return classes, dropped


def mask_of(d, runs, mirrored_ok, classes=None):
    m = np.isin(d['run_id'], runs)
    if classes is not None:
        m &= np.isin(d['y'], classes)
    if not mirrored_ok:
        m &= d['real']                     # testing: only what really happened
    elif not USE_AUGMENTED:
        m &= d['augmented'] == 0
    return m


def train_one(d, task, classes, train_runs, val_runs, epochs=EPOCHS, hidden=HIDDEN, log=print, seed=SEED):
    """Train one MLP. task: 'classes' or 'joystick'. seed: its random start."""
    cls = classes if task == "classes" else None
    mtr = mask_of(d, train_runs, USE_MIRRORED, cls)
    mva = mask_of(d, val_runs, False, cls) if len(val_runs) else None
    if task == "classes":
        net = MLP(d['X'].shape[1], hidden, classes, seed=seed, task="classify")
        ytr = np.array([classes.index(c) for c in d['y'][mtr]])
        yva = np.array([classes.index(c) for c in d['y'][mva]]) if mva is not None else None
        cw = balanced_weights(ytr, len(classes))
    else:
        net = MLP(d['X'].shape[1], hidden, seed=seed, task="joystick")
        ytr, yva, cw = d['joy'][mtr], (d['joy'][mva] if mva is not None else None), None
    hist = net.fit(d['X'][mtr], ytr, d['X'][mva] if mva is not None else None, yva,
                   epochs=epochs, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                   class_weight=cw, seed=seed, log=log, early_stop=EARLY_STOPPING)
    return net, hist


def predicted_actions(net, X):
    p = net.predict_proba(X)
    if net.task == "joystick":
        return list(joystick_to_action_array(p))
    return [net.classes[k] for k in p.argmax(1)]


def cross_validate(d, task, classes, log=print):
    """Leave-one-run-out: train on all runs but one, test on that one; repeat for every run."""
    runs = sorted(set(d['run_id'].tolist()))
    accs = []
    for r in runs:
        others = [x for x in runs if x != r]
        val = others[-1:]
        net, _ = train_one(d, task, classes, [x for x in others if x not in val], val, log=None)
        m = mask_of(d, [r], False, classes)
        if not m.any():
            continue
        acc = float(np.mean(np.array(predicted_actions(net, d['X'][m])) == d['y'][m]))
        accs.append(acc)
        log(f"  test on run {r + 1:2d} ({d['run_names'][r]}): {acc * 100:5.1f}%  ({int(m.sum())} samples)")
    log(f"  average over {len(accs)} runs: {np.mean(accs) * 100:.1f}%  (+/- {np.std(accs) * 100:.1f})")
    return accs


# ══════════════════════════════════════════════════════════════
#  Report picture
# ══════════════════════════════════════════════════════════════

def draw_report_png(path, hist, cm, classes, title_lines, scatter=None):
    """learning curves + confusion matrix (+ joystick scatter) as a picture (Pillow only)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return
    W, H = 1400, 560
    img = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        big  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
    except Exception:
        font = big = ImageFont.load_default()
    y = 10
    for i, t in enumerate(title_lines):
        dr.text((20, y), t, fill=(0, 0, 0), font=big if i == 0 else font)
        y += 24 if i == 0 else 19

    # learning curves
    x0, y0, w, h = 60, 150, 460, 330
    dr.text((x0, y0 - 26), "learning curves", fill=(0, 0, 0), font=big)
    dr.rectangle([x0, y0, x0 + w, y0 + h], outline=(160, 160, 160))
    series = [('loss', (31, 119, 180), 'train loss'), ('val_loss', (255, 127, 14), 'validation loss'),
              ('acc', (44, 160, 44), 'train accuracy'), ('val_acc', (214, 39, 40), 'validation accuracy')]
    top = max(1.0, max(max(r.get(k, 0) for r in hist) for k, _, _ in series))
    n_ep = len(hist)
    for k, col, name in series:
        pts = [(x0 + w * i / max(1, n_ep - 1), y0 + h - h * r[k] / top) for i, r in enumerate(hist) if k in r]
        if len(pts) > 1:
            dr.line(pts, fill=col, width=2)
    for i, (k, col, name) in enumerate(series):
        dr.line([(x0 + 200, y0 + 14 + 18 * i), (x0 + 220, y0 + 14 + 18 * i)], fill=col, width=3)
        dr.text((x0 + 226, y0 + 6 + 18 * i), name, fill=(0, 0, 0), font=font)
    dr.text((x0, y0 + h + 6), f"epoch 1 .. {n_ep}", fill=(80, 80, 80), font=font)
    dr.text((x0 - 40, y0 - 4), f"{top:.1f}", fill=(80, 80, 80), font=font)
    dr.text((x0 - 20, y0 + h - 8), "0", fill=(80, 80, 80), font=font)

    # confusion matrix
    n = len(classes)
    cx, cy = 610, 150
    cell = min(80, 300 // n)
    dr.text((cx - 20, cy - 46), "confusion matrix (test runs)", fill=(0, 0, 0), font=big)
    dr.text((cx - 20, cy - 24), "rows = driver, columns = network", fill=(80, 80, 80), font=font)
    for i in range(n):
        dr.text((cx - 22, cy + i * cell + cell // 2 - 8), classes[i], fill=(0, 0, 0), font=big)
        dr.text((cx + i * cell + cell // 2 - 6, cy + n * cell + 4), classes[i], fill=(0, 0, 0), font=big)
        row = cm[i].sum() or 1
        for j in range(n):
            v = cm[i, j] / row
            shade = int(255 - 200 * v)
            col = (shade, shade, 255) if i == j else (255, shade, shade)
            dr.rectangle([cx + j * cell, cy + i * cell, cx + (j + 1) * cell, cy + (i + 1) * cell],
                         fill=col, outline=(255, 255, 255))
            dr.text((cx + j * cell + 8, cy + i * cell + cell // 2 - 8), f"{cm[i, j]}", fill=(0, 0, 0), font=font)

    # joystick: driver's steering vs network's steering over time (test runs)
    if scatter is not None:
        sx, sy, sw, sh = 920, 150, 440, 330
        dr.text((sx, sy - 46), "steering on the test runs (joy_x)", fill=(0, 0, 0), font=big)
        dr.text((sx, sy - 24), "black = driver, red = network", fill=(80, 80, 80), font=font)
        dr.rectangle([sx, sy, sx + sw, sy + sh], outline=(160, 160, 160))
        dr.line([(sx, sy + sh / 2), (sx + sw, sy + sh / 2)], fill=(220, 220, 220))
        true_x, pred_x = scatter
        k = min(len(true_x), 400)
        for arr, col in ((true_x[:k], (0, 0, 0)), (pred_x[:k], (214, 39, 40))):
            pts = [(sx + sw * i / max(1, k - 1), sy + sh / 2 - sh / 2 * float(v)) for i, v in enumerate(arr)]
            if len(pts) > 1:
                dr.line(pts, fill=col, width=2)
        dr.text((sx - 26, sy - 6), "+1", fill=(80, 80, 80), font=font)
        dr.text((sx - 26, sy + sh - 10), "-1", fill=(80, 80, 80), font=font)
        dr.text((sx, sy + sh + 6), f"first {k} samples (0.1 s each)   up = right", fill=(80, 80, 80), font=font)
    img.save(path)


# ══════════════════════════════════════════════════════════════

def main(argv, on_log=None):
    """Returns 0 when a model was saved. on_log(text): also called with every line (the car's screen)."""
    log_lines = []

    def log(s=""):
        print(s, flush=True)
        log_lines.append(s)
        if on_log:
            try:
                on_log(s)
            except Exception:
                pass

    out_dir = os.path.join(HERE, ml_dataset.OUT_DIR)
    ds_path = os.path.join(out_dir, "dataset.npz")
    if "--no-build" not in argv or not os.path.exists(ds_path):
        log("STEP 1 - building the dataset from the database car_routes.db")
        if not ml_dataset.build(quiet=False):
            return 1
        log()
    d = load_dataset(ds_path)
    runs = list(range(len(d['run_names'])))
    classes, dropped = pick_classes(d)
    n_in = d['X'].shape[1]
    log(f"Dataset: {len(runs)} runs, {int(d['real'].sum())} samples, "
        f"{n_in} inputs (LIDAR sectors), actions: {', '.join(classes)}")
    if dropped:
        log(f"  left out of the classifier (too few examples to learn from): {', '.join(dropped)}")
    if len(runs) < 3:
        log("At least 3 recorded runs are needed (training / validation / test). Record more drives.")
        return 1
    if len(classes) < 2:
        log("Not enough different actions recorded to train. Record more drives.")
        return 1

    # ── split by runs (PDF section 8) ─────────────────────────
    n_test = min(TEST_RUNS, max(1, len(runs) // 5))
    test_runs, val_runs, train_runs = runs[-n_test:], runs[-2 * n_test:-n_test], runs[:-2 * n_test]
    log("")
    log("STEP 2 - split by runs (PDF section 8)")
    log(f"  training   = runs {train_runs[0] + 1}-{train_runs[-1] + 1}")
    log(f"  validation = runs {', '.join(str(r + 1) for r in val_runs)}")
    log(f"  test       = runs {', '.join(str(r + 1) for r in test_runs)}   (never seen while training)")

    # --fast (the TRAIN button on the car): only the network the car drives with; the
    # PDF-section-7 classification network for the comparison table is skipped
    tasks = [MODEL_TYPE] if "--fast" in argv else ["classes", "joystick"]
    mte = mask_of(d, test_runs, False, classes)
    Xte, true_te = d['X'][mte], list(d['y'][mte])

    # ── train both MLPs ───────────────────────────────────────
    results, nets = {}, {}
    for task in tasks:
        log("")
        outs = f"{len(classes)} ({', '.join(classes)})" if task == "classes" else "2 (joy_x, joy_y)"
        log(f"STEP 3 - training MLP '{task}':  {n_in} -> {' -> '.join(str(h) for h in HIDDEN)} -> {outs}")
        t0 = time.time()
        net, hist = train_one(d, task, classes, train_runs, val_runs, log=log)
        log(f"  training took {time.time() - t0:.1f} s")
        nets[task] = (net, hist)
        pred = predicted_actions(net, Xte)
        cm = confusion(true_te, pred, classes)
        txt, acc, f1 = metrics_text(cm, classes, f"STEP 4 - TEST runs, MLP '{task}'")
        log(txt)
        results[task] = (acc, f1, cm)
        if task == "joystick":
            p = net.predict_proba(Xte)
            jt = d['joy'][mte]
            log(f"    steering joy_x: R^2 = {r2(jt[:, 0], p[:, 0]):.3f}, correlation "
                f"{np.corrcoef(jt[:, 0], p[:, 0])[0, 1]:.3f}, average error {np.abs(jt[:, 0] - p[:, 0]).mean():.3f}")
            log(f"    throttle joy_y: average error {np.abs(jt[:, 1] - p[:, 1]).mean():.3f}")

    # ── baselines (PDF section 11) ────────────────────────────
    rb = [rule_based(v) for v in Xte]
    cm_rb = confusion(true_te, rb, classes)
    txt_rb, acc_rb, f1_rb = metrics_text(cm_rb, classes, "for comparison: RULE-BASED driver (no learning), TEST runs")
    log("")
    log(txt_rb)
    acc_f = float(np.mean(np.array(true_te) == 'F'))

    lstm_res = None
    if "--lstm" in argv:
        try:
            import ml_lstm
            log("")
            log("EXTRA - LSTM (PDF section 10): the last 20 LIDAR pictures (2 s) -> joy_x, joy_y")
            lstm_res = ml_lstm.train_and_test(d, train_runs, val_runs, test_runs, classes, log=log)
        except ImportError:
            log("ml_lstm.py not found - LSTM skipped")

    net, _ = nets[MODEL_TYPE]
    t = time.perf_counter()
    for v in Xte[:300]:
        net.predict(v)
    infer_ms = (time.perf_counter() - t) / max(1, min(300, len(Xte))) * 1000.0
    log("")
    log("MODEL COMPARISON on the test runs (PDF section 11)")
    log("  model                          accuracy   macro-F1   parameters")
    log(f"  always 'F' (no brain)           {acc_f * 100:6.1f}%        -           0")
    log(f"  rule-based (hand written)       {acc_rb * 100:6.1f}%    {f1_rb:6.3f}           0")
    for task in tasks:
        a, f, _ = results[task]
        log(f"  MLP {task:9s}                  {a * 100:6.1f}%    {f:6.3f}      {nets[task][0].n_params():6d}")
    if lstm_res:
        log(f"  LSTM joystick (2 s history)     {lstm_res['acc'] * 100:6.1f}%    {lstm_res['f1']:6.3f}"
            f"      {lstm_res['params']:6d}")
        # fair: the MLP on exactly the same test samples (the last picture of every LSTM sequence)
        same = np.array(predicted_actions(nets['joystick'][0], lstm_res['last_pictures']))
        log(f"    (on exactly the LSTM's {len(same)} test samples the MLP joystick gets "
            f"{np.mean(same == lstm_res['actions']) * 100:.1f}%)")
    log(f"  decision time of the MLP on this computer: {infer_ms:.3f} ms "
        f"(the LIDAR gives a new picture only every ~170 ms)")

    if "--cv" in argv:
        log("")
        log(f"EXTRA - leave-one-run-out cross-validation, MLP '{MODEL_TYPE}'")
        cross_validate(d, MODEL_TYPE, classes, log=log)

    # ── final model on all the runs ───────────────────────────
    net, hist = nets[MODEL_TYPE]
    best_epoch = (min(range(len(hist)), key=lambda i: hist[i].get('val_loss', hist[i]['loss'])) + 1
                  if EARLY_STOPPING else EPOCHS)
    log("")
    log(f"STEP 5 - final MLP '{MODEL_TYPE}' trained on ALL {len(runs)} runs ({best_epoch} epochs)"
        + (f", {ENSEMBLE_SIZE} networks from different random starts (an ensemble)" if ENSEMBLE_SIZE > 1 else ""))
    members = []
    for k in range(max(1, ENSEMBLE_SIZE)):
        t0 = time.time()
        net_k, _ = train_one(d, MODEL_TYPE, classes, runs, [], epochs=best_epoch, log=None, seed=SEED + 10 + k)
        members.append(net_k)
        log(f"  network {k + 1}/{max(1, ENSEMBLE_SIZE)} trained ({time.time() - t0:.0f} s)")
    final = Ensemble(members) if len(members) > 1 else members[0]
    acc, f1, cm = results[MODEL_TYPE]
    final.meta = {
        'n_sectors': int(n_in), 'max_mm': float(d['max_mm']),
        'percentile': mf.PERCENTILE, 'min_valid_mm': mf.MIN_VALID_MM,
        'pulse_s': d['pulse_s'], 'test_accuracy': acc, 'test_macro_f1': f1,
        'rule_based_accuracy': acc_rb, 'runs': len(runs), 'actions': classes,
        'samples': int(d['real'].sum()), 'trained': time.strftime('%Y-%m-%d %H:%M'),
    }
    model_path = os.path.join(HERE, MODEL_FILE)
    final.save(model_path)
    log(final.summary())
    log(f"  saved -> {model_path}")

    scatter = None
    if MODEL_TYPE == "joystick":
        scatter = (d['joy'][mte][:, 0], nets['joystick'][0].predict_proba(Xte)[:, 0])
    draw_report_png(os.path.join(out_dir, "train_report.png"), hist, cm, classes,
                    [f"CAR ML - neural network training report ({time.strftime('%Y-%m-%d %H:%M')})",
                     f"MLP '{MODEL_TYPE}': {n_in} LIDAR inputs -> {' -> '.join(str(h) for h in HIDDEN)} -> "
                     f"{'joy_x, joy_y' if MODEL_TYPE == 'joystick' else ', '.join(classes)}   |   "
                     f"{len(runs)} runs, {int(d['real'].sum())} samples",
                     f"TEST runs (never seen in training):  network {acc * 100:.1f}%   "
                     f"rule-based {acc_rb * 100:.1f}%   always-F {acc_f * 100:.1f}%",
                     f"actions: {', '.join(classes)}"], scatter)
    log(f"  report  -> ml_data/train_report.txt and ml_data/train_report.png")
    log(f"RESULT: test accuracy {acc * 100:.0f}% (rule-based {acc_rb * 100:.0f}%), "
        f"{len(runs)} runs, {int(d['real'].sum())} samples")
    with open(os.path.join(out_dir, "train_report.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
