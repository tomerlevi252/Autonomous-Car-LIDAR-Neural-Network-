"""
ml_lstm.py
==========
PDF sections 10-11 — the LSTM experiment: does HISTORY help?

The MLP sees ONE LIDAR picture. The LSTM sees the last WINDOW pictures
(20 x 0.1 s = 2 seconds), one after the other, and keeps a "memory" (its
hidden state) while reading them — so it can learn things like "the wall on
the right has been getting closer for a second, so the driver turns left".

        20 pictures x 16 LIDAR inputs
            |
        LSTM(32)          <- 32 memory cells
            |
        Dense(16) + ReLU
            |
        Dense(2)  + tanh  -> joy_x (steering), joy_y (throttle)

Plain NumPy again (training with back-propagation through time + Adam), so
it runs on the Pi. It is slower to train than the MLP (a few minutes on the Pi).

    python3 ml_lstm.py            train + test it, save ml_model_lstm.npz
    python3 ml_lstm.py --use      ... and make the car drive with it (copies it to ml_model.npz)
    python3 ml_train.py --lstm    the MLP and the LSTM side by side in one comparison table

Note: the research question of the PDF — "Does temporal information from an
LSTM improve autonomous navigation compared with a feed-forward MLP?" — is
answered by that comparison table (and by test drives / ml_simulate.py).
"""

import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

WINDOW        = 20      # pictures in one sequence (PDF: 20 samples = 2 s at 10 Hz)
UNITS         = 32      # PDF: LSTM(32)
DENSE         = 16      # PDF: Dense(16)
EPOCHS        = 40
BATCH_SIZE    = 64
LEARNING_RATE = 0.003
MAX_GAP_S     = 0.35    # a sequence must not have a hole (a pause in the recording) longer than this
SEED          = 1


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


# ══════════════════════════════════════════════════════════════
#  Sequences from the dataset
# ══════════════════════════════════════════════════════════════

def make_sequences(d, runs, mirrored_ok=True, window=WINDOW):
    """-> (S [n, window, inputs], Y [n, 2] joystick of the LAST picture, actions [n])"""
    S, Y, A = [], [], []
    for r in runs:
        for mir in ((0, 1) if mirrored_ok else (0,)):
            idx = np.where((d['run_id'] == r) & (d['mirrored'] == mir) & (d.get('augmented', 0) == 0))[0]
            if len(idx) < window:
                continue
            idx = idx[np.argsort(d['t'][idx])]
            t = d['t'][idx]
            for k in range(window - 1, len(idx)):
                part = idx[k - window + 1:k + 1]
                if np.max(np.diff(t[k - window + 1:k + 1])) > MAX_GAP_S:
                    continue
                S.append(d['X'][part]); Y.append(d['joy'][idx[k]]); A.append(d['y'][idx[k]])
    if not S:
        return np.zeros((0, window, d['X'].shape[1]), np.float32), np.zeros((0, 2), np.float32), np.array([])
    return np.array(S, np.float32), np.array(Y, np.float32), np.array(A)


# ══════════════════════════════════════════════════════════════
#  The LSTM network (training)
# ══════════════════════════════════════════════════════════════

class LSTMTrainer:
    def __init__(self, n_inputs, units=UNITS, dense=DENSE, seed=SEED):
        rng = np.random.default_rng(seed)
        u = units
        self.n_inputs, self.units, self.dense = n_inputs, units, dense
        self.P = {
            'Wx': (rng.standard_normal((n_inputs, 4 * u)) * np.sqrt(1.0 / n_inputs)).astype(np.float32),
            'Wh': (np.linalg.qr(rng.standard_normal((4 * u, u)))[0].T).astype(np.float32),
            'bl': np.zeros(4 * u, np.float32),
            'W0': (rng.standard_normal((u, dense)) * np.sqrt(2.0 / u)).astype(np.float32),
            'b0': np.zeros(dense, np.float32),
            'W1': (rng.standard_normal((dense, 2)) * np.sqrt(1.0 / dense)).astype(np.float32),
            'b1': np.zeros(2, np.float32),
        }
        self.P['bl'][u:2 * u] = 1.0          # forget gate starts "open" (as Keras does)

    def forward(self, S, keep=False):
        P, u = self.P, self.units
        B, T, _ = S.shape
        h = np.zeros((B, u), np.float32); c = np.zeros((B, u), np.float32)
        cache = []
        for t in range(T):
            z = S[:, t] @ P['Wx'] + h @ P['Wh'] + P['bl']
            i, f, g, o = _sig(z[:, :u]), _sig(z[:, u:2*u]), np.tanh(z[:, 2*u:3*u]), _sig(z[:, 3*u:])
            c_new = f * c + i * g
            tc = np.tanh(c_new)
            h_new = o * tc
            if keep:
                cache.append((h, c, i, f, g, o, tc))
            h, c = h_new, c_new
        a0 = h @ P['W0'] + P['b0']
        r0 = np.maximum(a0, 0.0)
        out = np.tanh(r0 @ P['W1'] + P['b1'])
        if keep:
            return out, (cache, h, r0)
        return out

    def backward(self, S, Y, out, state):
        """gradients of the mean squared error (back-propagation through time)"""
        P, u = self.P, self.units
        cache, hT, r0 = state
        B, T, _ = S.shape
        G = {k: np.zeros_like(v) for k, v in P.items()}
        dout = 2.0 * (out - Y) * (1.0 - out * out) / (Y.shape[1] * B)
        G['W1'] = r0.T @ dout; G['b1'] = dout.sum(0)
        dr0 = dout @ P['W1'].T * (r0 > 0)
        G['W0'] = hT.T @ dr0; G['b0'] = dr0.sum(0)
        dh = dr0 @ P['W0'].T
        dc = np.zeros((B, u), np.float32)
        for t in range(T - 1, -1, -1):
            h_prev, c_prev, i, f, g, o, tc = cache[t]
            do = dh * tc
            dc = dc + dh * o * (1.0 - tc * tc)
            di, dg, df = dc * g, dc * i, dc * c_prev
            dz = np.concatenate([di * i * (1 - i), df * f * (1 - f), dg * (1 - g * g), do * o * (1 - o)], axis=1)
            G['Wx'] += S[:, t].T @ dz
            G['Wh'] += h_prev.T @ dz
            G['bl'] += dz.sum(0)
            dh = dz @ P['Wh'].T
            dc = dc * f
        return G

    def fit(self, S, Y, S_val, Y_val, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
            patience=8, log=print):
        rng = np.random.default_rng(SEED)
        m = {k: np.zeros_like(v) for k, v in self.P.items()}
        v = {k: np.zeros_like(x) for k, x in self.P.items()}
        step, best, bad, hist = 0, (float('inf'), None, 0), 0, []
        for ep in range(1, epochs + 1):
            order = rng.permutation(len(S))
            tot = 0.0
            for s in range(0, len(S), batch_size):
                b = order[s:s + batch_size]
                out, state = self.forward(S[b], keep=True)
                tot += float(((out - Y[b]) ** 2).mean(axis=1).sum())
                G = self.backward(S[b], Y[b], out, state)
                norm = np.sqrt(sum(float((g * g).sum()) for g in G.values()))
                scale = min(1.0, 5.0 / (norm + 1e-9))          # gradient clipping
                step += 1
                for k in self.P:
                    g = G[k] * scale
                    m[k] = 0.9 * m[k] + 0.1 * g
                    v[k] = 0.999 * v[k] + 0.001 * g * g
                    self.P[k] -= (lr * (m[k] / (1 - 0.9 ** step)) /
                                  (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)).astype(np.float32)
            rec = {'epoch': ep, 'loss': tot / len(S)}
            if len(S_val):
                rec['val_loss'] = float(((self.forward(S_val) - Y_val) ** 2).mean())
                if rec['val_loss'] < best[0] - 1e-4:
                    best, bad = (rec['val_loss'], {k: x.copy() for k, x in self.P.items()}, ep), 0
                else:
                    bad += 1
            hist.append(rec)
            if log and (ep == 1 or ep % 5 == 0):
                log(f"  epoch {ep:3d}/{epochs}  loss {rec['loss']:.4f}"
                    + (f"  val_loss {rec['val_loss']:.4f}" if 'val_loss' in rec else ""))
            if len(S_val) and bad >= patience:
                if log:
                    log(f"  no improvement for {patience} epochs - stopping early")
                break
        self.best_epoch = best[2] if best[1] is not None else len(hist)
        if best[1] is not None:
            self.P = best[1]
            if log:
                log(f"  keeping the weights of epoch {best[2]} (best validation loss {best[0]:.4f})")
        return hist

    def n_params(self):
        return int(sum(x.size for x in self.P.values()))

    def save(self, path, meta):
        """Same file format as ml_model.py's LSTMNet (Keras weight layout), so ml_driver.py can drive with it."""
        info = {'kind': 'lstm', 'task': 'joystick', 'n_inputs': self.n_inputs, 'window': WINDOW,
                'hidden': [self.dense], 'classes': ['joy_x', 'joy_y'], 'meta': meta,
                'saved': time.strftime('%Y-%m-%d %H:%M:%S')}
        np.savez(path, info=json.dumps(info), lstm_kernel=self.P['Wx'], lstm_recurrent=self.P['Wh'],
                 lstm_bias=self.P['bl'], W0=self.P['W0'], b0=self.P['b0'], W1=self.P['W1'], b1=self.P['b1'])


# ══════════════════════════════════════════════════════════════

def train_and_test(d, train_runs, val_runs, test_runs, classes, log=print, epochs=EPOCHS):
    """Used by ml_train.py --lstm. Returns {'acc', 'f1', 'params', 'net'}."""
    from ml_model import joystick_to_action_array
    import ml_train
    Str, Ytr, _ = make_sequences(d, train_runs, True)
    Sva, Yva, _ = make_sequences(d, val_runs, False)
    Ste, Yte, Ate = make_sequences(d, test_runs, False)
    log(f"  sequences: {len(Str)} training, {len(Sva)} validation, {len(Ste)} test "
        f"(each {WINDOW} pictures x {d['X'].shape[1]} inputs)")
    if len(Str) < 100 or len(Ste) == 0:
        log("  not enough continuous driving for LSTM sequences - record longer runs")
        return None
    net = LSTMTrainer(d['X'].shape[1])
    t0 = time.time()
    net.fit(Str, Ytr, Sva, Yva, epochs=epochs, log=log)
    log(f"  training took {time.time() - t0:.0f} s")
    pred = net.forward(Ste)
    cm = ml_train.confusion(list(Ate), list(joystick_to_action_array(pred)), classes)
    txt, acc, f1 = ml_train.metrics_text(cm, classes, "  TEST runs, LSTM joystick")
    log(txt)
    log(f"    steering joy_x: R^2 = {ml_train.r2(Yte[:, 0], pred[:, 0]):.3f}, "
        f"correlation {np.corrcoef(Yte[:, 0], pred[:, 0])[0, 1]:.3f}")
    res = {'acc': acc, 'f1': f1, 'params': net.n_params(), 'net': net, 'epochs': net.best_epoch,
           'last_pictures': Ste[:, -1, :], 'actions': Ate}
    return res


def main(argv):
    import ml_dataset
    import ml_features as mf
    import ml_train
    ds = os.path.join(HERE, ml_dataset.OUT_DIR, "dataset.npz")
    if not os.path.exists(ds) or "--build" in argv:
        ml_dataset.build()
    d = ml_train.load_dataset(ds)
    classes, _ = ml_train.pick_classes(d)
    runs = list(range(len(d['run_names'])))
    if len(runs) < 3:
        print("At least 3 recorded runs are needed.")
        return 1
    n = min(ml_train.TEST_RUNS, max(1, len(runs) // 5))
    print(f"LSTM: {len(runs)} runs; test = runs {[r + 1 for r in runs[-n:]]}")
    res = train_and_test(d, runs[:-2 * n], runs[-2 * n:-n], runs[-n:], classes)
    if not res:
        return 1
    print("\nFinal LSTM on all runs ...")
    S, Y, _ = make_sequences(d, runs, True)
    net = LSTMTrainer(d['X'].shape[1])
    net.fit(S, Y, S[:0], Y[:0], epochs=max(3, res['epochs']), log=None)
    meta = {'n_sectors': int(d['X'].shape[1]), 'max_mm': float(d['max_mm']),
            'percentile': mf.PERCENTILE, 'min_valid_mm': mf.MIN_VALID_MM, 'pulse_s': d['pulse_s'],
            'test_accuracy': res['acc'], 'test_macro_f1': res['f1'], 'runs': len(runs),
            'samples': int(d['real'].sum()), 'trained': time.strftime('%Y-%m-%d %H:%M')}
    path = os.path.join(HERE, "ml_model_lstm.npz")
    net.save(path, meta)
    print(f"saved -> {path}  ({net.n_params()} parameters)")
    if "--use" in argv:
        shutil.copy(path, os.path.join(HERE, "ml_model.npz"))
        print("copied to ml_model.npz - CAR RUN now drives with the LSTM "
              "(press TRAIN NETWORK to go back to the MLP)")
        print("NOTE (V3): the LSTM learns only from your real drives, without the recovery copies, "
              "and V3 has no safety layer - for driving, the MLP ensemble is the tested one. "
              "The LSTM is mainly for the comparison of PDF section 11.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
