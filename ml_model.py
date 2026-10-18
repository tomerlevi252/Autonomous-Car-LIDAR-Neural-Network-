"""
ml_model.py
===========
The NEURAL NETWORK itself — a Multi-Layer Perceptron (MLP), exactly the
model of PDF sections 2 and 7, written in plain NumPy:

        N LIDAR inputs
            |
        Dense(16) + ReLU          <- "hidden layer 1": 16 neurons
            |
        Dense(8)  + ReLU          <- "hidden layer 2": 8 neurons
            |
        Dense(classes) + Softmax  <- one output per action: F, L, R, ...
            |
        probabilities, e.g.  F 0.81  L 0.04  R 0.15   -> the car does F

Why NumPy and not TensorFlow/Keras (the PDF's example)?
  It is the SAME network (same layers, same ReLU/Softmax, same Adam optimizer,
  same "sparse categorical cross-entropy" / mean-squared-error loss) — but
  TensorFlow has no version at all for the Python of the car (Python 3.13 on
  Raspberry Pi OS), while NumPy is already installed there. So the network is
  trained AND driven on the car itself, with nothing to install. Every line of
  the maths is visible below, which also makes it easy to explain.

How a neuron works (for the report):
  every neuron computes   output = ReLU( w1*x1 + w2*x2 + ... + b )
  The w's (weights) and b (bias) start random. TRAINING = showing the network
  thousands of (LIDAR picture -> driver's button) examples and nudging every
  weight a tiny bit, again and again, so the network's answer gets closer to
  the driver's (that "nudging" is back-propagation + the Adam optimizer).

The trained network is saved in ONE file:  ml_model.npz
(the weights + the list of classes + the LIDAR settings it was trained with).
"""

import json
import time

import numpy as np


# ══════════════════════════════════════════════════════════════
#  Joystick <-> action
# ══════════════════════════════════════════════════════════════

STEER_THRESHOLD = 0.33   # |joy_x| above this = "mostly turning" (a third of the time or more)


def joystick_to_action(joy_x: float, joy_y: float, threshold: float = STEER_THRESHOLD) -> str:
    """The button that best matches a joystick position: F / L / R / B."""
    if joy_y < -0.3:
        return 'B'
    if joy_x > threshold:
        return 'R'
    if joy_x < -threshold:
        return 'L'
    return 'F'


def joystick_to_action_array(J, threshold: float = STEER_THRESHOLD):
    J = np.asarray(J)
    out = np.full(len(J), 'F', dtype='<U1')
    out[J[:, 0] > threshold] = 'R'
    out[J[:, 0] < -threshold] = 'L'
    out[J[:, 1] < -0.3] = 'B'
    return out


# ══════════════════════════════════════════════════════════════
#  MLP
# ══════════════════════════════════════════════════════════════

class MLP:
    """
    A small fully-connected network: inputs -> [Dense+ReLU]* -> output layer.

    Two kinds of output (PDF sections 7 and 12):
      task='classify' : Dense(classes) + Softmax, one probability per action
                        (F, L, R ...), loss = cross-entropy
      task='joystick' : Dense(2) + tanh -> joy_x (steering), joy_y (throttle),
                        each -1 .. +1, loss = mean squared error (a "regression")
    """

    kind = "mlp"

    def __init__(self, n_inputs: int, hidden=(16, 8), classes=("F", "L", "R"), seed: int = 0,
                 task: str = "classify"):
        self.n_inputs = int(n_inputs)
        self.hidden   = tuple(int(h) for h in hidden)
        self.task     = task
        self.classes  = list(classes) if task == "classify" else ["joy_x", "joy_y"]
        self.meta     = {}                      # LIDAR settings etc., saved with the model
        rng   = np.random.default_rng(seed)
        sizes = [self.n_inputs, *self.hidden, len(self.classes)]
        self.W, self.b = [], []
        for a, c in zip(sizes[:-1], sizes[1:]):
            # "He" initialisation — the usual start for ReLU layers
            self.W.append((rng.standard_normal((a, c)) * np.sqrt(2.0 / a)).astype(np.float32))
            self.b.append(np.zeros(c, dtype=np.float32))

    # ── using the network ─────────────────────────────────────

    def _forward(self, X):
        """X [batch, inputs] -> (probabilities [batch, classes], every layer's output)."""
        acts = [X]
        h = X
        for i in range(len(self.W)):
            z = h @ self.W[i] + self.b[i]
            if i < len(self.W) - 1:
                h = np.maximum(z, 0.0)                     # ReLU
            elif self.task == "joystick":
                h = np.tanh(z)                             # -1 .. +1
            else:
                z = z - z.max(axis=1, keepdims=True)       # Softmax (numerically safe)
                e = np.exp(z)
                h = e / e.sum(axis=1, keepdims=True)
            acts.append(h)
        return h, acts

    def predict_proba(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        return self._forward(X)[0]

    def predict(self, x):
        """
        ONE LIDAR picture ->
          classify: (action, probability, {action: probability})
          joystick: (None, 1.0, {'joy_x': .., 'joy_y': ..})
        """
        p = self.predict_proba(x)[0]
        if self.task == "joystick":
            return None, 1.0, {'joy_x': float(p[0]), 'joy_y': float(p[1])}
        k = int(np.argmax(p))
        return self.classes[k], float(p[k]), {c: float(v) for c, v in zip(self.classes, p)}

    def n_params(self) -> int:
        return int(sum(w.size + b.size for w, b in zip(self.W, self.b)))

    def summary(self) -> str:
        lines = ["Layer            Output   Params"]
        sizes = [self.n_inputs, *self.hidden, len(self.classes)]
        lines.append(f"Input            {self.n_inputs:6d}        0")
        for i, (a, c) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = "Dense+tanh" if self.task == "joystick" else "Dense+Softmax"
            name = "Dense+ReLU" if i < len(sizes) - 2 else last
            lines.append(f"{name:16s} {c:6d}   {a * c + c:6d}")
        lines.append(f"Total parameters (weights + biases): {self.n_params()}")
        return "\n".join(lines)

    # ── training ──────────────────────────────────────────────

    def fit(self, X, y, X_val=None, y_val=None, epochs=60, batch_size=64, lr=0.003,
            class_weight=None, l2=1e-4, patience=15, seed=0, log=print, early_stop=True):
        """
        Train with mini-batches + Adam (PDF section 7: model.fit).
        classify: y = class indices (0 .. classes-1), loss = cross-entropy
        joystick: y = [[joy_x, joy_y], ...],          loss = mean squared error
        class_weight: {index: weight} — rare actions (L) count more, so the
                      network does not just learn "always F".
        Keeps the weights of the epoch with the best validation loss
        ("early stopping"), and stops when it has not improved for `patience` epochs.
        early_stop=False: always trains all `epochs` (the validation is only measured).
        Returns the history: [{'epoch', 'loss', 'acc', 'val_loss', 'val_acc'}, ...]
        """
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=np.float32)
        reg = self.task == "joystick"
        y = np.asarray(y, dtype=np.float32 if reg else np.int64)
        n_cls = len(self.classes)
        cw = np.ones(n_cls, dtype=np.float32)
        if class_weight:
            for k, v in class_weight.items():
                cw[int(k)] = v

        # Adam optimizer state
        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        beta1, beta2, eps, step = 0.9, 0.999, 1e-8, 0

        history = []
        best = (float('inf'), None)
        bad_epochs = 0
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(X))
            tot_loss = tot_ok = 0.0
            for s in range(0, len(X), batch_size):
                idx = order[s:s + batch_size]
                xb, yb = X[idx], y[idx]
                p, acts = self._forward(xb)
                if reg:
                    err = p - yb
                    tot_loss += float((err ** 2).mean(axis=1).sum())
                    tot_ok   += float(self._same_action(p, yb).sum())
                    # back-propagation through MSE + tanh
                    g = 2.0 * err * (1.0 - p * p) / (err.shape[1] * len(yb))
                else:
                    wts = cw[yb]
                    tot_loss += float(-(np.log(p[np.arange(len(yb)), yb] + 1e-9) * wts).sum())
                    tot_ok   += float((p.argmax(1) == yb).sum())
                    # back-propagation: how much each weight is to blame for the error
                    g = p.copy()
                    g[np.arange(len(yb)), yb] -= 1.0               # d(loss)/d(logits) for softmax+CE
                    g *= (wts / wts.sum())[:, None]
                step += 1
                for i in range(len(self.W) - 1, -1, -1):
                    gW = acts[i].T @ g + l2 * self.W[i]
                    gb = g.sum(axis=0)
                    if i > 0:
                        g = (g @ self.W[i].T) * (acts[i] > 0)      # through ReLU
                    # Adam update
                    mW[i] = beta1 * mW[i] + (1 - beta1) * gW; vW[i] = beta2 * vW[i] + (1 - beta2) * gW * gW
                    mb[i] = beta1 * mb[i] + (1 - beta1) * gb; vb[i] = beta2 * vb[i] + (1 - beta2) * gb * gb
                    c1, c2 = 1 - beta1 ** step, 1 - beta2 ** step
                    self.W[i] -= (lr * (mW[i] / c1) / (np.sqrt(vW[i] / c2) + eps)).astype(np.float32)
                    self.b[i] -= (lr * (mb[i] / c1) / (np.sqrt(vb[i] / c2) + eps)).astype(np.float32)

            rec = {'epoch': epoch, 'loss': tot_loss / len(X), 'acc': tot_ok / len(X)}
            if X_val is not None and len(X_val):
                rec['val_loss'], rec['val_acc'] = self.evaluate(X_val, y_val, cw)
                if rec['val_loss'] < best[0] - 1e-4:
                    best = (rec['val_loss'], ([w.copy() for w in self.W], [b.copy() for b in self.b]), epoch)
                    bad_epochs = 0
                else:
                    bad_epochs += 1
            history.append(rec)
            if log and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
                v = (f"  val_loss {rec['val_loss']:.3f}  val_acc {rec['val_acc']:.3f}"
                     if 'val_loss' in rec else "")
                log(f"  epoch {epoch:3d}/{epochs}  loss {rec['loss']:.3f}  acc {rec['acc']:.3f}{v}")
            if early_stop and X_val is not None and bad_epochs >= patience:
                if log:
                    log(f"  no improvement for {patience} epochs - stopping early")
                break
        if early_stop and best[1] is not None:
            self.W, self.b = best[1]
            if log:
                log(f"  keeping the weights of epoch {best[2]} (best validation loss {best[0]:.3f})")
        return history

    @staticmethod
    def _same_action(p, y):
        """joystick: do prediction and label point to the same action? (for an 'accuracy')"""
        return joystick_to_action_array(p) == joystick_to_action_array(y)

    def evaluate(self, X, y, cw=None):
        """-> (loss, accuracy).  classify: weighted cross-entropy;  joystick: MSE"""
        p = self.predict_proba(X)
        if self.task == "joystick":
            y = np.asarray(y, dtype=np.float32)
            return float(((p - y) ** 2).mean()), float(self._same_action(p, y).mean())
        y = np.asarray(y, dtype=np.int64)
        w = np.ones(len(y), dtype=np.float32) if cw is None else cw[y]
        loss = float(-(np.log(p[np.arange(len(y)), y] + 1e-9) * w).sum() / len(y))
        return loss, float((p.argmax(1) == y).mean())

    # ── saving / loading ──────────────────────────────────────

    def save(self, path: str) -> None:
        data = {f"W{i}": w for i, w in enumerate(self.W)}
        data.update({f"b{i}": b for i, b in enumerate(self.b)})
        info = {'kind': self.kind, 'task': self.task, 'n_inputs': self.n_inputs, 'hidden': list(self.hidden),
                'classes': self.classes, 'meta': self.meta, 'saved': time.strftime('%Y-%m-%d %H:%M:%S')}
        np.savez(path, info=json.dumps(info), **data)


# ══════════════════════════════════════════════════════════════
#  LSTM (inference only) — PDF section 10
#  Trained by ml_lstm.py (NumPy). The weights use Keras' layout, so an LSTM
#  trained in Keras on a computer could be copied into the same file format.
# ══════════════════════════════════════════════════════════════

class LSTMNet:
    """
    window x inputs  ->  LSTM(units)  ->  Dense(16)+ReLU  ->  Dense(2)+tanh (joystick)
                                                              or Dense(classes)+Softmax
    The network sees the last `window` LIDAR pictures (e.g. 20 x 0.1 s = 2 s of history).
    Weight layout = Keras': kernel [in, 4u], recurrent [u, 4u], bias [4u], gates i, f, c, o.
    """

    kind = "lstm"

    def __init__(self, info, arrays):
        self.n_inputs = int(info['n_inputs'])
        self.window   = int(info['window'])
        self.classes  = list(info['classes'])
        self.meta     = info.get('meta', {})
        self.hidden   = tuple(info.get('hidden', []))
        self.task     = info.get('task', 'classify')
        self.Wx, self.Wh, self.bl = arrays['lstm_kernel'], arrays['lstm_recurrent'], arrays['lstm_bias']
        self.units = self.Wh.shape[0]
        self.W = [arrays[f"W{i}"] for i in range(len(self.hidden) + 1)]
        self.b = [arrays[f"b{i}"] for i in range(len(self.hidden) + 1)]

    @staticmethod
    def _sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    def predict_proba(self, seqs) -> np.ndarray:
        """seqs [batch, window, inputs] -> probabilities [batch, classes]"""
        seqs = np.asarray(seqs, dtype=np.float32)
        if seqs.ndim == 2:
            seqs = seqs[None]
        B, u = seqs.shape[0], self.units
        h = np.zeros((B, u), np.float32); c = np.zeros((B, u), np.float32)
        for t in range(seqs.shape[1]):
            z = seqs[:, t] @ self.Wx + h @ self.Wh + self.bl
            i, f, g, o = self._sig(z[:, :u]), self._sig(z[:, u:2*u]), np.tanh(z[:, 2*u:3*u]), self._sig(z[:, 3*u:])
            c = f * c + i * g
            h = o * np.tanh(c)
        x = h
        for k in range(len(self.W)):
            x = x @ self.W[k] + self.b[k]
            if k < len(self.W) - 1:
                x = np.maximum(x, 0.0)
        if self.task == "joystick":
            return np.tanh(x)
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, seq):
        p = self.predict_proba(seq)[0]
        if self.task == "joystick":
            return None, 1.0, {'joy_x': float(p[0]), 'joy_y': float(p[1])}
        k = int(np.argmax(p))
        return self.classes[k], float(p[k]), {c: float(v) for c, v in zip(self.classes, p)}

    def n_params(self) -> int:
        return int(self.Wx.size + self.Wh.size + self.bl.size + sum(w.size + b.size for w, b in zip(self.W, self.b)))


# ══════════════════════════════════════════════════════════════
#  ENSEMBLE (V3) — several networks, their answers averaged
# ══════════════════════════════════════════════════════════════

class Ensemble:
    """
    N networks with the same layers, each trained from a different random
    start (different random first weights + different order of the examples).
    Every one of them learns to drive, but each makes its own small random
    mistakes; the AVERAGE of their answers cancels most of those out.
    A well-known machine-learning technique ("model ensembling") — still
    100% neural networks, just more than one.
    """

    kind = "ensemble"

    def __init__(self, members):
        self.members  = list(members)
        first         = self.members[0]
        self.n_inputs = first.n_inputs
        self.hidden   = first.hidden
        self.task     = first.task
        self.classes  = first.classes
        self.meta     = {}

    def predict_proba(self, X) -> np.ndarray:
        return np.mean([m.predict_proba(X) for m in self.members], axis=0)

    def predict(self, x):
        p = self.predict_proba(x)[0]
        if self.task == "joystick":
            return None, 1.0, {'joy_x': float(p[0]), 'joy_y': float(p[1])}
        k = int(np.argmax(p))
        return self.classes[k], float(p[k]), {c: float(v) for c, v in zip(self.classes, p)}

    def n_params(self) -> int:
        return int(sum(m.n_params() for m in self.members))

    def summary(self) -> str:
        return (f"ENSEMBLE of {len(self.members)} networks (their answers are averaged), each:\n"
                + self.members[0].summary()
                + f"\nAll {len(self.members)} together: {self.n_params()} parameters")

    def save(self, path: str) -> None:
        data = {}
        for k, m in enumerate(self.members):
            data.update({f"m{k}_W{i}": w for i, w in enumerate(m.W)})
            data.update({f"m{k}_b{i}": b for i, b in enumerate(m.b)})
        info = {'kind': 'ensemble', 'members': len(self.members), 'task': self.task,
                'n_inputs': self.n_inputs, 'hidden': list(self.hidden), 'classes': self.classes,
                'meta': self.meta, 'saved': time.strftime('%Y-%m-%d %H:%M:%S')}
        np.savez(path, info=json.dumps(info), **data)


def load_model(path: str):
    """ml_model.npz -> MLP, Ensemble or LSTMNet (whichever was saved)."""
    with np.load(path, allow_pickle=False) as z:
        info = json.loads(str(z['info']))
        arrays = {k: z[k] for k in z.files if k != 'info'}
    if info.get('kind') == 'lstm':
        return LSTMNet(info, arrays)

    def mlp(prefix=""):
        m = MLP(info['n_inputs'], info['hidden'], info['classes'], task=info.get('task', 'classify'))
        m.W = [arrays[f"{prefix}W{i}"] for i in range(len(m.hidden) + 1)]
        m.b = [arrays[f"{prefix}b{i}"] for i in range(len(m.hidden) + 1)]
        m.meta = info.get('meta', {})
        return m

    if info.get('kind') == 'ensemble':
        e = Ensemble([mlp(f"m{k}_") for k in range(int(info['members']))])
        e.meta = info.get('meta', {})
        return e
    return mlp()
