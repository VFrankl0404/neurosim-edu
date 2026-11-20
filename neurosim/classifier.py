"""From spiking neurons to an artificial neural network.

A small multilayer perceptron learns to identify the electrophysiological
class of a simulated Izhikevich neuron (RS, IB, CH, FS, LTS) from five
features of its spike train, the way an electrophysiologist classifies
cells recorded in a slice. Everything (data generation, features, network,
training and evaluation) is written with NumPy only.
"""
import numpy as np

from .point import IZH_CLASSES, simulate_izhikevich

FEATURES = ["log rate", "CV of ISI", "adaptation", "burst fraction", "first-spike latency"]
CLASS_KEYS = list(IZH_CLASSES)


# ----------------------------------------------------------------- dataset
def spike_train_features(spikes, onset=20.0, duration=500.0):
    """Five interpretable descriptors of one spike train.

    log rate            : log10(1 + mean rate in Hz) during the step
    CV of ISI           : irregularity of inter-spike intervals
    adaptation          : (last ISI - first ISI) / (last ISI + first ISI)
    burst fraction      : share of ISIs shorter than 10 ms
    first-spike latency : time from step onset to the first spike (ms, capped)
    """
    s = spikes[spikes >= onset]
    window = (duration - onset) / 1000.0
    rate = s.size / window
    if s.size >= 3:
        isi = np.diff(s)
        cv = isi.std() / isi.mean()
        adapt = (isi[-1] - isi[0]) / (isi[-1] + isi[0])
        burst = np.mean(isi < 10.0)
    else:
        cv, adapt, burst = 0.0, 0.0, 0.0
    latency = (s[0] - onset) if s.size else duration - onset
    return np.array([np.log10(1.0 + rate), cv, adapt, burst, min(latency, 100.0)])


def make_dataset(n_per_class=120, jitter=0.10, current_range=(6.0, 14.0),
                 noise_sd=2.0, duration=500.0, seed=0):
    """Simulate a heterogeneous population of each class and extract features.

    Each cell gets its class parameters multiplied by (1 + jitter·N(0,1)),
    a random step amplitude drawn uniformly from `current_range` (starting at
    20 ms) and white-noise input, so that classes overlap realistically.
    Returns X (n_cells x 5), y (class index) and the raw spike trains.
    """
    rng = np.random.default_rng(seed)
    X, y, trains = [], [], []
    for label, key in enumerate(CLASS_KEYS):
        a, b, c, d = IZH_CLASSES[key]["abcd"]
        n = n_per_class
        scale = lambda v: v * (1.0 + jitter * rng.standard_normal(n))
        amp = rng.uniform(*current_range, n)
        out = simulate_izhikevich(scale(a), scale(b), c + 2.0 * rng.standard_normal(n), scale(d),
                                  lambda t, amp=amp: amp * (t >= 20.0), duration=duration,
                                  dt=0.25, noise_sd=noise_sd, n_cells=n,
                                  seed=int(rng.integers(1 << 31)), record=False)
        for s in out["spikes"]:
            X.append(spike_train_features(s, duration=duration))
            y.append(label)
            trains.append(s)
    return np.array(X), np.array(y), trains


def stratified_split(y, test_fraction=0.3, seed=0):
    """Indices of a class-balanced random train/test split."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for k in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == k))
        n_test = int(round(test_fraction * idx.size))
        test.extend(idx[:n_test]); train.extend(idx[n_test:])
    return np.array(train), np.array(test)


# ----------------------------------------------------------------- network
class SpikeClassNet:
    """Multilayer perceptron with ReLU hidden layers and a softmax output,
    trained by mini-batch gradient descent with momentum and L2 penalty.

    hidden=() gives multinomial logistic regression (no hidden layer), which
    serves as the statistical baseline.
    """

    def __init__(self, n_in, n_out, hidden=(16, 16), seed=0):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, n_out]
        # He initialisation, appropriate for ReLU units
        self.W = [rng.normal(0.0, np.sqrt(2.0 / i), (i, o)) for i, o in zip(sizes, sizes[1:])]
        self.b = [np.zeros(o) for o in sizes[1:]]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.mu, self.sd = None, None

    @property
    def n_params(self):
        return int(sum(w.size + b.size for w, b in zip(self.W, self.b)))

    def _standardise(self, X):
        return (X - self.mu) / self.sd

    def _layers(self, Z):
        """Return the activity of every layer; the last one is class probabilities."""
        acts = [Z]
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b                    # weigh and add
            if k < len(self.W) - 1:
                acts.append(np.maximum(z, 0.0))      # decide: ReLU
            else:
                z = z - z.max(axis=1, keepdims=True)
                e = np.exp(z)
                acts.append(e / e.sum(axis=1, keepdims=True))   # softmax
        return acts

    def predict_proba(self, X):
        return self._layers(self._standardise(X))[-1]

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def fit(self, X, y, epochs=300, batch=32, lr=0.05, momentum=0.9, l2=1e-4,
            X_val=None, y_val=None, seed=0):
        """Train and return a history of training / validation loss and accuracy."""
        rng = np.random.default_rng(seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = self._standardise(X)
        Y = np.eye(self.W[-1].shape[1])[y]
        hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        for _ in range(epochs):
            for idx in np.array_split(rng.permutation(len(Z)), max(1, len(Z) // batch)):
                acts = self._layers(Z[idx])
                delta = (acts[-1] - Y[idx]) / idx.size          # dLoss/dz at the output
                for k in reversed(range(len(self.W))):
                    gW = acts[k].T @ delta + l2 * self.W[k]
                    gb = delta.sum(0)
                    if k:
                        delta = (delta @ self.W[k].T) * (acts[k] > 0)   # back through ReLU
                    self.vW[k] = momentum * self.vW[k] - lr * gW
                    self.vb[k] = momentum * self.vb[k] - lr * gb
                    self.W[k] += self.vW[k]
                    self.b[k] += self.vb[k]
            hist["train_loss"].append(cross_entropy(self.predict_proba(X), y))
            hist["train_acc"].append(np.mean(self.predict(X) == y))
            if X_val is not None:
                hist["val_loss"].append(cross_entropy(self.predict_proba(X_val), y_val))
                hist["val_acc"].append(np.mean(self.predict(X_val) == y_val))
        return hist


# ----------------------------------------------------------------- metrics
def cross_entropy(P, y):
    return float(-np.mean(np.log(P[np.arange(len(y)), y] + 1e-12)))


def confusion_matrix(y_true, y_pred, n_classes):
    M = np.zeros((n_classes, n_classes), int)
    np.add.at(M, (y_true, y_pred), 1)
    return M


def macro_f1(y_true, y_pred, n_classes):
    M = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(M).astype(float)
    precision = tp / np.maximum(M.sum(0), 1)
    recall = tp / np.maximum(M.sum(1), 1)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return float(f1.mean())


def bootstrap_ci(values, n_boot=10000, level=0.95, seed=0):
    """Percentile bootstrap confidence interval of the mean."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    means = rng.choice(values, (n_boot, values.size)).mean(1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(values.mean()), float(lo), float(hi)


# ------------------------------------------------- evaluation (added in v1.1)
# These helpers address dependence between overlapping resampled splits,
# give per-class and calibration metrics, and provide exact paired tests,
# using NumPy and the standard library only.
from math import comb, lgamma, exp, log, sqrt


def _betacf(a, b, x, max_iter=300, eps=3e-14):
    """Continued fraction for the regularised incomplete beta (Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c; c = c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c; c = c if abs(c) > 1e-300 else 1e-300
        delta = d * c; h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x)
    if x < (a + 1.0) / (a + b + 2.0):
        return exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    """Student t cumulative distribution function."""
    x = df / (df + t * t)
    tail = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def t_ppf(q, df):
    """Student t quantile by bisection on :func:`t_cdf`."""
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if t_cdf(mid, df) < q else (lo, mid)
    return 0.5 * (lo + hi)


def corrected_resampled_t(values, n_train, n_test, level=0.95):
    """Nadeau & Bengio (2003) corrected resampled t interval and test.

    For J repeated random train/test splits the estimates are positively
    correlated because training (and test) sets overlap. The naive variance
    s²/J is replaced by (1/J + n_test/n_train)·s². Returns a dict with mean,
    lower, upper, t, df and two-sided p (H0: mean = 0; useful for paired
    differences between models).
    """
    v = np.asarray(values, float)
    J = v.size
    mean, s2 = float(v.mean()), float(v.var(ddof=1))
    se = sqrt((1.0 / J + n_test / n_train) * s2)
    df = J - 1
    q = t_ppf(1 - (1 - level) / 2, df)
    t = mean / se if se > 0 else np.inf
    p = 2.0 * (1.0 - t_cdf(abs(t), df)) if np.isfinite(t) else 0.0
    return {"mean": mean, "lower": mean - q * se, "upper": mean + q * se,
            "se": se, "t": float(t), "df": df, "p": float(p)}


def per_class_metrics(y_true, y_pred, n_classes):
    """Precision, recall (sensitivity), specificity and F1 for each class."""
    M = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(M).astype(float)
    fp, fn = M.sum(0) - tp, M.sum(1) - tp
    tn = M.sum() - tp - fp - fn
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    specificity = tn / np.maximum(tn + fp, 1)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)
    return {"precision": precision, "recall": recall, "specificity": specificity,
            "f1": f1, "support": M.sum(1)}


def brier_score(P, y):
    """Multiclass Brier score: mean squared distance between the predicted
    probability vector and the one-hot outcome (0 = perfect, 2 = worst)."""
    Y = np.eye(P.shape[1])[y]
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def calibration_curve(P, y, n_bins=10):
    """Top-label reliability curve and expected calibration error (ECE).

    Returns (bin_confidence, bin_accuracy, bin_count, ece); empty bins are NaN.
    """
    conf = P.max(1)
    correct = (P.argmax(1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    bc, ba, bn = np.full(n_bins, np.nan), np.full(n_bins, np.nan), np.zeros(n_bins, int)
    for k in range(n_bins):
        m = which == k
        bn[k] = m.sum()
        if bn[k]:
            bc[k], ba[k] = conf[m].mean(), correct[m].mean()
    ece = float(np.nansum(bn / bn.sum() * np.abs(ba - bc)))
    return bc, ba, bn, ece


def mcnemar_exact(correct_a, correct_b):
    """Exact (binomial) McNemar test for paired predictions of two models.

    Returns n01 (A wrong, B right), n10 (A right, B wrong) and the two-sided p.
    """
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = n10 + n01
    if n == 0:
        return {"n10": n10, "n01": n01, "p": 1.0}
    k = min(n10, n01)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n)
    return {"n10": n10, "n01": n01, "p": float(p)}


def stratified_bootstrap(y, stat, n_boot=2000, level=0.95, seed=0):
    """Percentile CI of `stat(idx)` resampling cells with replacement within
    each class (the test cells, not the splits, are the sampling unit)."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == k) for k in np.unique(y)]
    full = np.arange(len(y))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(g, g.size) for g in groups])
        boots[b] = stat(idx)
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(stat(full)), float(lo), float(hi)
