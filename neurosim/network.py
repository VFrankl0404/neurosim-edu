"""Random recurrent network of excitatory (E) and inhibitory (I) LIF cells.

Current-based delta synapses: a presynaptic spike instantly shifts the
postsynaptic voltage by +J (excitatory) or -g·J (inhibitory). Each cell also
receives independent Poisson input from outside the network. The model
follows the spirit of Brunel (2000) without synaptic delays.
"""
import numpy as np

from .point import LIF_CELL


def simulate_ei_network(n_exc=800, n_inh=200, p_connect=0.1, J=0.2, g=5.0,
                        ext_rate=7.0, duration=500.0, dt=0.1, seed=0, params=None):
    """Simulate the network.

    J        : excitatory jump (mV); inhibitory jump is -g·J.
    g        : relative strength of inhibition.
    ext_rate : external Poisson events per cell per ms (kHz); each event adds J.

    Returns dict with ``times, ids`` (spike raster), ``n_exc, n_cells, duration``.
    """
    p = dict(LIF_CELL, **(params or {}))
    rng = np.random.default_rng(seed)
    n = n_exc + n_inh
    W = (rng.random((n, n)) < p_connect).astype(float)   # W[post, pre]
    np.fill_diagonal(W, 0.0)
    W[:, :n_exc] *= J
    W[:, n_exc:] *= -g * J
    V = rng.uniform(p["V_reset"], p["V_th"], n)          # random initial state
    ready_at = np.zeros(n)
    spiked = np.zeros(n, bool)
    times, ids = [], []
    for i in range(int(round(duration / dt))):
        now = i * dt
        recurrent = W @ spiked
        external = J * rng.poisson(ext_rate * dt, n)
        free = now >= ready_at
        V[free] += dt * (p["E_L"] - V[free]) / p["tau_m"] + recurrent[free] + external[free]
        V[~free] = p["V_reset"]
        spiked = V >= p["V_th"]
        if spiked.any():
            k = np.flatnonzero(spiked)
            ids.append(k)
            times.append(np.full(k.size, now))
            V[k] = p["V_reset"]
            ready_at[k] = now + p["t_ref"]
    return {"times": np.concatenate(times) if times else np.array([]),
            "ids": np.concatenate(ids) if ids else np.array([], int),
            "n_exc": n_exc, "n_cells": n, "duration": duration}


def network_statistics(net, t_start=100.0, bin_ms=1.0, every=5):
    """Mean rate (Hz), mean CV of inter-spike intervals and a synchrony index
    (variance of the population rate divided by its mean, a Fano factor)."""
    keep = net["times"] >= t_start
    times, ids = net["times"][keep], net["ids"][keep]
    window = (net["duration"] - t_start) / 1000.0
    rate = times.size / net["n_cells"] / window
    cvs = []
    for k in range(0, net["n_cells"], every):
        s = times[ids == k]
        if s.size > 3:
            isi = np.diff(s)
            cvs.append(isi.std() / isi.mean())
    counts, _ = np.histogram(times, bins=np.arange(t_start, net["duration"] + bin_ms, bin_ms))
    fano = counts.var() / counts.mean() if counts.mean() > 0 else np.nan
    return {"rate_hz": rate, "cv_isi": float(np.mean(cvs)) if cvs else np.nan,
            "synchrony": float(fano)}
