"""Hodgkin-Huxley membrane patch.

Conventions: time in ms, voltage in mV, current density in µA/cm²,
conductance in mS/cm², capacitance in µF/cm². Resting potential near -65 mV.
The equations are integrated with forward Euler so that each line of the
loop corresponds to one term of the model.
"""
import numpy as np

#: Squid giant axon parameters (Hodgkin & Huxley 1952, modern sign convention)
SQUID = {"C": 1.0, "gNa": 120.0, "gK": 36.0, "gL": 0.3,
         "ENa": 50.0, "EK": -77.0, "EL": -54.387}


def _vtrap(x, k):
    """x / (1 - exp(-x/k)), with its limit k at x = 0 (avoids 0/0 = NaN
    when V is exactly -40 mV for m or -55 mV for n)."""
    if np.ndim(x) == 0:
        return k if abs(x) < 1e-9 else x / (1.0 - np.exp(-x / k))
    x = np.asarray(x, float)
    safe = np.where(np.abs(x) < 1e-9, 1.0, x)
    return np.where(np.abs(x) < 1e-9, k, safe / (1.0 - np.exp(-safe / k)))


def gate_rates(V):
    """Return the opening (alpha) and closing (beta) rates, in 1/ms,
    of the sodium activation (m), sodium inactivation (h) and potassium
    activation (n) gates at membrane potential V."""
    alpha_m = 0.1 * _vtrap(V + 40.0, 10.0)
    beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)
    alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    alpha_n = 0.01 * _vtrap(V + 55.0, 10.0)
    beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)
    return (alpha_m, beta_m), (alpha_h, beta_h), (alpha_n, beta_n)


def steady_state(V):
    """Steady-state open fraction x_inf = alpha / (alpha + beta) for m, h, n."""
    return tuple(a / (a + b) for a, b in gate_rates(V))


def simulate_hh(stimulus, duration=100.0, dt=0.01, params=None, V0=-65.0, init=None):
    """Simulate one isopotential Hodgkin-Huxley patch.

    Parameters
    ----------
    stimulus : float or callable
        Injected current density (µA/cm²), constant or a function of time.
    duration, dt : float
        Simulated time and integration step (ms).
    params : dict, optional
        Overrides for :data:`SQUID`, e.g. ``{"gNa": 84}`` for a 30 % sodium block.
    V0 : float
        Initial voltage; gates start at their steady state for V0.
    init : dict, optional
        Full initial state ``{"V", "m", "h", "n"}`` (see :func:`settle`).

    Returns
    -------
    dict with arrays ``t, V, m, h, n, I_Na, I_K, I_L``.
    """
    p = dict(SQUID, **(params or {}))
    drive = stimulus if callable(stimulus) else (lambda _t: stimulus)
    n_steps = int(round(duration / dt))
    t = np.arange(n_steps) * dt
    V, m, h, n = (np.empty(n_steps) for _ in range(4))
    if init is None:
        V[0] = V0
        m[0], h[0], n[0] = steady_state(V0)
    else:
        V[0], m[0], h[0], n[0] = init["V"], init["m"], init["h"], init["n"]
    for i in range(n_steps - 1):
        (am, bm), (ah, bh), (an, bn) = gate_rates(V[i])
        i_na = p["gNa"] * m[i] ** 3 * h[i] * (V[i] - p["ENa"])   # fast inward
        i_k = p["gK"] * n[i] ** 4 * (V[i] - p["EK"])            # delayed outward
        i_l = p["gL"] * (V[i] - p["EL"])                        # passive leak
        V[i + 1] = V[i] + dt * (drive(t[i]) - i_na - i_k - i_l) / p["C"]
        m[i + 1] = m[i] + dt * (am * (1.0 - m[i]) - bm * m[i])
        h[i + 1] = h[i] + dt * (ah * (1.0 - h[i]) - bh * h[i])
        n[i + 1] = n[i] + dt * (an * (1.0 - n[i]) - bn * n[i])
    return {"t": t, "V": V, "m": m, "h": h, "n": n,
            "I_Na": p["gNa"] * m ** 3 * h * (V - p["ENa"]),
            "I_K": p["gK"] * n ** 4 * (V - p["EK"]),
            "I_L": p["gL"] * (V - p["EL"])}


def spike_times(t, V, threshold=0.0):
    """Upward crossings of `threshold` (mV)."""
    up = (V[1:] >= threshold) & (V[:-1] < threshold)
    return t[1:][up]


def hh_rate(current, params=None, duration=600.0, transient=100.0, dt=0.01):
    """Sustained firing rate (Hz) for a constant current, ignoring the transient."""
    r = simulate_hh(current, duration, dt, params)
    s = spike_times(r["t"], r["V"])
    return np.sum(s > transient) / ((duration - transient) / 1000.0)


def settle(params=None, duration=300.0, dt=0.01):
    """Run the unstimulated patch for `duration` ms and return its final
    state, so that experiments start from the cell's own resting condition
    (important when a parameter change moves the resting potential)."""
    r = simulate_hh(0.0, duration, dt, params)
    return {k: r[k][-1] for k in ("V", "m", "h", "n")}
