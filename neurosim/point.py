"""Point-neuron models: leaky integrate-and-fire (LIF) and Izhikevich.

LIF units: ms, mV, nA, MΩ (so R·I is in mV).
Izhikevich units: ms and mV, with a dimensionless input current.
"""
import numpy as np

#: A generic cortical LIF cell (values within the range reported by Gerstner et al. 2014)
LIF_CELL = {"tau_m": 10.0, "E_L": -65.0, "R_m": 10.0,
            "V_th": -50.0, "V_reset": -65.0, "t_ref": 2.0}


def simulate_lif(stimulus, duration=100.0, dt=0.1, params=None, spike_height=20.0):
    """tau_m dV/dt = -(V - E_L) + R_m I. When V crosses V_th a spike is
    recorded, V is reset to V_reset and held there for t_ref ms.

    For display only, the sample at the crossing is drawn at `spike_height`.
    Returns dict with ``t, V, spikes``.
    """
    p = dict(LIF_CELL, **(params or {}))
    drive = stimulus if callable(stimulus) else (lambda _t: stimulus)
    n_steps = int(round(duration / dt))
    t = np.arange(n_steps) * dt
    V = np.full(n_steps, p["E_L"])
    spikes = []
    hold_until = -1.0
    for i in range(n_steps - 1):
        if t[i] < hold_until:                       # absolute refractory period
            V[i + 1] = p["V_reset"]
            continue
        dV = (-(V[i] - p["E_L"]) + p["R_m"] * drive(t[i])) / p["tau_m"]
        V[i + 1] = V[i] + dt * dV
        if V[i + 1] >= p["V_th"]:
            spikes.append(t[i + 1])
            V[i + 1] = spike_height
            hold_until = t[i + 1] + p["t_ref"]
    return {"t": t, "V": V, "spikes": np.asarray(spikes)}


def lif_rate_closed_form(current, params=None):
    """Exact firing rate (Hz) of the LIF cell for a constant current (nA)."""
    p = dict(LIF_CELL, **(params or {}))
    current = np.asarray(current, dtype=float)
    drive = p["R_m"] * current
    gap = p["V_th"] - p["E_L"]
    rate = np.zeros_like(current)
    above = drive > gap
    period = p["t_ref"] + p["tau_m"] * np.log(drive[above] / (drive[above] - gap))
    rate[above] = 1000.0 / period
    return rate


#: Izhikevich (2003) parameter sets (a, b, c, d) for five cortical cell classes
IZH_CLASSES = {
    "RS": {"name": "Regular spiking", "abcd": (0.02, 0.20, -65.0, 8.0)},
    "IB": {"name": "Intrinsically bursting", "abcd": (0.02, 0.20, -55.0, 4.0)},
    "CH": {"name": "Chattering", "abcd": (0.02, 0.20, -50.0, 2.0)},
    "FS": {"name": "Fast spiking", "abcd": (0.10, 0.20, -65.0, 2.0)},
    "LTS": {"name": "Low-threshold spiking", "abcd": (0.02, 0.25, -65.0, 2.0)},
}


def simulate_izhikevich(a, b, c, d, stimulus, duration=300.0, dt=0.1,
                        noise_sd=0.0, n_cells=1, seed=None, record=True):
    """Vectorised Izhikevich model for one or many cells.

        dv/dt = 0.04 v² + 5 v + 140 - u + I
        du/dt = a (b v - u)
        if v >= 30 mV:  v <- c,  u <- u + d

    a, b, c, d and `stimulus` may be scalars or arrays of length `n_cells`;
    `stimulus` may also be a callable of time. `noise_sd` adds white-noise
    current (units of I per sqrt(ms)).

    Returns dict with ``t``, ``v`` (n_cells x steps, if record), and
    ``spikes`` (list of spike-time arrays, one per cell).
    """
    rng = np.random.default_rng(seed)
    a, b, c, d = (np.broadcast_to(np.asarray(x, float), (n_cells,)) for x in (a, b, c, d))
    drive = stimulus if callable(stimulus) else (lambda _t: stimulus)
    n_steps = int(round(duration / dt))
    t = np.arange(n_steps) * dt
    v = np.full(n_cells, -65.0)
    u = b * v
    trace = np.empty((n_cells, n_steps)) if record else None
    fired_at = [[] for _ in range(n_cells)]
    for i in range(n_steps):
        I = np.broadcast_to(np.asarray(drive(t[i]), float), (n_cells,))
        if noise_sd:
            I = I + noise_sd * rng.standard_normal(n_cells) / np.sqrt(dt)
        peak = v >= 30.0
        if record:
            trace[:, i] = np.where(peak, 30.0, v)
        for k in np.flatnonzero(peak):
            fired_at[k].append(t[i])
        v = np.where(peak, c, v)
        u = np.where(peak, u + d, u)
        dv = 0.04 * v ** 2 + 5.0 * v + 140.0 - u + I
        du = a * (b * v - u)
        v = v + dt * dv
        u = u + dt * du
    return {"t": t, "v": trace, "spikes": [np.asarray(s) for s in fired_at]}


def izhikevich_cell(a, b, c, d, stimulus, duration=300.0, dt=0.1):
    """Single-cell, scalar version of the Izhikevich model: the clearest
    form for reading the equations (and the fastest for one cell)."""
    drive = stimulus if callable(stimulus) else (lambda _t: stimulus)
    n_steps = int(round(duration / dt))
    t = np.arange(n_steps) * dt
    v_trace = np.empty(n_steps)
    v, u = -65.0, b * -65.0
    spikes = []
    for i in range(n_steps):
        if v >= 30.0:                       # spike: record, then reset
            v_trace[i] = 30.0
            spikes.append(t[i])
            v, u = c, u + d
        else:
            v_trace[i] = v
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + drive(t[i])
        du = a * (b * v - u)
        v, u = v + dt * dv, u + dt * du
    return {"t": t, "v": v_trace, "spikes": np.asarray(spikes)}
