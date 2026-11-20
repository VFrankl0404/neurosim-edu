"""Clinically motivated "what if" experiments on the teaching models.

These are qualitative illustrations of mechanisms, not patient models. The
squid-axon Hodgkin-Huxley parameters are used throughout, so numbers should
be read as trends (direction and shape of the effect), not as human values.
"""
import numpy as np

from .hh import SQUID, hh_rate, settle, simulate_hh, spike_times

RT_OVER_F_37C = 26.7   # mV, RT/F at 37 °C


def sodium_block(fraction_blocked):
    """Parameter override for a use-independent block of Na+ channels,
    the main action of local anaesthetics (lidocaine) and of several
    antiseizure drugs (phenytoin, carbamazepine)."""
    return {"gNa": SQUID["gNa"] * (1.0 - fraction_blocked)}


def potassium_block(fraction_blocked):
    """Override for a block of delayed-rectifier K+ channels, a simplified
    stand-in for 4-aminopyridine (dalfampridine), used to improve conduction
    in multiple sclerosis."""
    return {"gK": SQUID["gK"] * (1.0 - fraction_blocked)}


def extracellular_potassium(K_out_mM, K_out_ref=4.0):
    """Shift E_K according to the Nernst equation when extracellular K+
    changes from the reference value (hyperkalaemia if K_out > 4 mM)."""
    return {"EK": SQUID["EK"] + RT_OVER_F_37C * np.log(K_out_mM / K_out_ref)}


def fi_curve(currents, params=None, duration=500.0):
    """Firing rate (Hz) for each constant current in `currents`."""
    return np.array([hh_rate(I, params, duration=duration) for I in currents])


def spike_width(params=None, pulse_amp=20.0, level=-20.0):
    """Width (ms) of the first action potential evoked by a 1 ms pulse,
    measured at `level` mV. Returns the trace too."""
    r = simulate_hh(lambda t: pulse_amp if 5.0 <= t < 6.0 else 0.0,
                    duration=40.0, params=params, init=settle(params))
    up = spike_times(r["t"], r["V"], level)
    down = spike_times(r["t"], -r["V"], -level)          # downward crossings
    width = (down[down > up[0]][0] - up[0]) if up.size and np.any(down > up[0]) else np.nan
    return width, r
