"""Figures for the practice workbook (Supplementary File S1): the expected
outcome of one key exercise per module, so learners can check their work.

    python make_supplement_figures.py            # English  -> figures_supp/
    python make_supplement_figures.py --lang es  # Spanish  -> figures_supp_es/
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import neurosim as ns
from make_figures import P, CLASS_COLOURS, style, save, tag

T = {
 "en": dict(t="Time (ms)", V="V (mV)", rate="Rate (Hz)", I="Current (µA/cm²)", Ina="Current (nA)",
            control="control", ttx="TTX (gNa = 0)", halfk="gK halved", theory="exact", acc="Test accuracy",
            g="Relative inhibition g", sync="Synchrony index", ninh="Inhibitory cells", c="Reset c (mV)",
            burst="Spikes in first burst", model="Model", ko="[K⁺]o (mM)", vm="Mean V (mV)",
            two="2 features", five="5 features", lin="linear hidden", relu="ReLU hidden", logreg="logistic",
            dt="Δt (ms)", shift="First-spike time shift (ms)", spikes="Spikes in 100 ms"),
 "es": dict(t="Tiempo (ms)", V="V (mV)", rate="Frecuencia (Hz)", I="Corriente (µA/cm²)", Ina="Corriente (nA)",
            control="control", ttx="TTX (gNa = 0)", halfk="gK a la mitad", theory="exacta", acc="Exactitud de prueba",
            g="Inhibición relativa g", sync="Índice de sincronía", ninh="Células inhibitorias", c="Reinicio c (mV)",
            burst="Disparos en la primera ráfaga", model="Modelo", ko="[K⁺]o (mM)", vm="V media (mV)",
            two="2 rasgos", five="5 rasgos", lin="ocultas lineales", relu="ocultas ReLU", logreg="logística",
            dt="Δt (ms)", shift="Desplazamiento del primer disparo (ms)", spikes="Disparos en 100 ms"),
}


def s1(out, L, R):
    pulse = lambda t: 10.0 if 10 <= t < 60 else 0.0
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for prm, lab, c, ls in ((None, L["control"], P["wine"], "-"), ({"gNa": 0.0}, L["ttx"], P["grey"], "--"),
                            ({"gK": 18.0}, L["halfk"], P["teal"], ":")):
        r = ns.simulate_hh(pulse, 80, params=prm)
        ax[0].plot(r["t"], r["V"], color=c, ls=ls, lw=1.1, label=lab)
    ax[0].set(xlabel=L["t"], ylabel=L["V"]); ax[0].legend(fontsize=6.5); tag(ax[0], "A")
    amps = np.arange(5.0, 10.01, 0.25); fired = []
    for a in amps:
        r = ns.simulate_hh(lambda t, a=a: a if 5 <= t < 6 else 0.0, 30)
        fired.append(ns.spike_times(r["t"], r["V"]).size > 0)
    thr = float(amps[np.argmax(fired)])
    ax[1].step(amps, np.array(fired, int), where="post", color=P["indigo"])
    ax[1].axvline(thr, color=P["rose"], lw=0.8); ax[1].text(thr + 0.1, 0.5, f"{thr:g} µA/cm²", fontsize=7, color=P["rose"])
    ax[1].set(xlabel="1-ms pulse amplitude (µA/cm²)" if L is T["en"] else "Amplitud del pulso de 1 ms (µA/cm²)",
              ylabel="spike (0/1)" if L is T["en"] else "disparo (0/1)", yticks=[0, 1]); tag(ax[1], "B")
    R["S1_brief_pulse_threshold"] = thr
    fig.tight_layout(); save(fig, out, "figS1_module1")


def s2(out, L, R):
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.6))
    I = np.linspace(0, 5, 41)
    ax.plot(I, ns.lif_rate_closed_form(I), color=P["ink"], lw=1.4, label=L["theory"])
    for dt, c, m in ((0.05, P["indigo"], "s"), (0.5, P["olive"], "^"), (1.0, P["rose"], "o")):
        sim = [ns.simulate_lif(i, 2000, dt)["spikes"].size / 2 for i in I]
        ax.plot(I, sim, m, ms=3, mfc="white", color=c, label=f"Δt = {dt} ms")
    ax.set(xlabel=L["Ina"], ylabel=L["rate"]); ax.legend(fontsize=6.5)
    fig.tight_layout(); save(fig, out, "figS2_module2")


def s3(out, L, R):
    a, b, _, d = ns.IZH_CLASSES["RS"]["abcd"]
    cs = [-65, -60, -55, -50]
    fig, ax = plt.subplots(len(cs), 1, figsize=(7.2, 4.2), sharex=True)
    first = {}
    for axx, c, col in zip(ax, cs, CLASS_COLOURS):
        r = ns.izhikevich_cell(a, b, c, d, lambda t: 10.0 * (t >= 20), 200)
        axx.plot(r["t"], r["v"], color=col, lw=0.8)
        isi = np.diff(r["spikes"]); first[c] = int(1 + np.argmax(isi > 15)) if isi.size and np.any(isi > 15) else int(r["spikes"].size)
        axx.set_title(f"c = {c} mV", fontsize=7.5, loc="left"); axx.set_ylim(-85, 38)
    ax[-1].set_xlabel(L["t"]); ax[1].set_ylabel(L["V"])
    R["S3_first_burst_spikes_by_c"] = first
    fig.tight_layout(h_pad=0.3); save(fig, out, "figS3_module3")


def s4(out, L, R):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ninh = [200, 150, 100, 50]
    res = []
    for n_i in ninh:
        vals = [ns.network_statistics(ns.simulate_ei_network(n_inh=n_i, g=6.0, ext_rate=7.0, duration=400, seed=s)) for s in range(3)]
        res.append([np.mean([v["rate_hz"] for v in vals]), np.mean([v["synchrony"] for v in vals])])
    res = np.array(res)
    ax[0].plot(ninh, res[:, 0], "o-", color=P["indigo"]); ax[0].invert_xaxis()
    ax[0].set(xlabel=L["ninh"], ylabel=L["rate"]); tag(ax[0], "A")
    ax[1].semilogy(ninh, res[:, 1], "D-", color=P["wine"]); ax[1].invert_xaxis()
    ax[1].set(xlabel=L["ninh"], ylabel=L["sync"]); tag(ax[1], "B")
    R["S4_interneuron_loss"] = {"n_inh": ninh, "rate": res[:, 0].tolist(), "synchrony": res[:, 1].tolist()}
    fig.tight_layout(); save(fig, out, "figS4_module4")


def s5(out, L, R):
    X, y, _ = ns.make_dataset(seed=0)
    variants = {L["five"] + " · " + L["relu"]: (slice(None), (16, 16)),
                L["two"] + " · " + L["relu"]: ([3, 1], (16, 16)),
                L["five"] + " · " + L["logreg"]: (slice(None), ())}
    scores = {k: [] for k in variants}
    for s in range(10):
        tr, te = ns.stratified_split(y, 0.3, seed=s)
        for k, (cols, hid) in variants.items():
            Xc = X[:, cols]
            net = ns.SpikeClassNet(Xc.shape[1], 5, hid, seed=s); net.fit(Xc[tr], y[tr], epochs=150, seed=s)
            scores[k].append(np.mean(net.predict(Xc[te]) == y[te]))
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    names = list(scores)
    for i, k in enumerate(names):
        v = np.array(scores[k]); jitter = np.linspace(-0.12, 0.12, v.size)
        ax.scatter(np.full(v.size, i) + jitter, v, s=10, color=[P["indigo"], P["olive"], P["teal"]][i], alpha=0.8, lw=0)
        ax.hlines(v.mean(), i - 0.25, i + 0.25, color=P["ink"], lw=1.4)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=6.5); ax.set_ylabel(L["acc"])
    R["S5_feature_ablation"] = {k: [(c := ns.corrected_resampled_t(v, 420, 180))["mean"], c["lower"], c["upper"]]
                                for k, v in scores.items()}   # overlap-corrected 95% CI
    fig.tight_layout(); save(fig, out, "figS5_module5")


def s6(out, L, R):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for K, c in ((2.5, P["cyan"]), (4.0, P["green"])):
        p = ns.extracellular_potassium(K)
        r = ns.simulate_hh(lambda t: 8.0 if 20 <= t < 120 else 0.0, 150, params=p, init=ns.settle(p))
        ax[0].plot(r["t"], r["V"], color=c, lw=1, label=f"{K} mM, 8 µA/cm²")
    ax[0].set(xlabel=L["t"], ylabel=L["V"]); ax[0].legend(fontsize=6.5); tag(ax[0], "A")
    I = np.arange(0, 30.1, 2.0)
    blocks = np.linspace(0, 0.3, 7)
    onset = []
    for bl in blocks:
        fi = ns.fi_curve(I, ns.sodium_block(bl), duration=300)
        onset.append(I[np.argmax(fi > 0)] if np.any(fi > 0) else np.nan)
    ax[1].plot(blocks * 100, onset, "s-", color=P["wine"])
    ax[1].set(xlabel="Na⁺ block (%)" if L is T["en"] else "Bloqueo de Na⁺ (%)",
              ylabel="Onset current (µA/cm²)" if L is T["en"] else "Corriente de inicio (µA/cm²)")
    tag(ax[1], "B")
    R["S6_na_block_onset"] = dict(zip(map(float, blocks), map(float, onset)))
    fig.tight_layout(); save(fig, out, "figS6_module6")


def s7(out, L, R):
    ref = ns.simulate_hh(10.0, 100, 0.01); ref_s = ns.spike_times(ref["t"], ref["V"])
    dts = [0.01, 0.02, 0.05, 0.1]
    shifts, counts = [], []
    for dt in dts:
        r = ns.simulate_hh(10.0, 100, dt); s = ns.spike_times(r["t"], r["V"])
        counts.append(int(s.size)); shifts.append(float(s[0] - ref_s[0]) if s.size else np.nan)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
    ax[0].plot(dts, shifts, "o-", color=P["indigo"]); ax[0].set(xlabel=L["dt"], ylabel=L["shift"]); tag(ax[0], "A")
    ax[1].bar([str(d) for d in dts], counts, color=P["teal"]); ax[1].set(xlabel=L["dt"], ylabel=L["spikes"]); tag(ax[1], "B")
    R["S7_hh_dt"] = {"dt": dts, "first_spike_shift_ms": shifts, "spikes_100ms": counts}
    fig.tight_layout(); save(fig, out, "figS7_module7")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lang", default="en", choices=["en", "es"])
    lang = ap.parse_args().lang; L = T[lang]
    out = Path("figures_supp" if lang == "en" else "figures_supp_es"); out.mkdir(exist_ok=True)
    style(); R = {}
    for f in (s1, s2, s3, s4, s5, s6, s7):
        f(out, L, R); print(f.__name__, "done", flush=True)
    if lang == "en":
        Path("results").mkdir(exist_ok=True)
        Path("results/supplement_results.json").write_text(json.dumps(R, indent=2, default=float))
        print(json.dumps(R, indent=1, default=float))


if __name__ == "__main__":
    main()
