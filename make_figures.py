"""Regenerate every figure and numerical result reported in the manuscript.

    python make_figures.py                  # English labels  -> figures/
    python make_figures.py --lang es        # Spanish labels  -> figures_es/
    python make_figures.py --out some/dir   # write figures (and results.json) elsewhere

Numerical results are written to results/results.json (English run only) and,
with --out, also next to the figures. Figures are drawn at their final print
size (double column 175 mm), with every text >= 6.5 pt, and saved as PNG
(600 dpi) and vector PDF (TrueType fonts embedded, no creation date so that
re-runs are byte-stable). Requires NumPy and Matplotlib only.

Review version (1.1.0-review): see CHANGES.md for the list of changes.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text

import neurosim as ns

MM = 1 / 25.4
DOUBLE, SINGLE = 175 * MM, 85 * MM          # journal column widths (inches)

# Paul Tol "muted" qualitative palette (colour-blind safe). "sand" and "cyan"
# are used only as fills: as lines or text on white their WCAG contrast is < 2:1.
P = {"indigo": "#332288", "cyan": "#88CCEE", "teal": "#44AA99", "green": "#117733",
     "olive": "#999933", "sand": "#DDCC77", "rose": "#CC6677", "wine": "#882255",
     "purple": "#AA4499", "ink": "#222222", "grey": "#777777", "paper": "#F7F5EF"}
CLASS_COLOURS = [P["indigo"], P["rose"], P["olive"], P["teal"], P["purple"]]
CLASS_MARKERS = ["o", "s", "^", "D", "v"]

T = {
 "en": dict(zoom="repolarisation (zoom)", t="Time (ms)", V="Membrane potential (mV)", Iua="Injected current (µA/cm²)",
   Ina="Injected current (nA)", rate="Firing rate (Hz)", gate="Open fraction",
   stim="Stimulus", theory="Closed form", sim="Euler", dt="Time step Δt (ms)",
   err="Max. |rate error| (Hz)", neuron="Neuron", pop="Population\nrate (Hz)",
   g="Relative inhibition g", cv="CV of ISIs", sync="Synchrony index\n(Fano factor)",
   weak="Weak inhibition (g = 2)", strong="Strong inhibition (g = 6)",
   epoch="Epoch", loss="Cross-entropy loss", acc="Accuracy", train="training", test="test",
   true="True class", pred="Predicted class", feat_x="Burst fraction", feat_y="CV of ISIs",
   na="Na⁺ channel block (lidocaine-like)", k="K⁺ channel block (4-AP-like)",
   hyperk="Extracellular K⁺", hyperk_sum="Hyperkalaemia: mean potential and firing",
   Vrest="Mean potential (mV)", ko="[K⁺]ₒ (mM)", blocked="blocked",
   learn=["Ion channels", "Threshold & reset", "Firing patterns", "Circuits",
          "Learning machines", "Clinical what-ifs"],
   learn_sub=["Hodgkin-Huxley", "LIF", "Izhikevich", "E/I network", "Perceptron", "Drugs, K⁺, E/I"],
   axis_l="more biological detail", axis_r="more abstraction, lower cost", applies="applies\nmodules 1 and 4",
   iz_names={k: v["name"] for k, v in ns.IZH_CLASSES.items()},
   isi_err="ISI-based rate", cnt_err="count-based rate", resolution="count resolution (0.5 Hz)",
   slope1="slope 1", mlp="MLP", lr="LR", split="repeated\nsplits (×10)", indep="independent\ntest set",
   confidence="Predicted probability (top class)", frac_correct="Observed accuracy",
   perfect="perfect calibration", exc="excitatory", inh="inhibitory", width="width at −20 mV",
   rest_firing="spontaneous firing", seeds="single seeds", mean_sd="mean ± SD",
   stepnote="Step I = 10 from 20 ms (bar); Δt = 0.1 ms", cells0="{n} cells at 0",
   median_range="median, range", binnote="bins with n ≥ 10\ntop bin: {m} ({mn}),\n{l} ({ln}) of {n}"),
 "es": dict(rate_short="Frec. (Hz)", zoom="repolarización (ampliación)", t="Tiempo (ms)", V="Potencial de membrana (mV)", Iua="Corriente inyectada (µA/cm²)",
   Ina="Corriente inyectada (nA)", rate="Frecuencia de disparo (Hz)", gate="Fracción abierta",
   stim="Estímulo", theory="Solución exacta", sim="Euler", dt="Paso de tiempo Δt (ms)",
   err="Error máx. |frecuencia| (Hz)", neuron="Neurona", pop="Frecuencia\npoblacional (Hz)",
   g="Inhibición relativa g", cv="CV de los ISI", sync="Índice de sincronía\n(factor de Fano)",
   weak="Inhibición débil (g = 2)", strong="Inhibición fuerte (g = 6)",
   epoch="Época", loss="Pérdida (entropía cruzada)", acc="Exactitud", train="entrenamiento", test="prueba",
   true="Clase real", pred="Clase predicha", feat_x="Fracción en ráfaga", feat_y="CV de los ISI",
   na="Bloqueo de canales de Na⁺ (tipo lidocaína)", k="Bloqueo de canales de K⁺ (tipo 4-AP)",
   hyperk="K⁺ extracelular", hyperk_sum="Hiperpotasemia: potencial medio y disparo",
   Vrest="Potencial medio (mV)", ko="[K⁺]ₒ (mM)", blocked="bloqueado",
   learn=["Canales iónicos", "Umbral y reinicio", "Patrones de disparo", "Circuitos",
          "Máquinas que aprenden", "Escenarios clínicos"],
   learn_sub=["Hodgkin-Huxley", "LIF", "Izhikevich", "Red E/I", "Perceptrón", "Fármacos, K⁺, E/I"],
   axis_l="más detalle biológico", axis_r="más abstracción, menor costo", applies="aplica los\nmódulos 1 y 4",
   iz_names={"RS": "Disparo regular", "IB": "Ráfagas intrínsecas", "CH": "Ráfagas rápidas (chattering)",
             "FS": "Disparo rápido", "LTS": "Umbral bajo"},
   isi_err="frecuencia por ISI", cnt_err="frecuencia por conteo", resolution="resolución del conteo (0,5 Hz)",
   slope1="pendiente 1", mlp="MLP", lr="RL", split="divisiones\nrepetidas (×10)", indep="prueba\nindependiente",
   confidence="Probabilidad predicha (clase elegida)", frac_correct="Exactitud observada",
   perfect="calibración perfecta", exc="excitadoras", inh="inhibidoras", width="ancho a −20 mV",
   rest_firing="disparo espontáneo", seeds="semillas individuales", mean_sd="media ± DE",
   stepnote="Escalón I = 10 desde 20 ms (barra); Δt = 0,1 ms", cells0="{n} células en 0",
   median_range="mediana, rango", binnote="intervalos con n ≥ 10\nintervalo superior: {m} ({mn}),\n{l} ({ln}) de {n}"),
}


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 600, "font.size": 7, "axes.titlesize": 7.5,
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "legend.title_fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#444444", "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "legend.frameon": False, "font.family": "DejaVu Sans",
        "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 1.0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": True})


def tag(ax, s, x=-0.16, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom",
            ha="left", color=P["ink"])


def minus(v):
    """Number with a true minus sign (U+2212), for tables drawn inside figures."""
    return f"{v:g}".replace("-", "−")


AUDIT = {}


def save(fig, out, name):
    """Save at the exact figure size (no tight cropping) and record the smallest text size."""
    fig.canvas.draw()
    sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    w, h = fig.get_size_inches()
    AUDIT[name] = {"width_mm": round(w / MM, 1), "height_mm": round(h / MM, 1),
                   "min_font_pt": float(min(sizes)), "png_dpi": plt.rcParams["savefig.dpi"]}
    fig.savefig(out / f"{name}.png", facecolor="white", metadata={"Software": None})
    fig.savefig(out / f"{name}.pdf", facecolor="white", metadata={"CreationDate": None})
    plt.close(fig)


def stim_bar(ax, t, I, ymin, height, colour):
    """Draw the stimulus time course as a small step trace under a voltage plot."""
    I = np.asarray(I, float)
    scale = height / max(I.max(), 1e-9)
    ax.fill_between(t, ymin, ymin + I * scale, step="post", color=colour, alpha=0.8, lw=0)


def break_jumps(x, y):
    """Insert NaN between a zero and a non-zero rate so that a discontinuous
    f-I curve is not drawn as a steep but continuous line."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    xs, ys = [x[0]], [y[0]]
    for i in range(1, len(x)):
        if (y[i - 1] == 0) != (y[i] == 0):
            xs.append(np.nan); ys.append(np.nan)
        xs.append(x[i]); ys.append(y[i])
    return np.array(xs), np.array(ys)


def contrast(hex_colour, bg="#FFFFFF"):
    """WCAG 2 contrast ratio of a colour against a background."""
    def lum(c):
        c = np.array(to_rgb(c))
        c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    a, b = sorted((lum(hex_colour), lum(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


# ===================================================================== Fig 1
def fig_overview(out, L):
    fig = plt.figure(figsize=(DOUBLE, 78 * MM))
    xs = np.linspace(0.015, 0.845, 6)
    w = 0.14
    cols = [P["wine"], P["indigo"], P["purple"], P["teal"], P["olive"], P["rose"]]
    hh = ns.simulate_hh(lambda t: 10.0 if t >= 5 else 0.0, 40)
    lif = ns.simulate_lif(lambda t: 2.5 if t >= 5 else 0.0, 60, 0.05)
    iz = ns.simulate_izhikevich(*ns.IZH_CLASSES["CH"]["abcd"], lambda t: 10.0 * (t >= 10), 150)
    net = ns.simulate_ei_network(g=2.0, ext_rate=8.0, duration=200, seed=3)
    for i, x in enumerate(xs):
        card = FancyBboxPatch((x, 0.24), w, 0.72, boxstyle="round,pad=0.004,rounding_size=0.015",
                              transform=fig.transFigure, fc=P["paper"], ec=cols[i], lw=1.2, zorder=0)
        fig.add_artist(card)
        title = L["learn"][i]
        if len(title) > 12:
            title = title.replace(" ", "\n", 1)
        fig.text(x + w / 2, 0.885, f"{i + 1}. {title}", ha="center", va="center", fontsize=7,
                 fontweight="bold", color=P["ink"], linespacing=1.0)
        fig.text(x + w / 2, 0.80, L["learn_sub"][i], ha="center", va="center", fontsize=6.5, color=P["ink"])
        ax = fig.add_axes([x + 0.01, 0.30, w - 0.02, 0.44], zorder=2)
        ax.set_axis_off(); ax.patch.set_alpha(0)
        if i == 0:
            ax.plot(hh["t"], hh["V"], color=cols[i], lw=1)
        elif i == 1:
            ax.plot(lif["t"], lif["V"], color=cols[i], lw=0.9)
            ax.axhline(-50, ls=":", color=P["grey"], lw=0.7)
        elif i == 2:
            ax.plot(iz["t"], iz["v"][0], color=cols[i], lw=0.7)
        elif i == 3:
            m = (net["times"] > 120) & (net["ids"] % 8 == 0)
            ax.scatter(net["times"][m], net["ids"][m], s=0.6,
                       c=np.where(net["ids"][m] < 800, P["teal"], P["rose"]), lw=0)
        elif i == 4:
            prev = None
            for layer, n in enumerate((5, 7, 7, 5)):
                ys = np.linspace(0.1, 0.9, n)
                if prev is not None:
                    for y0 in prev:
                        for y1 in ys:
                            ax.plot([layer - 1, layer], [y0, y1], color=cols[i], lw=0.25, alpha=0.6)
                ax.scatter(np.full(n, layer), ys, s=10, color=cols[i], zorder=3, ec="white", lw=0.4)
                prev = ys
        else:
            I = np.arange(0, 25.01, 1.0)
            for f, ls in ((0.0, "-"), (0.2, "--")):
                yy = ns.fi_curve(I, ns.sodium_block(f), duration=300).astype(float)
                yy[yy == 0] = np.nan          # miniature shows only the firing branch
                ax.plot(I, yy, color=cols[i], lw=1.1, ls=ls)
            ax.set_ylim(-40, 105)               # zero-rate branch omitted; fixed scale
    fig.add_artist(FancyArrowPatch((0.03, 0.15), (xs[4] + w, 0.15), transform=fig.transFigure,
                                   arrowstyle="<|-|>", mutation_scale=8, color=P["ink"], lw=0.8))
    fig.text(0.03, 0.05, L["axis_l"], fontsize=7, color=P["ink"])
    fig.text(xs[4] + w, 0.05, L["axis_r"], fontsize=7, color=P["ink"], ha="right")
    fig.text(xs[5] + w / 2, 0.15, L["applies"], fontsize=6.5, color=P["ink"], ha="center",
             va="center", style="italic")
    for i in range(5):
        fig.add_artist(FancyArrowPatch((xs[i] + w + 0.002, 0.57), (xs[i + 1] - 0.002, 0.57),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=7, color=P["grey"], lw=0.7))
    save(fig, out, "fig1_overview")


# ===================================================================== Fig 2
def fig_hh(out, L, R):
    stim = lambda t: 10.0 if 10 <= t < 60 else 0.0
    r = ns.simulate_hh(stim, 80)
    I_grid = np.arange(0, 20.01, 0.25)
    rates = np.array([ns.hh_rate(I) for I in I_grid])
    onset = float(I_grid[np.argmax(rates > 0)])
    conv = {str(dt): {f"{I:.2f}": float(ns.hh_rate(I, dt=dt)) for I in (6.2, 6.25, 6.3, 6.5, 10.0)}
            for dt in (0.01, 0.005, 0.0025)}           # Euler time-step sensitivity near threshold
    Vi = r["V"][1:-1]
    R["hh"] = {"onset_uA_cm2": onset, "rate_at_onset_hz": float(rates[rates > 0][0]),
               "rate_at_20_hz": float(rates[-1]), "grid_step_uA_cm2": 0.25,
               "rate_resolution_hz": 2.0, "onset_dt_sensitivity": conv,
               "peaks_mV": [float(v) for v in Vi[(Vi > r["V"][:-2]) & (Vi > r["V"][2:]) & (Vi > 0)]],
               "min_V_during_pulse_mV": float(r["V"][(r["t"] > 10) & (r["t"] < 60)].min())}
    fig, ax = plt.subplots(2, 2, figsize=(DOUBLE, 112 * MM), layout="constrained")
    a = ax[0, 0]
    a.plot(r["t"], r["V"], color=P["wine"], lw=1.0)
    stim_bar(a, r["t"], [stim(x) for x in r["t"]], -97, 10, P["sand"])
    a.text(61, -92, f"{L['stim']}: 10 µA/cm²", fontsize=6.5, color=P["ink"], va="center")
    a.set(xlabel=L["t"], ylabel=L["V"], ylim=(-99, 50)); tag(a, "A")
    a = ax[0, 1]
    for g, c, ls in (("m", P["wine"], "-"), ("h", P["indigo"], "--"), ("n", P["teal"], "-.")):
        a.plot(r["t"], r[g], color=c, lw=1.2, ls=ls, label=g)
    a.set(xlabel=L["t"], ylabel=L["gate"], xlim=(8, 32), ylim=(-0.02, 1.02))
    a.legend(ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.0), handlelength=2.4, borderaxespad=0.1)
    tag(a, "B")
    a = ax[1, 0]
    sel = (r["t"] > 25) & (r["t"] < 42)
    sc = a.scatter(r["V"][sel], r["n"][sel], c=r["t"][sel], cmap="viridis", s=1.2, lw=0, rasterized=True)
    a.plot(r["V"][sel][0], r["n"][sel][0], "o", ms=4, mfc="none", mec=P["ink"], mew=0.8)
    a.set(xlabel=L["V"], ylabel="n"); tag(a, "C")
    cb = fig.colorbar(sc, ax=a, fraction=0.05, pad=0.02); cb.set_label(L["t"]); cb.ax.tick_params(labelsize=6.5)
    a = ax[1, 1]
    xx, yy = break_jumps(I_grid, rates)
    a.plot(xx, yy, color=P["wine"], lw=1.0)
    a.plot(I_grid, rates, ls="none", marker="o", ms=1.8, color=P["wine"])
    a.annotate(f"{onset:g} µA/cm²", xy=(onset, R['hh']['rate_at_onset_hz']), xytext=(onset + 3, 22),
               arrowprops=dict(arrowstyle="->", color=P["ink"], lw=0.6), fontsize=6.5)
    a.set(xlabel=L["Iua"], ylabel=L["rate"]); tag(a, "D")
    save(fig, out, "fig2_hodgkin_huxley")


# ===================================================================== Fig 3
def _lif_rates(I, dt, duration=2000.0):
    """(count-based, ISI-based) rates. The count-based rate is quantised to
    1/duration (0.5 Hz for 2 s) and includes the onset transient; the
    ISI-based rate (1 / mean ISI) is not quantised and isolates Euler error."""
    cnt, isi = [], []
    for i in I:
        s = ns.simulate_lif(i, duration, dt)["spikes"]
        cnt.append(s.size / (duration / 1000.0))
        isi.append(1000.0 / np.mean(np.diff(s)) if s.size > 2 else 0.0)
    return np.array(cnt), np.array(isi)


def fig_lif(out, L, R):
    fig, ax = plt.subplots(1, 3, figsize=(DOUBLE, 60 * MM), layout="constrained",
                           gridspec_kw=dict(width_ratios=[1.4, 1, 1]))
    a = ax[0]
    for I, c, ls in ((1.4, P["grey"], ":"), (2.0, P["indigo"], "-"), (3.0, P["teal"], "--")):
        r = ns.simulate_lif(lambda t, I=I: I if t >= 10 else 0.0, 80, 0.05)
        a.plot(r["t"], r["V"], color=c, lw=0.9, ls=ls, label=f"{I} nA")
    a.axhline(ns.LIF_CELL["V_th"], color=P["rose"], lw=0.7)
    a.text(1, ns.LIF_CELL["V_th"] + 1.5, "V$_{th}$", color=P["ink"], fontsize=6.5)
    a.set(xlabel=L["t"], ylabel=L["V"])
    a.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=2.4, borderaxespad=0.1)
    tag(a, "A", -0.2)
    a = ax[1]
    I = np.linspace(0, 5, 26)
    sim_cnt, sim_isi = _lif_rates(I, 0.05)
    fine = np.linspace(0, 5, 400)
    a.plot(fine, ns.lif_rate_closed_form(fine), color=P["indigo"], lw=1.2, label=L["theory"])
    a.scatter(I, sim_isi, s=10, marker="s", facecolor="white", edgecolor=P["ink"], lw=0.7,
              label=L["sim"] + ", Δt = 0.05 ms", zorder=3)
    a.set(xlabel=L["Ina"], ylabel=L["rate"], ylim=(-5, 200))
    a.legend(loc="upper left", handlelength=1.5); tag(a, "B", -0.28)
    a = ax[2]
    dts = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    I2 = np.linspace(1.6, 5, 18)
    th = ns.lif_rate_closed_form(I2)
    e_cnt, e_isi, e_isi_mean = [], [], []
    for d in dts:
        c_, i_ = _lif_rates(I2, d)
        e_cnt.append(np.max(np.abs(c_ - th))); e_isi.append(np.max(np.abs(i_ - th)))
        e_isi_mean.append(np.mean(np.abs(i_ - th)))
    slope = float(np.polyfit(np.log10(dts), np.log10(e_isi), 1)[0])
    a.loglog(dts, e_isi, "o-", color=P["teal"], ms=3.5, lw=1.0, label=L["isi_err"])
    a.loglog(dts, e_cnt, "s--", color=P["grey"], ms=3, lw=0.8, mfc="white", label=L["cnt_err"])
    a.axhline(0.5, color=P["grey"], lw=0.6, ls=":")
    a.text(0.06, 0.38, L["resolution"], fontsize=6.5, color=P["ink"])
    a.loglog(dts, e_isi[3] * dts / dts[3], color=P["ink"], lw=0.5, ls="-.", label=L["slope1"])
    a.set(xlabel=L["dt"], ylabel=L["err"]); a.legend(loc="upper left", handlelength=2.4); tag(a, "C", -0.3)
    R["lif"] = {"max_err_dt005_hz": float(np.max(np.abs(sim_cnt - ns.lif_rate_closed_form(I)))),   # v1 key
                "err_by_dt": dict(zip(map(float, dts), map(float, e_cnt))),                          # v1 key
                "max_err_dt005_hz_count": float(np.max(np.abs(sim_cnt - ns.lif_rate_closed_form(I)))),
                "max_err_dt005_hz_isi": float(np.max(np.abs(sim_isi - ns.lif_rate_closed_form(I)))),
                "rheobase_nA": (ns.LIF_CELL["V_th"] - ns.LIF_CELL["E_L"]) / ns.LIF_CELL["R_m"],
                "err_by_dt_count": dict(zip(map(float, dts), map(float, e_cnt))),
                "err_by_dt_isi_max": dict(zip(map(float, dts), map(float, e_isi))),
                "err_by_dt_isi_mean": dict(zip(map(float, dts), map(float, e_isi_mean))),
                "loglog_slope_isi_max": slope, "error_current_grid_nA": "linspace(1.6, 5, 18)"}
    save(fig, out, "fig3_lif")


# ===================================================================== Fig 4
def fig_izh(out, L, R):
    fig, axs = plt.subplots(2, 3, figsize=(DOUBLE, 92 * MM), layout="constrained")
    axs = axs.ravel()
    counts = {}
    for k, (key, spec) in enumerate(ns.IZH_CLASSES.items()):
        r = ns.simulate_izhikevich(*spec["abcd"], lambda t: 10.0 * (t >= 20), 250)
        a = axs[k]
        a.plot(r["t"], r["v"][0], color=CLASS_COLOURS[k], lw=0.7)
        a.fill_between([20, 250], -89, -85, color=P["sand"], lw=0)
        a.set_title(f"{key} · {L['iz_names'][key]}", fontsize=7, color=P["ink"])
        a.set_ylim(-90, 38); a.set_yticks([-80, -40, 0, 30]); a.set_xlim(0, 250)
        if k % 3 == 0: a.set_ylabel("v (mV)")
        a.set_xlabel(L["t"])
        counts[key] = int(r["spikes"][0].size)
        tag(a, "ABCDE"[k], -0.24, 1.04)
    a = axs[5]
    a.set_axis_off()
    for j, hcell in enumerate(["", "a", "b", "c", "d"]):
        a.text(0.05 + 0.2 * j, 0.9, hcell, fontweight="bold", fontsize=7, transform=a.transAxes, style="italic")
    for k, (key, spec) in enumerate(ns.IZH_CLASSES.items()):
        yy = 0.75 - 0.14 * k
        a.add_patch(plt.Rectangle((0.0, yy - 0.05), 1.0, 0.12, transform=a.transAxes,
                                  color=CLASS_COLOURS[k], alpha=0.15, lw=0))
        a.text(0.05, yy, key, color=P["ink"], fontweight="bold", fontsize=7, transform=a.transAxes)
        for j, val in enumerate(spec["abcd"]):
            a.text(0.25 + 0.2 * j, yy, minus(val), fontsize=7, transform=a.transAxes)
    a.text(0.0, 0.0, L["stepnote"], fontsize=6.5, transform=a.transAxes)
    R["izh_spike_counts_250ms"] = counts
    save(fig, out, "fig4_izhikevich")


# ===================================================================== Fig 5
def fig_network(out, L, R):
    fig = plt.figure(figsize=(DOUBLE, 118 * MM), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.95])
    ex = {}
    for j, (g, ext, title) in enumerate(((2.0, 8.0, L["weak"]), (6.0, 7.0, L["strong"]))):
        net = ns.simulate_ei_network(g=g, ext_rate=ext, duration=500, seed=0)
        ex[g] = ns.network_statistics(net)
        sub = gs[0, j].subgridspec(2, 1, height_ratios=[3, 1])
        a = fig.add_subplot(sub[0]); a2 = fig.add_subplot(sub[1], sharex=a)
        m = (net["times"] >= 350) & (net["ids"] % 4 == 0)
        e = net["ids"][m] < net["n_exc"]
        a.scatter(net["times"][m][e], net["ids"][m][e], s=1.0, marker="|", color=P["teal"], lw=0.5,
                  label=L["exc"], rasterized=True)
        a.scatter(net["times"][m][~e], net["ids"][m][~e], s=1.0, marker="|", color=P["rose"], lw=0.5,
                  label=L["inh"], rasterized=True)
        a.axhline(net["n_exc"] - 0.5, color=P["grey"], lw=0.4, ls=":")
        h, edges = np.histogram(net["times"], bins=np.arange(350, 500.01, 1.0))
        a2.fill_between(edges[:-1], h / net["n_cells"] * 1000, color=P["ink"], lw=0, step="post")
        a2.set_xlim(350, 500); a2.set_xticks([350, 400, 450, 500]); a2.set_ylabel(L["pop"]); a2.set_xlabel(L["t"])
        a.tick_params(labelbottom=False)
        a.set_title(title); a.set_ylabel(L["neuron"]); tag(a, "AB"[j], -0.24)
        if j == 1:
            a.legend(loc="upper right", ncol=2, markerscale=5, handletextpad=0.2, borderaxespad=0,
                     bbox_to_anchor=(1.0, 1.0), framealpha=0.9, frameon=True, edgecolor="none")
    g_vals = np.array([1, 2, 3, 4, 5, 6, 7, 8], float)
    seeds = list(range(5))
    stats = np.array([[list(ns.network_statistics(ns.simulate_ei_network(g=g, ext_rate=7.0, duration=400, seed=s)).values())
                       for s in seeds] for g in g_vals])       # g x seed x (rate, cv, sync)
    mean, sd = stats.mean(1), stats.std(1, ddof=1)
    ci = ns.t_ppf(0.975, len(seeds) - 1) * sd / np.sqrt(len(seeds))
    sub = gs[0, 2].subgridspec(3, 1)
    jit = np.linspace(-0.16, 0.16, len(seeds))                  # horizontal offset, display only
    for row, (col, lab, logy) in enumerate(((0, L["rate"], False), (1, L["cv"], False), (2, L["sync"], True))):
        a = fig.add_subplot(sub[row])
        for s in range(len(seeds)):
            a.plot(g_vals + jit[s], stats[:, s, col], ls="none", marker="o", ms=1.8, color=P["grey"],
                   alpha=0.9, label=L["seeds"] if s == 0 else None)
        a.errorbar(g_vals, mean[:, col], sd[:, col], color=P["indigo"], marker="o", ms=2.8, lw=0.9,
                   capsize=1.5, elinewidth=0.8, label=L["mean_sd"])
        if logy: a.set_yscale("log")
        a.set_ylabel(lab); a.set_xticks(g_vals)
        if row == 2: a.set_xlabel(L["g"])
        else: a.tick_params(labelbottom=False)
        if row == 0: a.legend(loc="upper right", handlelength=1.2)
        tag(a, "CDE"[row], -0.34, 0.92)
    R["network"] = {"examples": {str(k): v for k, v in ex.items()},
                    "sweep_g": g_vals.tolist(), "sweep_mean": mean.tolist(), "sweep_sd": sd.tolist(),
                    "sweep_ci95_halfwidth_t4": ci.tolist(), "sweep_per_seed": stats.tolist(),
                    "n_seeds": len(seeds), "stat_order": ["rate_hz", "cv_isi", "synchrony"]}
    save(fig, out, "fig5_network")


# ===================================================================== Fig 6
def _fit(X, y, hidden, seed, Xv=None, yv=None):
    net = ns.SpikeClassNet(X.shape[1], len(ns.CLASS_KEYS), hidden, seed=seed)
    h = net.fit(X, y, epochs=150, X_val=Xv, y_val=yv, seed=seed)
    return net, h


def fig_classifier(out, L, R):
    X, y, trains = ns.make_dataset(n_per_class=120, seed=0)
    K = len(ns.CLASS_KEYS)
    models = (("mlp", (16, 16)), ("logreg", ()))
    accs = {m: [] for m, _ in models}; f1s = {m: [] for m, _ in models}
    briers = {m: [] for m, _ in models}; percls = {m: [] for m, _ in models}
    M = np.zeros((K, K), int); hists = []
    n_splits, test_frac = 10, 0.3
    for s in range(n_splits):
        tr, te = ns.stratified_split(y, test_frac, seed=s)
        for name, hidden in models:
            net, h = _fit(X[tr], y[tr], hidden, s, X[te], y[te])
            Pte = net.predict_proba(X[te]); pred = Pte.argmax(1)
            accs[name].append(float(np.mean(pred == y[te]))); f1s[name].append(ns.macro_f1(y[te], pred, K))
            briers[name].append(ns.brier_score(Pte, y[te]))
            percls[name].append(ns.per_class_metrics(y[te], pred, K)["f1"])
            if name == "mlp":
                M += ns.confusion_matrix(y[te], pred, K)
                hists.append(h)
                if s == 0: n_params = net.n_params
    n_te = int(len(te)); n_tr = len(y) - n_te
    diff = np.array(accs["mlp"]) - np.array(accs["logreg"])
    crt = lambda v: ns.corrected_resampled_t(v, n_tr, n_te)

    # Independent test set: 600 new cells from the same generative process (seed 1).
    # Models are trained once on all 600 development cells (seed 0).
    X2, y2, _ = ns.make_dataset(n_per_class=120, seed=1)
    ext, P2 = {}, {}
    for name, hidden in models:
        net, _ = _fit(X, y, hidden, 0)
        P2[name] = net.predict_proba(X2)
        pred = P2[name].argmax(1)
        pc = ns.per_class_metrics(y2, pred, K)
        bc, ba, bn, ece = ns.calibration_curve(P2[name], y2, n_bins=10)
        ext[name] = {
            "acc": ns.stratified_bootstrap(y2, lambda i, p=pred: np.mean(p[i] == y2[i]), seed=0),
            "macro_f1": ns.stratified_bootstrap(y2, lambda i, p=pred: ns.macro_f1(y2[i], p[i], K), seed=0),
            "brier": ns.brier_score(P2[name], y2), "log_loss": ns.cross_entropy(P2[name], y2), "ece": ece,
            "reliability": {"conf": bc.tolist(), "acc": ba.tolist(), "n": bn.tolist()},
            "per_class": {ns.CLASS_KEYS[k]: {m_: float(pc[m_][k]) for m_ in ("precision", "recall", "specificity", "f1")}
                          for k in range(K)},
            "per_class_ci": {ns.CLASS_KEYS[k]: {
                "recall": ns.stratified_bootstrap(y2, lambda i, p=pred, k=k: ns.per_class_metrics(y2[i], p[i], K)["recall"][k], n_boot=1000, seed=0),
                "precision": ns.stratified_bootstrap(y2, lambda i, p=pred, k=k: ns.per_class_metrics(y2[i], p[i], K)["precision"][k], n_boot=1000, seed=0),
                "f1": ns.stratified_bootstrap(y2, lambda i, p=pred, k=k: ns.per_class_metrics(y2[i], p[i], K)["f1"][k], n_boot=1000, seed=0)}
                for k in range(K)},
            "confusion_counts": ns.confusion_matrix(y2, pred, K).tolist()}
    c_mlp, c_lr = P2["mlp"].argmax(1) == y2, P2["logreg"].argmax(1) == y2
    ext["diff_acc"] = ns.stratified_bootstrap(y2, lambda i: np.mean(c_mlp[i]) - np.mean(c_lr[i]), seed=0)
    ext["mcnemar"] = ns.mcnemar_exact(c_mlp, c_lr)
    seed_acc = {name: [float(np.mean(_fit(X, y, hidden, s)[0].predict(X2) == y2)) for s in range(10)]
                for name, hidden in models}            # initialisation / mini-batch-order variability
    ext["train_seed_acc"] = {k: {"values": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
                             for k, v in seed_acc.items()}

    R["classifier"] = {
        "n_cells": int(len(y)), "n_per_class": 120, "n_splits": n_splits, "n_train": n_tr, "n_test": n_te,
        "mlp_params": n_params,
        "mlp_acc": ns.bootstrap_ci(accs["mlp"]), "logreg_acc": ns.bootstrap_ci(accs["logreg"]),
        "mlp_f1": ns.bootstrap_ci(f1s["mlp"]), "logreg_f1": ns.bootstrap_ci(f1s["logreg"]),
        "acc_diff": ns.bootstrap_ci(diff), "mlp_acc_sd": float(np.std(accs["mlp"], ddof=1)),
        "logreg_acc_sd": float(np.std(accs["logreg"], ddof=1)),
        "per_split_acc": accs, "per_split_f1": f1s,
        "corrected_t": {"mlp_acc": crt(accs["mlp"]), "logreg_acc": crt(accs["logreg"]),
                        "mlp_f1": crt(f1s["mlp"]), "logreg_f1": crt(f1s["logreg"]), "acc_diff": crt(diff)},
        "cv_brier": {k: float(np.mean(v)) for k, v in briers.items()},
        "cv_per_class_f1_mean": {k: dict(zip(ns.CLASS_KEYS, np.mean(v, 0).tolist())) for k, v in percls.items()},
        "confusion_counts": M.tolist(),
        "confusion_note": "summed over 10 overlapping test sets (1800 predictions on 600 cells)",
        "independent_test": ext,
        "feature_means": {ns.CLASS_KEYS[k]: X[y == k].mean(0).tolist() for k in range(K)}}

    # ------------------------------------------------------------- figure
    fig = plt.figure(figsize=(DOUBLE, 122 * MM), layout="constrained")
    gs = fig.add_gridspec(2, 3)
    a = fig.add_subplot(gs[0, 0])
    for k in range(K):
        idx = np.flatnonzero(y == k)[:4]
        for r_, i in enumerate(idx):
            s = trains[i]; s = s[s < 300]
            a.vlines(s, k * 5 + r_, k * 5 + r_ + 0.8, color=CLASS_COLOURS[k], lw=0.6)
    a.axvline(20, color=P["grey"], lw=0.5, ls=":")
    a.set_yticks([k * 5 + 1.9 for k in range(K)]); a.set_yticklabels(ns.CLASS_KEYS)
    a.set(xlabel=L["t"], xlim=(0, 300)); a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0)
    tag(a, "A", -0.14)

    a = fig.add_subplot(gs[0, 1])
    cols2 = [3, 1]   # burst fraction, CV
    net2 = ns.SpikeClassNet(2, K, (16, 16), seed=0); net2.fit(X[:, cols2], y, epochs=150, seed=0)
    gx, gy = np.meshgrid(np.linspace(-0.03, 1.0, 300), np.linspace(0, 2.6, 300))
    Z = net2.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    a.contourf(gx, gy, Z, levels=np.arange(-0.5, K), colors=CLASS_COLOURS, alpha=0.18)
    a.contour(gx, gy, Z, levels=np.arange(0.5, K - 1), colors=P["grey"], linewidths=0.4)
    for k in reversed(range(K)):           # RS drawn last so it is not hidden at burst fraction 0
        m = y == k
        a.scatter(X[m, 3], X[m, 1], s=6, marker=CLASS_MARKERS[k], facecolor="none",
                  edgecolor=CLASS_COLOURS[k], lw=0.5, alpha=0.9, label=ns.CLASS_KEYS[k], rasterized=True)
    n0 = int(np.sum(X[:, 3] == 0))
    a.annotate(L["cells0"].format(n=n0), xy=(0.0, 0.3), xytext=(0.03, 2.25), fontsize=6.5,
               arrowprops=dict(arrowstyle="->", lw=0.5, color=P["ink"]))
    a.set(xlabel=L["feat_x"], ylabel=L["feat_y"], xlim=(-0.03, 1.0), ylim=(0, 2.6))
    h_, l_ = a.get_legend_handles_labels()
    a.legend(h_[::-1], l_[::-1], ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.1,
             columnspacing=0.6, markerscale=1.5, borderaxespad=0.1)
    tag(a, "B", -0.22)

    a = fig.add_subplot(gs[0, 2])
    ep = np.arange(1, 151)
    for key, ls, lab in (("train_loss", "-", L["train"]), ("val_loss", "--", L["test"])):
        H = np.array([h[key] for h in hists])
        a.fill_between(ep, H.min(0), H.max(0), color=P["indigo"], alpha=0.15, lw=0)
        a.plot(ep, np.median(H, 0), color=P["indigo"], ls=ls, lw=1.0, label=f"{lab} ({L['median_range']})")
    a.set(xlabel=L["epoch"], ylabel=L["loss"], yscale="log"); a.legend(loc="lower left"); tag(a, "C", -0.24)

    a = fig.add_subplot(gs[1, 0])
    Mi = np.array(ext["mlp"]["confusion_counts"]); Mn = Mi / Mi.sum(1, keepdims=True)
    cmap = ListedColormap(plt.cm.BuPu(np.linspace(0.02, 0.85, 256)))
    a.imshow(Mn, cmap=cmap, vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            a.text(j, i, f"{Mi[i, j]}\n{100 * Mn[i, j]:.0f}%", ha="center", va="center", fontsize=6.5,
                   color="white" if Mn[i, j] > 0.55 else P["ink"], linespacing=0.95)
    a.set_xticks(range(K)); a.set_yticks(range(K)); a.set_xticklabels(ns.CLASS_KEYS); a.set_yticklabels(ns.CLASS_KEYS)
    a.set(xlabel=L["pred"], ylabel=L["true"]); a.spines[:].set_visible(False); a.tick_params(length=0)
    tag(a, "D", -0.3)

    a = fig.add_subplot(gs[1, 1])
    for s in range(n_splits):
        a.plot([0, 1], [100 * accs["mlp"][s], 100 * accs["logreg"][s]], color=P["grey"], lw=0.5,
               marker="o", ms=2.0, alpha=0.8)
    for name, x0, c, mk in (("mlp", 0, P["indigo"], "o"), ("logreg", 1, P["rose"], "s")):
        ct = R["classifier"]["corrected_t"][f"{name}_acc"]
        a.errorbar(x0 + (-0.2 if name == "mlp" else 0.2), 100 * ct["mean"],
                   [[100 * (ct["mean"] - ct["lower"])], [100 * (ct["upper"] - ct["mean"])]],
                   fmt=mk, color=c, ms=3.5, capsize=2, lw=0.9)
        e_ = ext[name]["acc"]
        a.errorbar(x0 + 2.3, 100 * e_[0], [[100 * (e_[0] - e_[1])], [100 * (e_[2] - e_[0])]],
                   fmt=mk, color=c, ms=3.5, capsize=2, lw=0.9)
    a.set_xticks([0, 1, 2.3, 3.3], [L["mlp"], L["lr"], L["mlp"], L["lr"]])
    a.set_xlim(-0.5, 3.8)
    a.axvline(1.65, color=P["grey"], lw=0.4)
    a.text(0.5, 1.0, L["split"], transform=a.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5)
    a.text(2.8, 1.0, L["indep"], transform=a.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5)
    a.set_ylabel(L["acc"] + " (%)"); tag(a, "E", -0.26, 1.06)

    a = fig.add_subplot(gs[1, 2])
    a.plot([0, 1], [0, 1], color=P["grey"], lw=0.6, ls=":", label=L["perfect"])
    for name, c, mk, ls in (("mlp", P["indigo"], "o", "-"), ("logreg", P["rose"], "s", "--")):
        rel = ext[name]["reliability"]
        cf, ac, nn = (np.array(rel[k_], float) for k_ in ("conf", "acc", "n"))
        ok = nn >= 10                                      # show bins with at least 10 cells
        a.plot(cf[ok], ac[ok], ls=ls, marker=mk, ms=3, color=c, lw=0.9,
               label=f"{L['mlp'] if name == 'mlp' else L['lr']} (ECE {ext[name]['ece']:.3f})")
    a.set(xlabel=L["confidence"], ylabel=L["frac_correct"], xlim=(0.3, 1.02), ylim=(0.3, 1.02))
    a.text(0.33, 1.0, L["binnote"].format(m=ext['mlp']['reliability']['n'][-1], mn=L['mlp'],
                                          l=ext['logreg']['reliability']['n'][-1], ln=L['lr'], n=len(y2)),
           fontsize=6.5, va="top", color=P["ink"])
    a.set_aspect("equal"); a.legend(loc="lower right"); tag(a, "F", -0.3)
    save(fig, out, "fig6_classifier")


# ===================================================================== Fig 7
def fig_clinic(out, L, R):
    fig = plt.figure(figsize=(DOUBLE, 118 * MM), layout="constrained")
    gs = fig.add_gridspec(2, 2)
    a = fig.add_subplot(gs[0, 0])
    I = np.arange(0, 30.01, 1.0)
    styles = [(P["indigo"], "-", "o"), (P["purple"], "--", "s"), (P["rose"], "-.", "^"), (P["grey"], ":", "v")]
    na = {}
    for f, (c, ls, mk) in zip((0.0, 0.1, 0.2, 0.3), styles):
        fi = ns.fi_curve(I, ns.sodium_block(f), duration=400)
        xx, yy = break_jumps(I, fi)
        a.plot(xx, yy, color=c, lw=1.0, ls=ls, marker=mk, ms=2.2, label=f"{int(f*100)} % {L['blocked']}")
        na[f] = float(I[np.argmax(fi > 0)]) if np.any(fi > 0) else None
    a.set(xlabel=L["Iua"], ylabel=L["rate"], ylim=(-4, 105)); a.set_title(L["na"])
    a.legend(loc="center left", bbox_to_anchor=(0.0, 0.38)); tag(a, "A", -0.14)

    a = fig.add_subplot(gs[0, 1])
    kw = {}
    for f, c, ls in zip((0.0, 0.2, 0.3), (P["indigo"], P["wine"], P["rose"]), ("-", "--", ":")):
        w, r = ns.spike_width(ns.potassium_block(f))
        a.plot(r["t"], r["V"], color=c, lw=1.1, ls=ls, label=f"{int(f*100)} % · {w:.2f} ms")
        kw[f] = float(w)
    a.axhline(-20, color=P["grey"], lw=0.5, ls=":")
    a.set(xlabel=L["t"], ylabel=L["V"], xlim=(4, 14)); a.set_title(L["k"])
    a.legend(loc="upper right", title=L["width"]); tag(a, "B", -0.14)
    ins = a.inset_axes([0.62, 0.2, 0.34, 0.34])       # zoom on repolarisation at -20 mV
    for f, c, ls in zip((0.0, 0.2, 0.3), (P["indigo"], P["wine"], P["rose"]), ("-", "--", ":")):
        _, r = ns.spike_width(ns.potassium_block(f))
        ins.plot(r["t"], r["V"], color=c, lw=1.1, ls=ls)
    ins.axhline(-20, color=P["grey"], lw=0.5, ls=":")
    ins.set(xlim=(7.6, 8.3), ylim=(-30, -10), xticks=[7.7, 8.0, 8.3], yticks=[-30, -20, -10])
    ins.tick_params(labelsize=6.5)
    a.indicate_inset_zoom(ins, edgecolor=P["grey"], lw=0.5)
    spont = {}
    for f in (0.3, 0.4, 0.5):
        p = ns.potassium_block(f); rr = ns.simulate_hh(0.0, 600, params=p, init=ns.settle(p))
        s = ns.spike_times(rr["t"], rr["V"]); spont[str(f)] = float(np.sum(s > 100) / 0.5)

    a = fig.add_subplot(gs[1, 0])
    for ko, c, ls in ((4, P["indigo"], "-"), (7, P["rose"], "--"), (12, P["wine"], "-.")):
        pk = ns.extracellular_potassium(ko)
        r = ns.simulate_hh(0.0, 150, params=pk, init=ns.settle(pk))
        a.plot(r["t"], r["V"], color=c, lw=0.9, ls=ls, label=f"{ko} mM")
    a.set(xlabel=L["t"], ylabel=L["V"], ylim=(-75, 48)); a.set_title(L["hyperk"])
    a.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), handlelength=2.4); tag(a, "C", -0.14)

    sub = gs[1, 1].subgridspec(2, 1, height_ratios=[1.4, 1])
    kos = np.arange(3, 14.01, 0.5)
    vm, fr = [], []
    for ko in kos:
        r = ns.simulate_hh(0.0, 400, params=ns.extracellular_potassium(ko))
        s = ns.spike_times(r["t"], r["V"]); vm.append(r["V"][r["t"] > 200].mean()); fr.append(np.sum(s > 100) / 0.3)
    fr = np.array(fr); firing = fr > 0
    lo_, hi_ = kos[firing].min() - 0.25, kos[firing].max() + 0.25
    a = fig.add_subplot(sub[0])
    a.axvspan(lo_, hi_, color=P["sand"], alpha=0.4, lw=0)
    a.plot(kos, vm, color=P["indigo"], lw=1.0, marker="s", ms=2.2)
    a.set_ylabel(L["Vrest"]); a.set_title(L["hyperk_sum"]); a.tick_params(labelbottom=False)
    a.text((lo_ + hi_) / 2, -38, L["rest_firing"], ha="center", va="center", fontsize=6.5, color=P["ink"])
    tag(a, "D", -0.24)
    b = fig.add_subplot(sub[1], sharex=a)
    b.axvspan(lo_, hi_, color=P["sand"], alpha=0.4, lw=0)
    b.vlines(kos, 0, fr, color=P["rose"], lw=1.2)
    b.plot(kos, fr, ls="none", marker="o", ms=2.2, color=P["rose"])
    b.set(xlabel=L["ko"], ylabel=L.get("rate_short", L["rate"]), ylim=(0, 90)); tag(b, "E", -0.24)
    R["clinic"] = {"na_block_onset_uA_cm2": {str(k): v for k, v in na.items()}, "na_block_grid_uA_cm2": 1.0,
                   "k_block_width_ms": {str(k): v for k, v in kw.items()},
                   "k_block_spontaneous_rate_hz": spont,
                   "hyperk_K": kos.tolist(), "hyperk_mean_V": list(map(float, vm)), "hyperk_rate": list(map(float, fr)),
                   "hyperk_EK_mV": [float(ns.extracellular_potassium(k)["EK"]) for k in kos]}
    save(fig, out, "fig7_clinical")


# ===================================================================== cost
def environment():
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip(); break
    except OSError:
        pass
    return {"python": sys.version.split()[0], "numpy": np.__version__, "matplotlib": matplotlib.__version__,
            "neurosim": ns.__version__, "platform": platform.platform(), "cpu": cpu,
            "logical_cpus": os.cpu_count()}


def benchmark(R, reps=5):
    runs = {"Hodgkin-Huxley": (lambda: ns.simulate_hh(10.0, 1000, 0.01), 0.01, 4),
            "LIF": (lambda: ns.simulate_lif(2.0, 1000, 0.1), 0.1, 1),
            "Izhikevich": (lambda: ns.izhikevich_cell(*ns.IZH_CLASSES["RS"]["abcd"], 10.0, 1000, 0.25), 0.25, 2)}
    rows = []
    for name, (f, dt, nv) in runs.items():
        times = []
        for _ in range(reps):
            t0 = time.perf_counter(); f(); times.append(time.perf_counter() - t0)
        best = min(times)
        rows.append({"model": name, "dt_ms": dt, "state_variables": nv, "wall_s_per_sim_s": best,
                     "median_s": float(np.median(times)), "us_per_step": 1e6 * best / (1000 / dt), "reps": reps})
    R["cost"] = rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lang", default="en", choices=["en", "es"])
    ap.add_argument("--out", default=None, help="output folder for figures (default figures/ or figures_es/)")
    args = ap.parse_args(); lang = args.lang
    L = T[lang]
    out = Path(args.out or ("figures" if lang == "en" else "figures_es")); out.mkdir(parents=True, exist_ok=True)
    style()
    R = {"environment": environment(),
         "palette_contrast_on_white": {k: round(contrast(v), 2) for k, v in P.items() if k != "paper"}}
    t_start = time.perf_counter()
    fig_overview(out, L); print("fig_overview done", flush=True)
    for f in (fig_hh, fig_lif, fig_izh, fig_network, fig_classifier, fig_clinic):
        f(out, L, R); print(f.__name__, "done", flush=True)
    R["figure_audit"] = AUDIT
    if lang == "en":
        benchmark(R)
        R["runtime_s"] = time.perf_counter() - t_start
        Path("results").mkdir(exist_ok=True)
        txt = json.dumps(R, indent=2)
        Path("results/results.json").write_text(txt)
        if args.out:
            (out / "results.json").write_text(txt)
        print(json.dumps({k: R[k] for k in ("cost", "figure_audit")}, indent=1))


if __name__ == "__main__":
    main()
