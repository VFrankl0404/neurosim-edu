---
title: "Supplementary File S2. Parameters, additional results and reporting checklist"
---

Supplement to: *From ion channels to machine learning: an open, bilingual simulation pathway for teaching computational neuroscience to physicians and life-science students.* All values are produced by `make_figures.py` and stored in `results/results.json`.

# Table S2.1. Model and simulation parameters

| Model | Parameter | Symbol | Value | Unit | Source / note |
|---|---|---|---|---|---|
| HH | membrane capacitance | C | 1 | µF/cm² | Hodgkin & Huxley 1952, modern convention |
| HH | max. Na⁺ conductance | ḡ_Na | 120 | mS/cm² | idem |
| HH | max. K⁺ conductance | ḡ_K | 36 | mS/cm² | idem |
| HH | leak conductance | g_L | 0.3 | mS/cm² | idem |
| HH | reversal potentials | E_Na, E_K, E_L | 50, −77, −54.387 | mV | idem |
| HH | integration | — | forward Euler, Δt = 0.01 ms | — | initial state: gates at steady state for V₀ = −65 mV, or `settle()` (300 ms unstimulated) where stated |
| HH | spike detection | — | upward crossing of 0 mV | — | rate = spikes in (100, 600] ms / 0.5 s (Fig. 2D); (100, 400] ms / 0.3 s (Fig. 7) |
| HH | f-I grid | — | 0–20 by 0.25 (Fig. 2D); 0–30 by 1 (Fig. 7A) | µA/cm² | grid explains 6.25 vs 7 µA/cm² onset |
| HH, Na⁺ block | ḡ_Na scaling | — | 1 − f, f = 0, 0.1, 0.2, 0.3 | — | use-independent block |
| HH, K⁺ block | ḡ_K scaling | — | 1 − f, f = 0, 0.2, 0.3 (width); 0.3, 0.4, 0.5 (spontaneous firing) | — | delayed rectifier only |
| HH, spike width | test pulse | — | 20 µA/cm² for 1 ms at t = 5 ms, from settled state; width at −20 mV | — | resolution 0.01 ms |
| HH, hyperkalaemia | E_K | — | −77 + 26.7 ln([K⁺]ₒ/4) | mV | RT/F at 37 °C; [K⁺]ₒ = 3–14 mM by 0.5; E_L fixed |
| LIF | membrane time constant | τ_m | 10 | ms | Gerstner et al. 2014 |
| LIF | resting / reset potential | E_L = V_reset | −65 | mV | |
| LIF | threshold | V_th | −50 | mV | |
| LIF | membrane resistance | R_m | 10 | MΩ | rheobase 1.5 nA |
| LIF | absolute refractory period | t_ref | 2 | ms | V clamped to V_reset |
| LIF | f-I test | — | 0–5 nA (26 values), 2 s, Δt = 0.05 ms; error test 1.6–5 nA (18 values), Δt = 0.01–1 ms | — | rate = 1/mean ISI |
| Izhikevich | (a, b, c, d) | — | RS (0.02, 0.2, −65, 8); IB (0.02, 0.2, −55, 4); CH (0.02, 0.2, −50, 2); FS (0.1, 0.2, −65, 2); LTS (0.02, 0.25, −65, 2) | —, —, mV, — | Izhikevich 2003 |
| Izhikevich | integration | — | forward Euler, Δt = 0.1 ms (Fig. 4), 0.25 ms (classifier data, benchmark); v₀ = −65 mV, u₀ = b·v₀ | — | spike when v ≥ 30 mV (checked at start of step) |
| Izhikevich | stimulus (Fig. 4) | I | 10 from t = 20 ms, 250 ms | dimensionless | |
| Network | cells | N_E, N_I | 800, 200 | — | |
| Network | connection probability | p | 0.1 (Bernoulli, no autapses) | — | in-degree varies (binomial) |
| Network | excitatory jump | J | 0.2 | mV | delta synapse, effective delay = Δt = 0.1 ms |
| Network | inhibitory jump | −gJ | g = 1–8 | mV | |
| Network | external input | ν_ext | 7 (sweep), 8 (g = 2 example) | spikes/ms per cell (kHz) | each event +J |
| Network | integration | — | Euler, Δt = 0.1 ms; V₀ ~ U(V_reset, V_th) | — | 400 ms (sweep), 500 ms (examples); analysis after 100 ms |
| Network | statistics | — | rate; mean CV of ISIs (every 5th cell, > 3 spikes); Fano factor of population count in 1 ms bins | — | seeds 0–4 (sweep), 0 (examples), 3 (Fig. 1) |
| Classifier data | cells | — | 120 per class × 5 = 600 (seed 0); independent test set 600 (seed 1) | — | |
| Classifier data | heterogeneity | — | a, b, d × (1 + 0.1·N(0,1)); c + 2·N(0,1) mV | — | |
| Classifier data | input | — | step from 20 ms, amplitude U(6, 14); white noise σ = 2 (Euler-Maruyama, per-step SD σ/√Δt = 4) | — | 500 ms, Δt = 0.25 ms |
| Classifier data | features | — | log₁₀(1 + rate in 20–500 ms); CV of ISI; adaptation index; fraction of ISIs < 10 ms; first-spike latency capped at 100 ms (CV, adaptation, burst set to 0 if < 3 spikes; latency 480 ms → capped 100 ms if no spike) | — | |
| MLP | architecture | — | 5-16-16-5, ReLU, softmax, He initialisation (453 parameters) | — | |
| MLP / LR | training | — | mini-batch SGD, batch ≈ 32, lr 0.05, momentum 0.9, L2 10⁻⁴ (weights only), 150 epochs, features z-scored on the training set | — | no early stopping; hyperparameters fixed a priori |
| Evaluation | resampling | — | 10 stratified 70/30 splits (seeds 0–9; 420 train / 180 test); independent test set of 600 cells (seed 1) | — | model seed = split seed; corrected resampled t (Nadeau & Bengio 2003) |

# Table S2.2. Per-class performance on the independent test set (n = 120 cells per class)

Recall (sensitivity) and precision with 95% stratified-bootstrap intervals over cells (1,000 resamples). Percentile intervals that reach 1.000 are unreliable near the boundary.

| Class | MLP recall | MLP precision | MLP F1 | LR recall | LR precision | LR F1 |
|---|---|---|---|---|---|---|
| RS | 0.958 [0.925, 0.992] | 0.975 [0.943, 1.000] | 0.966 | 0.992 [0.975, 1.000] | 0.960 [0.922, 0.992] | 0.975 |
| IB | 0.967 [0.933, 0.992] | 0.921 [0.874, 0.966] | 0.943 | 0.900 [0.842, 0.950] | 0.956 [0.918, 0.991] | 0.927 |
| CH | 0.983 [0.958, 1.000] | 0.992 [0.975, 1.000] | 0.987 | 0.983 [0.958, 1.000] | 0.992 [0.975, 1.000] | 0.987 |
| FS | 0.983 [0.958, 1.000] | 1.000 [1.000, 1.000] | 0.992 | 1.000 [1.000, 1.000] | 0.968 [0.938, 0.992] | 0.984 |
| LTS | 0.967 [0.933, 0.992] | 0.975 [0.944, 1.000] | 0.971 | 0.942 [0.892, 0.983] | 0.942 [0.898, 0.982] | 0.942 |

# Table S2.3. Network sweep: five seeds per value of g (ν_ext = 7 kHz, 400 ms)

Mean ± SD over seeds 0–4. With n = 5 the half-width of a 95% t interval is 1.24 × SD. For independent Poisson firing the synchrony index is close to 1.

| g | Rate (Hz) | CV of ISIs | Synchrony index (Fano factor) |
|---|---|---|---|
| 1 | 43.8 ± 2.1 | 0.055 ± 0.004 | 736 ± 18 |
| 2 | 43.2 ± 0.2 | 0.076 ± 0.003 | 578 ± 32 |
| 3 | 35.7 ± 1.8 | 0.247 ± 0.011 | 77 ± 37 |
| 4 | 25.6 ± 1.6 | 0.313 ± 0.011 | 21.6 ± 7.1 |
| 5 | 19.4 ± 1.0 | 0.344 ± 0.014 | 9.9 ± 3.1 |
| 6 | 16.5 ± 1.0 | 0.353 ± 0.018 | 8.5 ± 2.1 |
| 7 | 14.3 ± 1.1 | 0.366 ± 0.015 | 6.6 ± 0.9 |
| 8 | 12.6 ± 0.8 | 0.369 ± 0.019 | 5.5 ± 1.3 |

# Table S2.5. Full classifier statistics (moved from the main text)

The main text reports only headline accuracy, the McNemar test and one sentence on calibration (Results 3.6). This table gives the complete evaluation.

**Procedure.** *Repeated splits:* ten stratified 70/30 splits of the 600 development cells (420 training, 180 test; seeds 0–9). Because the training sets overlap, split estimates are correlated and naive intervals are too narrow (Dietterich 1998). Intervals use the corrected resampled t statistic of Nadeau and Bengio (2003): the variance of the J = 10 split estimates is inflated by 1/J + n_test/n_train (= 0.1 + 180/420), with 9 degrees of freedom; the same procedure is applied to the paired MLP − LR difference. *Independent test set:* 600 new cells (seed 1); each model trained once on all 600 development cells. Intervals are stratified bootstrap percentile intervals over cells (2,000 resamples). Paired predictions are compared with the exact McNemar test. Calibration: Brier score (multiclass, mean over cells of the summed squared error of the probability vector), log-loss, and expected calibration error (ECE; ten equal-width bins of the top-class probability). Variability over initialisation: the MLP and LR were retrained with ten initialisation seeds on the development set and evaluated on the independent test set.

| Metric | MLP | Logistic regression | Difference (MLP − LR) |
|---|---|---|---|
| **Repeated splits (n = 10; 420 train / 180 test)** | | | |
| Accuracy, mean (SD) | 95.5% (1.3) | 94.7% (0.7) | +0.8 points |
| 95% CI, corrected resampled t (df = 9) | 93.4 to 97.6 | 93.6 to 95.7 | −1.6 to 3.3; t = 0.77, p = .46 |
| 95% CI, naive (uncorrected; for teaching only) | 94.8 to 96.3 | 94.3 to 95.0 | −0.1 to 1.7 |
| Macro-F1, mean [corrected 95% CI] | 0.955 [0.934, 0.976] | 0.946 [0.935, 0.958] | — |
| Brier score, mean over splits | 0.070 | 0.088 | — |
| **Independent test set (n = 600 new cells)** | | | |
| Accuracy [95% CI] | 97.2% [95.8, 98.3] | 96.3% [94.8, 97.7] | +0.8 points [−0.8, 2.5] |
| Discordant cells; exact McNemar test | — | — | 16 MLP-only vs 11 LR-only correct; p = .44 |
| Macro-F1 [95% CI] | 0.972 [0.959, 0.983] | 0.963 [0.948, 0.977] | — |
| Brier score | 0.044 | 0.067 | — |
| Log-loss | 0.103 | 0.135 | — |
| Expected calibration error (10 bins) | 0.017 | 0.046 | — |
| Accuracy over 10 training seeds, mean (SD; range) | 97.1% (0.5; 96.2–97.7) | 96.4% (0.1; 96.3–96.7) | — |

Notes. The naive intervals are about three times narrower than the corrected ones (width 1.5 vs 4.2 points for the MLP; 0.7 vs 2.1 points for LR), which is the common error discussed in the main text (Section 5.3). On the independent test set, 572 of the MLP's 600 predictions fell in the top probability bin, so the ECE comparison rests mainly on that bin; it was not tested formally. Intrinsically bursting cells were the hardest class for both models (F1 0.94 and 0.93; Table S2.2). The task is easy by construction: the features were chosen to separate classes that the simulator itself defines.

References for Table S2.5: Dietterich TG. Approximate statistical tests for comparing supervised classification learning algorithms. Neural Comput. 1998;10(7):1895-923. doi:10.1162/089976698300017197. Nadeau C, Bengio Y. Inference for the generalization error. Mach Learn. 2003;52(3):239-81. doi:10.1023/A:1024068626366. McNemar Q. Psychometrika. 1947;12(2):153-7. doi:10.1007/BF02295996. Brier GW. Mon Weather Rev. 1950;78(1):1-3. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. PMLR. 2017;70:1321-30.

# Table S2.4. Reporting checklist

Applicable guidance: for the simulations, the ten rules for reproducible computational research (1), the Minimum Information About a Simulation Experiment (MIASE) (2) and the guidelines for describing network models (3); for the classifier, the analysis and reporting items of TRIPOD+AI (4) and the DOME recommendations for machine learning in biology (5). TRIPOD+AI is written for clinical prediction models built on patient data, so only its analysis and reporting items apply. CONSORT and STROBE do not apply (no participants). A future evaluation of learning outcomes should follow a reporting guide for medical-education research.

| Item | Source | Where reported |
|---|---|---|
| Model equations stated | MIASE; Nordlie | Methods 2.3; code `neurosim/hh.py` (α/β rate functions); Table S2.1 |
| All parameter values | MIASE; Nordlie | Table S2.1 |
| Numerical method and step | MIASE | Methods 2.3; convergence: Results 3.2 (HH onset vs Δt), 3.3 (LIF slope 0.98); Workbook Module 7 |
| Initial conditions | MIASE | Methods 2.3 (steady state at −65 mV or 300 ms settling, per experiment); Table S2.1 |
| Network connectivity rule, in-degree, delays | Nordlie | Methods 2.3; Table S2.1 |
| Random seeds | Sandve rule 6 | Table S2.1; `results.json` |
| Software versions | Sandve rule 3 | Methods 2.2; `results.json` (environment block); `requirements.txt` |
| Hardware for timings | Sandve | Methods 2.2, 2.6; `results.json` |
| Code that regenerates every number | Sandve rules 1, 8 | Methods 2.8; `make_figures.py`, `make_supplement_figures.py` |
| Archived code with DOI | Sandve rule 10 | Methods 2.8; Declarations ([Zenodo DOI — to be added]) |
| Data-generating process for ML | DOME "data"; TRIPOD+AI 5–7 | Methods 2.4; Table S2.1 |
| Train/test separation and leakage | DOME; TRIPOD+AI 12 | Methods 2.4 (standardisation on each training set; independent test set) |
| Hyperparameter selection | DOME "optimisation"; TRIPOD+AI 12 | Methods 2.4 (set before evaluation, not tuned on test data) |
| Uncertainty of performance | TRIPOD+AI 12, 23 | Methods 2.4; Table S2.5 (corrected resampled t; bootstrap over cells) |
| Baseline comparison | DOME "evaluation" | Methods 2.4; Results 3.6; Table S2.5 (paired corrected t; exact McNemar) |
| Calibration | TRIPOD+AI 12e, 23 | Table S2.5; Fig. 6F (Brier score, log-loss, ECE) |
| Per-class performance | DOME | Table S2.2 |
| Limitations (simulated data, task easy by construction, squid parameters) | TRIPOD+AI 27 | Discussion 5.7 |
| Agreement between browser simulator and Python package | — | Methods 2.7; Results 3.9 |

## References for Table S2.4

1. Sandve GK, Nekrutenko A, Taylor J, Hovig E. Ten simple rules for reproducible computational research. PLoS Comput Biol. 2013;9(10):e1003285. doi:10.1371/journal.pcbi.1003285.
2. Waltemath D, Adams R, Beard DA, Bergmann FT, Bhalla US, Britten R, et al. Minimum Information About a Simulation Experiment (MIASE). PLoS Comput Biol. 2011;7(4):e1001122. doi:10.1371/journal.pcbi.1001122.
3. Nordlie E, Gewaltig MO, Plesser HE. Towards reproducible descriptions of neuronal network models. PLoS Comput Biol. 2009;5(8):e1000456. doi:10.1371/journal.pcbi.1000456.
4. Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378.
5. Walsh I, Fishman D, Garcia-Gasulla D, Titma T, Pollastri G; ELIXIR Machine Learning Focus Group; et al. DOME: recommendations for supervised machine learning validation in biology. Nat Methods. 2021;18(10):1122-7. doi:10.1038/s41592-021-01205-4.
