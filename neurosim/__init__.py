"""NeuroSim-Edu: neuron models from Hodgkin-Huxley to artificial networks,
written with NumPy only for teaching."""
from .hh import SQUID, gate_rates, steady_state, simulate_hh, spike_times, hh_rate, settle
from .point import (LIF_CELL, simulate_lif, lif_rate_closed_form,
                    IZH_CLASSES, simulate_izhikevich, izhikevich_cell)
from .network import simulate_ei_network, network_statistics
from .classifier import (FEATURES, CLASS_KEYS, spike_train_features, make_dataset,
                         stratified_split, SpikeClassNet, cross_entropy,
                         confusion_matrix, macro_f1, bootstrap_ci,
                         corrected_resampled_t, per_class_metrics, brier_score,
                         calibration_curve, mcnemar_exact, stratified_bootstrap, t_ppf)
from .clinic import (sodium_block, potassium_block, extracellular_potassium,
                     fi_curve, spike_width)

__version__ = "1.1.0"
