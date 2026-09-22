"""
Global simulation parameters for the connectome robustness pipeline.
Edit values here; all scripts that import this file pick up the changes.
-------------------------------------------------------------------------------
created on:
    Wed 16 Apr 2026
-------------------------------------------------------------------------------
last change:
    Tue 22 Sep 2026
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
-------------------------------------------------------------------------------
"""

# ── Global ─────────────────────────────────────────────────────────────────────
rng_seed = 1674  # global random seed for reproducibility
block_perturb = 128  # block size for memory-efficient Monte Carlo perturbation loop

# ── FlyWire error simulation ───────────────────────────────────────────────────
error_sigma = 1.0
error_n_draws = int(1e3)
error_n_perturb = int(1e3)
error_p_fire_vals = [0.5, 0.01, 0.03, 0.1, 0.3]  # sigma fixed at error_sigma
sparse_p_fire_vals = error_p_fire_vals[1:]  # firing probabilities for the sparse_error_vs_robustness sweep (matches error_p_fire_vals minus the 0.5 baseline)
error_sigma_vals = [0.1, 0.3, 1.0, 3.0, 10.0]  # p_fire fixed at 0.5
error_sweep_n_draws = int(5e2)
error_sweep_n_perturb = int(5e2)

# ── FlyWire periphery scoring ──────────────────────────────────────────────────
periphery_n_sim = int(1e3)
periphery_threshold = 0.3
periphery_repeats = 10

# ── z/ztilde simulation ────────────────────────────────────────────────────────
# Each entry is (distribution, mean) or (distribution, mean, variance).
zztilde_param_sets = [
    ("dirac", 1.0),
    ("gamma", 1.0, 0.8),
    ("gamma", 1.0, 3.0),
]
zztilde_n_inputs = int(1e3)
zztilde_eps = 1.0
zztilde_n_draws = int(5e3)
zztilde_n_perturb = int(5e3)

# ── Parametric distribution simulations ───────────────────────────────────────
parametric_n_neurons = int(1e3)
parametric_n_inputs = int(1e3)
parametric_n_draws = int(1e3)
parametric_n_perturb = int(1e3)
parametric_mean_vals = [
    2.0,
    4.0,
    8.0,
    16.0,
]  # manually chosen mean values for parametric sims
parametric_n_var = 10
parametric_sparse_n_draws = int(5e2)  # reduced precision for the p_fire sweep
parametric_sparse_n_perturb = int(5e2)

# ── Neurotransmitter classification ───────────────────────────────────────────
nt_class_thresh = 0.6  # min outgoing fraction to assign exc/inh nt_class per connection
nt_plurality_thresh = (
    0.7  # min outgoing fraction to assign a specific nt_type per neuron
)

# ── Network shuffling ──────────────────────────────────────────────────────────
shuffle_k_min = 10
shuffle_n_threshold_default = 0
shuffle_n_threshold_banc = (
    3  # BANC connectome requires minimum 3 synapses per connection
)
