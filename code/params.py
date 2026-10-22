"""
Global simulation parameters for the connectome robustness pipeline.
Edit values here; all scripts that import this file pick up the changes.
-------------------------------------------------------------------------------
created on:
    Wed 16 Apr 2026
-------------------------------------------------------------------------------
last change:
    Wed 21 Oct 2026
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
-------------------------------------------------------------------------------
"""

# ── Connectomes ────────────────────────────────────────────────────────────────
connectomes = [
    "c_elegans",
    "mouse_retina",
    "fafb",
    "banc",
    "manc",
]  # canonical names; also the stems of processed_data/*.npz
connectome_colors = {  # RGB in [0, 1], shared by all figure scripts
    name: tuple(c / 255 for c in rgb)
    for name, rgb in {
        "fafb": (0, 77, 128),
        "banc": (181, 23, 0),
        "manc": (1, 113, 0),
        "c_elegans": (242, 112, 0),
        "mouse_retina": (120, 0, 150),
    }.items()
}

# ── Global ─────────────────────────────────────────────────────────────────────
rng_seed = 1674  # global random seed for reproducibility
block_perturb = 128  # block size for memory-efficient Monte Carlo perturbation loop

# ── FAFB error simulation ───────────────────────────────────────────────────
error_sigma = 1.0
error_n_draws = int(1e3)
error_n_perturb = int(1e3)
error_p_fire_vals = [0.5, 0.02, 0.05, 0.1, 0.2]  # sigma fixed at error_sigma
sparse_p_fire_vals = error_p_fire_vals[
    1:
]  # firing probabilities for the sparse_error_vs_robustness sweep (matches error_p_fire_vals minus the 0.5 baseline)
error_sigma_vals = [0.2, 0.5, 1.0, 2.0, 5.0]  # p_fire fixed at 0.5
error_sweep_n_draws = int(1e3)
error_sweep_n_perturb = int(1e3)

# ── FAFB periphery scoring ──────────────────────────────────────────────────
periphery_n_sim = int(1e3)
periphery_threshold = 0.3
periphery_repeats = 10
periphery_max_steps = None  # no cap on propagation steps

# ── z/ztilde simulation ────────────────────────────────────────────────────────
# Each entry is (distribution, mean) or (distribution, mean, variance).
zztilde_param_sets = [
    ("dirac", 1.0),
    ("gamma", 1.0, 0.8),
    ("gamma", 1.0, 3.0),
]
zztilde_n_inputs = int(1e3)
zztilde_sigma = 1.0
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
parametric_sparse_n_draws = int(1e3)  # reduced precision for the p_fire sweep
parametric_sparse_n_perturb = int(1e3)

# ── Network shuffling ──────────────────────────────────────────────────────────
shuffle_k_min = 10
shuffle_n_threshold_default = 0
shuffle_n_threshold_banc = (
    3  # BANC connectome requires minimum 3 synapses per connection
)
