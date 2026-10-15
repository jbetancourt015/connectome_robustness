"""
    This script computes summary statistics for each connectome and writes a
    publication-ready LaTeX table to ../tables/connectome_statistics.tex.
-------------------------------------------------------------------------------
created on:
    Tue 24 Mar 2026
-------------------------------------------------------------------------------
last change:
    Thu 15 Oct 2026
-------------------------------------------------------------------------------
notes:
    Run from the repo/ directory:
        python code/connectome_statistics.py
    Output: ../tables/connectome_statistics.tex

    Renamed from statistics.py: that name shadowed Python's stdlib `statistics`
    module (script's own directory is prepended to sys.path), which broke
    seaborn's internal `from statistics import NormalDist` import whenever
    another script in code/ ran in the same process/directory context.
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
-------------------------------------------------------------------------------
"""

import os
import sys
import numpy as np

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass  # interactive kernel: relies on jupyter.notebookFileRoot = repo/code/
from network_processing import load_connectome, connectomes

_here = os.path.dirname(os.path.abspath(__file__))
tables_dir = os.path.join(_here, "../../tables/")
os.makedirs(tables_dir, exist_ok=True)

# ── Human-readable column labels ──────────────────────────────────────────────
# Column headers rendered with \makecell (requires \usepackage{makecell} in preamble).
# Drosophila labels are split across two rows: genus on top, identifier below.
display_names = {
    "drosophila_whole_brain": r"\makecell[b]{\textit{Drosophila} \\ FAFB\cite{dorkenwald_neuronal_2024, schlegel_whole-brain_2024}}",
    "drosophila_banc": r"\makecell[b]{\textit{Drosophila} \\ BANC\cite{bates_distributed_2025}}",
    "drosophila_manc": r"\makecell[b]{\textit{Drosophila} \\ MANC\cite{takemura_connectome_2024}}",
    "mouse_retina": r"\makecell[b]{Mouse retina\cite{helmstaedter_connectomic_2013}}",
    "c_elegans": r"\makecell[b]{\textit{C. elegans}\cite{varshney_structural_2011, white_structure_1986}}",
}

# ── Row labels ─────────────────────────────────────────────────────────────────
stat_labels = [
    "Number of neurons",
    "Number of connections",
    "Incoming connections per neuron",
    "Synapses per connection (contact area for mouse retina)",
    r"Fraction of neurons with $k_{\mathrm{in}} \geq 10$",
]

k_min = 10

# ── Compute statistics ─────────────────────────────────────────────────────────
stats = {}
for idx, name in enumerate(connectomes):
    if name in display_names.keys():
        A = load_connectome(idx)
        N = A.shape[0]
        in_deg = np.asarray(A.getnnz(axis=0)).ravel()  # column nnz = in-degree
        in_str = np.asarray(A.sum(axis=0)).ravel()

        stats[name] = [
            N,  # number of neurons
            A.nnz,  # number of connections
            np.mean(in_deg),  # average number of incoming connections
            A.sum() / A.nnz,  # average strength per connection
            np.mean(in_deg >= k_min),  # fraction with k_in >= 10
        ]


# ── Format individual values ───────────────────────────────────────────────────
def fmt(val, label, connectome=None):
    if "Number of neurons" in label:
        return f"{int(val):,}"
    elif label == "Number of connections":
        mantissa, exp = f"{val:.2e}".split("e")
        return f"{mantissa} $\\times$ 10$^{{{int(exp)}}}$"
    elif "contact area" in label and connectome == "mouse_retina":
        return f"{val:.1f}" + r"$\ \mu$m$^2$"
    elif "Fraction" in label:
        return f"{val:.3f}"
    else:
        return f"{val:.1f}"


# ── Build LaTeX table ──────────────────────────────────────────────────────────
n_cols = len(display_names.keys())
col_labels = display_names.values()

lines = [
    r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X "
    + "r" * n_cols
    + r"@{}}",
    r"\toprule",
    "Statistic & " + " & ".join(col_labels) + r" \\",
    r"\midrule",
]

for row_idx, label in enumerate(stat_labels):
    vals = [fmt(stats[c][row_idx], label, c) for c in display_names.keys()]
    lines.append(label + " & " + " & ".join(vals) + r" \\")

lines += [
    r"\bottomrule",
    r"\end{tabularx}",
]

out_path = tables_dir + "connectome_statistics.tex"
with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Table saved to {out_path}")
