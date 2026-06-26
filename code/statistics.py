"""
    This script computes summary statistics for each connectome and writes a
    publication-ready LaTeX table to ../tables/connectome_statistics.tex.
-------------------------------------------------------------------------------
created on:
    Tue 24 Mar 2026
-------------------------------------------------------------------------------
last change:
    Thu 26 Jun 2026
-------------------------------------------------------------------------------
notes:
    Run from the repo/ directory:
        python code/statistics.py
    Output: ../tables/connectome_statistics.tex
             ../tables/large_neuron_statistics.tex
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
tables_dir = os.path.join(_here, '../../tables/')
os.makedirs(tables_dir, exist_ok=True)

# ── Human-readable column labels ──────────────────────────────────────────────
# Column headers rendered with \makecell (requires \usepackage{makecell} in preamble).
# Drosophila labels are split across two rows: genus on top, identifier below.
display_names = {
    'drosophila_whole_brain':    r'\makecell[b]{\textit{Drosophila} \\ FAFB}',
    'drosophila_banc':           r'\makecell[b]{\textit{Drosophila} \\ BANC}',
    'drosophila_manc':           r'\makecell[b]{\textit{Drosophila} \\ MANC}',
    'mouse_retina':              r'\makecell[b]{Mouse retina}',
    'c_elegans':                 r'\makecell[b]{\textit{C. elegans}}',
}

# ── Row labels ─────────────────────────────────────────────────────────────────
stat_labels = [
    'Number of neurons',
    'Number of connections',
    'Average in/out-degree',
    'Average in/out-strength',
    'Average strength per connection',
]

k_min = 10

# ── Compute statistics ─────────────────────────────────────────────────────────
stats = {}
large_stats = {}
for idx, name in enumerate(connectomes):
    if name in display_names.keys():
        A = load_connectome(idx)
        N       = A.shape[0]
        in_deg  = np.asarray(A.getnnz(axis=0)).ravel()   # column nnz = in-degree
        in_str  = np.asarray(A.sum(axis=0)).ravel()

        stats[name] = [
            N,                             # number of neurons
            A.nnz,                         # number of connections
            np.mean(in_deg),               # average in-degree
            np.mean(in_str),               # average in-strength
            A.sum() / A.nnz,               # average strength per connection
        ]

        mask         = in_deg >= k_min
        large_in_deg = in_deg[mask]
        large_in_str = in_str[mask]
        large_stats[name] = [
            int(mask.sum()),                          # neurons with k_in >= 10
            np.mean(in_deg >= k_min),                 # fraction with k_in >= 10
            np.mean(large_in_deg),                    # avg in-degree
            np.mean(large_in_str),                    # avg in-strength
            large_in_str.sum() / large_in_deg.sum(),  # avg strength per connection
        ]

# ── Format individual values ───────────────────────────────────────────────────
def fmt(val, label, connectome=None):
    if label == 'Number of neurons':
        return f'{int(val):,}'
    elif label == 'Number of connections':
        mantissa, exp = f'{val:.2e}'.split('e')
        return f'{mantissa} $\\times$ 10$^{{{int(exp)}}}$'
    elif label in ('Average in/out-strength', 'Average in-strength') and connectome == 'mouse_retina':
        return f'{val:.1f}' + r'$\ \mu$m$^2$'
    elif label == 'Average strength per connection' and connectome == 'mouse_retina':
        return f'{val:.1f}' + r'$\ \mu$m$^2$'
    elif 'Fraction' in label:
        return f'{val:.3f}'
    else:
        return f'{val:.1f}'

# ── Build LaTeX table ──────────────────────────────────────────────────────────
n_cols = len(display_names.keys())
col_labels = display_names.values()

lines = [
    r'\begin{tabular}{l' + 'r' * n_cols + '}',
    r'\toprule',
    'Statistic & ' + ' & '.join(col_labels) + r' \\',
    r'\midrule',
]

for row_idx, label in enumerate(stat_labels):
    vals = [fmt(stats[c][row_idx], label, c) for c in display_names.keys()]
    lines.append(label + ' & ' + ' & '.join(vals) + r' \\')

lines += [
    r'\bottomrule',
    r'\end{tabular}',
]

out_path = tables_dir + 'connectome_statistics.tex'
with open(out_path, 'w') as f:
    f.write('\n'.join(lines) + '\n')

print(f'Table saved to {out_path}')

# ── Build large-neuron LaTeX table ─────────────────────────────────────────────
large_stat_labels = [
    r'Number of neurons with $k_{\mathrm{in}} \geq 10$',
    r'Fraction with $k_{\mathrm{in}} \geq 10$',
    'Average in-degree',
    'Average in-strength',
    'Average strength per connection',
]

large_lines = [
    r'\begin{tabular}{l' + 'r' * n_cols + '}',
    r'\toprule',
    'Statistic & ' + ' & '.join(col_labels) + r' \\',
    r'\midrule',
]

for row_idx, label in enumerate(large_stat_labels):
    vals = [fmt(large_stats[c][row_idx], label, c) for c in display_names.keys()]
    large_lines.append(label + ' & ' + ' & '.join(vals) + r' \\')

large_lines += [
    r'\bottomrule',
    r'\end{tabular}',
]

out_path = tables_dir + 'large_neuron_statistics.tex'
with open(out_path, 'w') as f:
    f.write('\n'.join(large_lines) + '\n')

print(f'Table saved to {out_path}')
