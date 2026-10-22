# Heavy-tailed connections support robust neural communication

This repository contains the full computational pipeline used to preprocess connectome data, run simulations, and generate the figures for the paper "Heavy-tailed connections support robust neural communication" by Jose M. Betancourt, Varun Varanasi, Damon A. Clark, and Christopher W. Lynn. Raw connectome data is hosted on Zenodo and fetched by `code/fetch_data.py` (see "Data Sources" below).

---

## Repository Structure

    .
    ├── code/
    │   ├── fetch_data.py
    │   ├── data_processing.py
    │   ├── simulations.py
    │   ├── params.py
    │   ├── figure_formatting.py
    │   ├── framework.py
    │   ├── robustness_comparison.py
    │   ├── drosophila_robustness.py
    │   ├── connection_strength_distributions.py
    │   ├── simulated_error.py
    │   ├── connectome_statistics.py
    ├── data/              (empty until `fetch_data.py` is run — see Data Sources)
    ├── processed_data/
    ├── simulation_results/
    ├── figures/
    ├── tables/
    ├── requirements.txt

- `code/` — All scripts for fetching data, preprocessing, simulations, and figure generation.
- `data/` — Raw connectome data, populated by `code/fetch_data.py` from Zenodo. Empty (gitignored) in a fresh clone.
- `processed_data/` — Intermediate outputs from preprocessing. Empty (gitignored) in a fresh clone.
- `simulation_results/` — Outputs from simulation runs. Empty (gitignored) in a fresh clone.
- `figures/` / `tables/` — Output figures and LaTeX tables. Empty (gitignored) in a fresh clone.
- `requirements.txt` — Python dependencies required to run the pipeline.

---

## Setup

Requires **Python 3.14.3**. It is recommended to use a virtual environment.

### Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Reproducing the Results

Run the scripts in the following order from the `code/` directory (scripts resolve `data/`, `processed_data/`, `simulation_results/`, `figures/`, and `tables/` as relative paths from here).

### 0. Fetch data

```bash
cd code
python fetch_data.py
```

Downloads the raw connectome data (see "Data Sources" below) from Zenodo into `data/`. Skips any file already present, so it's safe to re-run.

---

### 1. Preprocess data

```bash
python data_processing.py
```

This step processes the raw data in `data/` and saves intermediate outputs to:

```
processed_data/
```

---

### 2. Run simulations

```bash
python simulations.py
```

This step runs the simulation pipeline and saves outputs to:

```
simulation_results/
```

---

### 3. Generate figures

```bash
python robustness_comparison.py
python drosophila_robustness.py
python connection_strength_distributions.py
python simulated_error.py
python framework.py
```

Each script saves its figures to disk (see script headers for output paths).

---

### 4. Generate tables

```bash
python connectome_statistics.py
```

This generates a LaTeX table of summary statistics (neuron counts, connection counts, degree and strength averages, fraction of high-degree neurons) for all five connectomes, saved to:

```
tables/connectome_statistics.tex
```

---

## Reproducibility Notes

- Preprocessing must be completed before running simulations.
- Simulations must be completed before generating figures.
- If parameters are changed, previously generated outputs in `processed_data/` or `simulation_results/` may need to be deleted and regenerated.

---

## Data Sources

Raw connectome data is hosted on Zenodo: **https://doi.org/10.5281/zenodo.23427196**. Run `code/fetch_data.py` to download it into `data/` (see "0. Fetch data" above). `code/data_processing.py` processes these files into the five connectomes analyzed in the manuscript. Citations, included files, and access dates:

### C. elegans

- **Files:** `c_elegans.xls`

Varshney, L. R., Chen, B. L., Paniagua, E., Hall, D. H. & Chklovskii, D. B. Structural
Properties of the *Caenorhabditis elegans* Neuronal Network. *PLoS Computational Biology*
7, e1001066 (2011).
White, J. G., Southgate, E., Thomson, J. N. & Brenner, S. The structure of the nervous
system of the nematode *Caenorhabditis elegans*. *Philosophical Transactions of the Royal
Society of London. B, Biological Sciences* 314, 1–340 (1986).
Accessed October 2026.

**Selection procedure:** the raw source (`data/c_elegans.xls`) is a WormAtlas export listing every neuron-pair contact by type (`S`/`Sp` = directed
chemical synapse, `R`/`Rp` = mirror of a chemical synapse from the receiving neuron's
perspective, `EJ` = electrical/gap junction, `NMJ` = neuromuscular junction). We keep
chemical synapses (`S`, `Sp`) and gap junctions (`EJ`), summed per neuron pair as the
connection strength.

### Mouse retina

- **Files:** `mouse_retina.mat`

Helmstaedter, M. et al. Connectomic reconstruction of the inner plexiform layer in the
mouse retina. *Nature* 500, 168–174 (2013).
Accessed October 2026.

**Selection procedure:** mouse retina contacts are classified by membrane contact area
(inherently undirected) in the original per-interface connection list (`data/mouse_retina.mat`).
Cell IDs 1–840 are classified neurons; 841–950 are orphans and 951–1123 are glia
(Helmstaedter et al. 2013, Supplementary Data 6). We restrict to neuron-neuron contacts (cell ID ≤ 840
on both ends).

### FAFB (Drosophila whole brain)

- **Files:** `fafb_connections.csv.gz`, `fafb_classification.csv.gz`, `fafb_cell_types.csv.gz`, `fafb_neurotransmitter_types.csv.gz`

Dorkenwald, S. et al. Neuronal wiring diagram of an adult brain. *Nature* 634, 124–138
(2024).
Schlegel, P. et al. Whole-brain annotation and multi-connectome cell typing of *Drosophila*.
*Nature* 634, 139–152 (2024).
Accessed March 2025.

### BANC (Drosophila brain-and-cord)

- **Files:** `banc_connections.csv.gz`

Bates, A. S. et al. Distributed control circuits across a brain-and-cord connectome.
*Nature* 656, 957–970 (2026).
Accessed October 2026.

### MANC (Drosophila male nerve cord)

- **Files:** `manc_connections.csv.gz`

Takemura, S.-y. et al. A connectome of the male *Drosophila* ventral nerve cord. *eLife*
13, RP97769 (2024).
Accessed October 2026.
