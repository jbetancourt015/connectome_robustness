# Figure Generation Pipeline for "Signatures of Robust Computations in Connectomes"

This repository contains the full computational pipeline used to preprocess connectome data, run simulations, and generate the figures for the associated manuscript.

---

## Repository Structure

    repo/
    ├── code/
    │   ├── data_processing.py
    │   ├── simulations.py
    │   ├── network_processing.py
    │   ├── framework.py
    │   ├── robustness_comparison.py
    │   ├── simulated_loss.py
    │   ├── drosophila_robustness.py
    │   ├── connection_strength_distributions.py
    │   ├── statistics.py
    ├── processed_data/
    ├── simulation_results/
    ├── tables/
    ├── requirements.txt

- `code/` — All scripts for preprocessing, simulations, and figure generation.  
- `processed_data/` — Intermediate outputs from preprocessing.  
- `simulation_results/` — Outputs from simulation runs.  
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

Run the scripts in the following order from the `repo/` directory.

### 1. Preprocess data

```bash
python code/data_processing.py
```

This step processes the raw data and saves intermediate outputs to:

```
processed_data/
```

---

### 2. Run simulations

```bash
python code/simulations.py
```

This step runs the simulation pipeline and saves outputs to:

```
simulation_results/
```

---

### 3. Generate figures

```bash
python code/framework.py
python code/robustness_comparison.py
python code/simulated_loss.py
python code/drosophila_robustness.py
python code/connection_strength_distributions.py
```

Each script saves its figures to disk (see script headers for output paths).

---

### 4. Generate tables

```bash
python code/statistics.py
```

This generates a LaTeX table of summary statistics (neuron counts, connection counts, degree and strength averages, fraction of high-degree neurons) for all eight connectomes, saved to:

```
tables/connectome_statistics.tex
```

---

## Reproducibility Notes

- Preprocessing must be completed before running simulations.
- Simulations must be completed before generating figures.
- If parameters are changed, previously generated outputs in `processed_data/` or `simulation_results/` may need to be deleted and regenerated.
- Large generated outputs should not be committed to version control.

---

## Citation

If using this code, please cite:



---

## License

License information here.
