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
    │   ├── flywire_robustness_plots.py
    │   ├── connection_strength_distributions.py
    ├── processed_data/
    ├── simulation_results/
    ├── dependencies.txt

- `code/` — All scripts for preprocessing, simulations, and figure generation.  
- `processed_data/` — Intermediate outputs from preprocessing.  
- `simulation_results/` — Outputs from simulation runs.  
- `dependencies.txt` — Python dependencies required to run the pipeline.

---

## Setup

It is recommended to use a conda environment.

### Create and activate a conda environment

```bash
conda create -n connectome_robustness python=3.11
conda activate connectome_robustness
```

### Install dependencies

```bash
pip install -r dependencies.txt
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
python code/flywire_robustness_plots.py
python code/connection_strength_distributions.py
```

Each script saves its figures to disk (see script headers for output paths).

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
