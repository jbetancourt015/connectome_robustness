"""
    This script processes the raw connectome data and generates all processed
    datasets (sparse matrices and parquet files) used by the figure-generation
    scripts.
-------------------------------------------------------------------------------
created on:
    Mon 12 May 2024
-------------------------------------------------------------------------------
last change:
    Tue 20 Oct 2026
-------------------------------------------------------------------------------
notes:
    Reads from data/ (the publication data drop; see data/README.md for
    citations and access dates).
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
-------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import pickle
from scipy.sparse import coo_matrix, save_npz
from scipy.io import loadmat
import params

# ------------------------------------------------------------------------------
# CONNECTOMES AND DIRECTORIES
# ------------------------------------------------------------------------------
data_dir = "../data/"
processed_dir = "../processed_data/"

# Raw input file for each connectome (the .npz output is named after the key)
raw_files = {
    "c_elegans": "c_elegans.xls",
    "mouse_retina": "mouse_retina.mat",
    "fafb": "fafb_connections.csv.gz",
    "banc": "banc_connections.csv.gz",
    "manc": "manc_connections.csv.gz",
}

# ------------------------------------------------------------------------------
# PROCESSING C. ELEGANS
# ------------------------------------------------------------------------------
# C. elegans connections are restricted to chemical synapses (S, Sp) and electrical junctions (EJ) only.
celegans_raw = pd.read_excel(data_dir + raw_files["c_elegans"])
chem = celegans_raw[celegans_raw["Type"].isin(["S", "Sp", "EJ"])]
df = (
    chem.groupby(["Neuron 1", "Neuron 2"])["Nbr"]
    .sum()
    .reset_index()
    .rename(columns={"Neuron 1": "pre", "Neuron 2": "post", "Nbr": "strength"})
)

# Get unique node IDs
all_nodes = pd.unique(df[["pre", "post"]].values.ravel())
N = len(all_nodes)
node_to_idx = {node: i for i, node in enumerate(all_nodes)}

# Apply mapping to row/col arrays
rows = df["pre"].map(node_to_idx).values
cols = df["post"].map(node_to_idx).values
data = df["strength"].values

# Create sparse matrix using SciPy
A_coo = coo_matrix((data, (rows, cols)), shape=(N, N))
A_csc = A_coo.tocsc()

# Save it to disk
save_npz(processed_dir + "c_elegans.npz", A_csc)

# ------------------------------------------------------------------------------
# PROCESSING MOUSE RETINA
# ------------------------------------------------------------------------------
# Mouse retina connections are symmetric and restricted to neuron-neuron contacts only (glia and orphans are dropped).
NEURON_MAX_ID = 840

mat = loadmat(data_dir + raw_files["mouse_retina"])
arr = mat["kn_allContactData_Interfaces_duplCorr_output_IDconv"]
pre, post, area = arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2]

# Restrict to neuron-neuron contacts (drops the 110 orphans + 173 glia and
# every contact touching them)
neuron_mask = (pre <= NEURON_MAX_ID) & (post <= NEURON_MAX_ID)
pre, post, area = pre[neuron_mask], post[neuron_mask], area[neuron_mask]

# Aggregate individual contact-interface segments by unordered cell pair
rows_a = np.minimum(pre, post)
rows_b = np.maximum(pre, post)
agg = (
    pd.DataFrame({"a": rows_a, "b": rows_b, "area": area})
    .groupby(["a", "b"])["area"]
    .sum()
    .reset_index()
)

# Build a symmetric matrix, indexed directly by cell ID (1..840 -> 0..839)
rows = np.concatenate([agg["a"].values - 1, agg["b"].values - 1])
cols = np.concatenate([agg["b"].values - 1, agg["a"].values - 1])
data = np.concatenate([agg["area"].values, agg["area"].values])

A_coo = coo_matrix((data, (rows, cols)), shape=(NEURON_MAX_ID, NEURON_MAX_ID))
A_csc = A_coo.tocsc()

# Save it to disk
save_npz(processed_dir + "mouse_retina.npz", A_csc)

# ------------------------------------------------------------------------------
# PROCESSING FAFB-SCHEMA CONNECTOMES (FAFB, BANC, MANC)
# ------------------------------------------------------------------------------
for name in ["fafb", "banc", "manc"]:
    # Load dataset into pandas DataFrame
    df = pd.read_csv(data_dir + raw_files[name], compression="gzip")

    # Sum over neuropils
    weights = (
        df.groupby(["pre_root_id", "post_root_id"])["syn_count"].sum().reset_index()
    )

    # Get unique node IDs
    all_nodes = pd.unique(weights[["pre_root_id", "post_root_id"]].values.ravel())
    N = len(all_nodes)
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}

    # Apply mapping to row/col arrays
    rows = weights["pre_root_id"].map(node_to_idx).values
    cols = weights["post_root_id"].map(node_to_idx).values
    data = weights["syn_count"].values

    # Create sparse matrix using SciPy
    A_coo = coo_matrix((data, (rows, cols)), shape=(N, N))
    A_csc = A_coo.tocsc()

    # Save it to disk
    save_npz(processed_dir + f"{name}.npz", A_csc)

# ------------------------------------------------------------------------------
# PROCESSING FAFB NEURON-LEVEL DATASET
# ------------------------------------------------------------------------------
# Load FAFB datasets
conn_file = "fafb_connections.csv.gz"
types_file = "fafb_cell_types.csv.gz"
class_file = "fafb_classification.csv.gz"

conn_df = pd.read_csv(data_dir + conn_file, compression="gzip")
types_df = pd.read_csv(data_dir + types_file, compression="gzip")
class_df = pd.read_csv(data_dir + class_file, compression="gzip")

# Map of neuropil to coarse brain region
region_map = {
    "AME_L": "Optic Lobe",
    "LA_L": "Optic Lobe",
    "LO_L": "Optic Lobe",
    "LOP_L": "Optic Lobe",
    "ME_L": "Optic Lobe",
    "AME_R": "Optic Lobe",
    "LA_R": "Optic Lobe",
    "LO_R": "Optic Lobe",
    "LOP_R": "Optic Lobe",
    "ME_R": "Optic Lobe",
    "BU_L": "Lateral Complex",
    "LAL_L": "Lateral Complex",
    "GA_L": "Lateral Complex",
    "BU_R": "Lateral Complex",
    "LAL_R": "Lateral Complex",
    "GA_R": "Lateral Complex",
    "LH_L": "Lateral Horn",
    "LH_R": "Lateral Horn",
    "CAN_L": "Periesophageal NP",
    "AMMC_L": "Periesophageal NP",
    "FLA_L": "Periesophageal NP",
    "CAN_R": "Periesophageal NP",
    "AMMC_R": "Periesophageal NP",
    "FLA_R": "Periesophageal NP",
    "SAD": "Periesophageal NP",
    "PRW": "Periesophageal NP",
    "ICL_L": "Inferior NP",
    "IB_L": "Inferior NP",
    "ATL_L": "Inferior NP",
    "CRE_L": "Inferior NP",
    "SCL_L": "Inferior NP",
    "ICL_R": "Inferior NP",
    "IB_R": "Inferior NP",
    "ATL_R": "Inferior NP",
    "CRE_R": "Inferior NP",
    "SCL_R": "Inferior NP",
    "VES_L": "Ventromedial NP",
    "GOR_L": "Ventromedial NP",
    "SPS_L": "Ventromedial NP",
    "IPS_L": "Ventromedial NP",
    "EPA_L": "Ventromedial NP",
    "VES_R": "Ventromedial NP",
    "GOR_R": "Ventromedial NP",
    "SPS_R": "Ventromedial NP",
    "IPS_R": "Ventromedial NP",
    "EPA_R": "Ventromedial NP",
    "MB_PED_L": "Mushroom Body",
    "MB_VL_L": "Mushroom Body",
    "MB_ML_L": "Mushroom Body",
    "MB_CA_L": "Mushroom Body",
    "MB_PED_R": "Mushroom Body",
    "MB_VL_R": "Mushroom Body",
    "MB_ML_R": "Mushroom Body",
    "MB_CA_R": "Mushroom Body",
    "AL_L": "Antennal Lobe",
    "AL_R": "Antennal Lobe",
    "SLP_L": "Superior NP",
    "SIP_L": "Superior NP",
    "SMP_L": "Superior NP",
    "SLP_R": "Superior NP",
    "SIP_R": "Superior NP",
    "SMP_R": "Superior NP",
    "AVLP_L": "Ventrolateral NP",
    "PVLP_L": "Ventrolateral NP",
    "WED_L": "Ventrolateral NP",
    "PLP_L": "Ventrolateral NP",
    "AOTU_L": "Ventrolateral NP",
    "AVLP_R": "Ventrolateral NP",
    "PVLP_R": "Ventrolateral NP",
    "WED_R": "Ventrolateral NP",
    "PLP_R": "Ventrolateral NP",
    "AOTU_R": "Ventrolateral NP",
    "NO": "Central Complex",
    "PB": "Central Complex",
    "EB": "Central Complex",
    "FB": "Central Complex",
    "GNG": "Gnathal Ganglia",
    "OCG": "Ocelli",
    "UNASGD": "Other Regions",
}

# Classify every synapse record into a brain region directly
conn_df["brain_region"] = conn_df["neuropil"].map(region_map)

# Get brain regions based on incoming connections (plurality over classified synapses)
neuron_df = (
    conn_df.groupby(["post_root_id", "brain_region"])["syn_count"]
    .sum()
    .reset_index()
    .sort_values(["post_root_id", "syn_count"], ascending=[True, False])
    .drop_duplicates("post_root_id")
    .loc[:, ["post_root_id", "brain_region"]]
)

neuron_df = neuron_df.rename(columns={"post_root_id": "root_id"})

# Compute aggregate weights
weights = (
    conn_df.groupby(["pre_root_id", "post_root_id"])["syn_count"]
    .sum()
    .rename("weight")
    .reset_index()
)

# Compute statistics of incoming connections
in_stats = (
    weights.groupby("post_root_id")
    .agg(
        in_deg=("pre_root_id", "nunique"),
        in_strength=("weight", "sum"),
        sum_w2=("weight", lambda s: (s**2).sum()),
    )
    .assign(
        robustness=lambda d: np.sqrt(d["sum_w2"] / d["in_strength"]),
        norm_robustness=lambda d: (
            (
                np.sqrt(d["sum_w2"] / d["in_strength"])
                - np.sqrt(d["in_strength"] / d["in_deg"])
            )
            / (
                np.sqrt(1.0 + (d["in_strength"] / d["in_deg"]))
                - np.sqrt(d["in_strength"] / d["in_deg"])
            )
        ),
    )
    .reset_index()
    .rename(columns={"post_root_id": "root_id"})
)

# Compute statistics of outgoing connections
out_stats = (
    weights.groupby("pre_root_id")
    .agg(out_deg=("post_root_id", "nunique"), out_strength=("weight", "sum"))
    .reset_index()
    .rename(columns={"pre_root_id": "root_id"})
)

# Compute reciprocity of neurons
in_neighbors = (
    weights.groupby("post_root_id")["pre_root_id"]
    .agg(lambda x: set(x))
    .reset_index()
    .rename(columns={"post_root_id": "root_id", "pre_root_id": "in_neighbors"})
)

out_neighbors = (
    weights.groupby("pre_root_id")["post_root_id"]
    .agg(lambda x: set(x))
    .reset_index()
    .rename(columns={"pre_root_id": "root_id", "post_root_id": "out_neighbors"})
)

recip = in_neighbors.merge(out_neighbors, on="root_id", how="outer")

recip["in_neighbors"] = recip["in_neighbors"].apply(
    lambda s: s if isinstance(s, set) else set()
)
recip["out_neighbors"] = recip["out_neighbors"].apply(
    lambda s: s if isinstance(s, set) else set()
)


def _reciprocity(row):
    inter = row["in_neighbors"] & row["out_neighbors"]
    union = row["in_neighbors"] | row["out_neighbors"]
    return (
        len(inter) / len(union) if len(union) > 0 else 0.0
    )  # 0 if completely isolated


recip["reciprocity"] = recip.apply(_reciprocity, axis=1)
recip = recip[["root_id", "reciprocity"]]

# Put all neuron-level stats together
node_stats = in_stats.merge(out_stats, on="root_id", how="outer").merge(
    recip, on="root_id", how="outer"
)

# Append neuron robustness and primary type
neuron_df = neuron_df.merge(node_stats, on="root_id")
neuron_df = neuron_df.merge(types_df, on="root_id").drop("additional_type(s)", axis=1)
neuron_df = neuron_df.merge(class_df, on="root_id")

# Set network statistics to 0 if there are no neighbors
for col in ["in_deg", "in_strength", "out_deg", "out_strength"]:
    neuron_df[col] = neuron_df[col].fillna(0)

# Assign neurotransmitter type from the fafb_neurotransmitter_types external classification
nt_types_df = pd.read_csv(
    data_dir + "fafb_neurotransmitter_types.csv.gz",
    usecols=["root_id", "nt_type"],
    compression="gzip",
)
nt_types_df["nt_type"] = nt_types_df["nt_type"].replace("", np.nan)

neuron_df = neuron_df.merge(nt_types_df, on="root_id", how="left")

# Save dataset as parquet
neuron_df.to_parquet(processed_dir + "fafb_neurons.parquet")

# Sort regions by median (excluding Other Regions)
neuron_df_regions = neuron_df[neuron_df["brain_region"] != "Other Regions"].copy()

region_order = (
    neuron_df_regions.groupby("brain_region")["norm_robustness"]
    .median()
    .sort_values()
    .index
)

with open(processed_dir + "region_order.pkl", "wb") as f:
    pickle.dump(region_order, f)

# ------------------------------------------------------------------------------
# PROCESSING FAFB CONNECTIONS-LEVEL DATASET
# ------------------------------------------------------------------------------
# Aggregate by connection and brain_region (brain_region was classified per-synapse above)
conn_by_region = (
    conn_df.groupby(["pre_root_id", "post_root_id", "brain_region"])["syn_count"]
    .sum()
    .reset_index()
)

# Find brain_region with max synapses for each connection (plurality)
plurality_region = (
    conn_by_region.sort_values(
        ["pre_root_id", "post_root_id", "syn_count"], ascending=[True, True, False]
    )
    .drop_duplicates(["pre_root_id", "post_root_id"])
    .loc[:, ["pre_root_id", "post_root_id", "brain_region"]]
    .rename(columns={"brain_region": "conn_brain_region"})
)

# Compute total synapse count per connection
total_syn = (
    conn_by_region.groupby(["pre_root_id", "post_root_id"])["syn_count"]
    .sum()
    .reset_index()
)

# Merge to get connection dataset with plurality brain_region
conn_df = total_syn.merge(plurality_region, on=["pre_root_id", "post_root_id"])

# Append neuron data
neuron_vars = ["brain_region", "primary_type"]

for var in neuron_vars:
    var_map = neuron_df.set_index("root_id")[var]
    conn_df["pre_" + var] = conn_df["pre_root_id"].map(var_map)
    conn_df["post_" + var] = conn_df["post_root_id"].map(var_map)

# Save dataset as parquet
conn_df.to_parquet(processed_dir + "fafb_connections.parquet")
