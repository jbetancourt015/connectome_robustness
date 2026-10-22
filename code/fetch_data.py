"""
    Fetches the raw connectome data files into data/ from the project's
    Zenodo deposit, if they aren't already present. Run this once before
    data_processing.py. See README.md's "Data Sources" section for per-file
    citations and the Zenodo record itself.
-------------------------------------------------------------------------------
created on:
    Fri 16 Oct 2026
-------------------------------------------------------------------------------
last change:
    Mon 19 Oct 2026
-------------------------------------------------------------------------------
notes:
    Uses the public Zenodo REST API (https://developers.zenodo.org/) --
    no authentication is needed to read a published record. The record is
    set by ZENODO_RECORD_ID below.

    Re-running when all expected files are already present in
    data/ does nothing. If a download's checksum doesn't match, the partial
    file is removed so a re-run will retry it.
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
-------------------------------------------------------------------------------
"""

import hashlib
import json
import os
import urllib.request

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
ZENODO_RECORD_ID = "23427196"  # concept DOI 10.5281/zenodo.23427196 (always resolves to latest version)

data_dir = "../data/"

expected_files = [
    "c_elegans.xls",
    "mouse_retina.mat",
    "fafb_connections.csv.gz",
    "fafb_classification.csv.gz",
    "fafb_cell_types.csv.gz",
    "fafb_neurotransmitter_types.csv.gz",
    "banc_connections.csv.gz",
    "manc_connections.csv.gz",
]


def _md5(path, block_size=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def fetch_data():
    os.makedirs(data_dir, exist_ok=True)

    missing = [f for f in expected_files if not os.path.exists(data_dir + f)]
    if not missing:
        print("All expected data files already present in data/ -- nothing to fetch.")
        return

    print(f"Fetching Zenodo record {ZENODO_RECORD_ID}...")
    with urllib.request.urlopen(
        f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
    ) as resp:
        record = json.load(resp)

    files_by_name = {f["key"]: f for f in record["files"]}

    for file_name in missing:
        if file_name not in files_by_name:
            raise RuntimeError(
                f"Expected file '{file_name}' not found in Zenodo record "
                f"{ZENODO_RECORD_ID}. Files in record: {sorted(files_by_name)}"
            )
        entry = files_by_name[file_name]
        url = entry["links"]["self"]
        dest = data_dir + file_name

        print(f"Downloading {file_name} ({entry['size'] / 1e6:.1f} MB)...")
        urllib.request.urlretrieve(url, dest)

        expected_checksum = entry["checksum"].split(":")[-1]
        actual_checksum = _md5(dest)
        if actual_checksum != expected_checksum:
            os.remove(dest)
            raise RuntimeError(
                f"Checksum mismatch for {file_name}: expected "
                f"{expected_checksum}, got {actual_checksum}. File removed; "
                f"please re-run."
            )
        print(f"  Verified {file_name} (md5 {actual_checksum})")

    print("All data files fetched and verified.")


if __name__ == "__main__":
    fetch_data()
