"""
Dataset Registry
-----------------
Maps a short dataset key (used in URLs, and as the ResultsExporter
dataset_name) to its source file and display info. Add a new dataset by
adding an entry here — nothing else in tpm/web/ needs to change.
"""

import os

# tpm/web/datasets.py -> tpm/web -> tpm -> project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = {
    "te_process": {
        "display_name": "Tennessee Eastman Process",
        "filepath": os.path.join(PROJECT_ROOT, "te_process_normal.csv"),
        "description": (
            "500 independent time-series runs, 52 numeric variables per run — "
            "treated as the normal/baseline dataset for this analysis. What "
            "each variable represents and what process this data comes from "
            "aren't given; the analysis below infers what it can from the "
            "data itself."
        ),
    },
    "te_process_train_100": {
        "display_name": "Tennessee Eastman Process (100-run sample)",
        "filepath": os.path.join(PROJECT_ROOT, "te_process_normal_train.csv"),
        "description": (
            "500 independent time-series runs, 52 numeric variables per run — "
            "100 sampled for this analysis, treated as the normal/baseline "
            "dataset. What each variable represents and what process this "
            "data comes from aren't given; the analysis below infers what it "
            "can from the data itself."
        ),
    },
}


def get_dataset(name: str) -> dict:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset: {name!r}. Known: {list(DATASETS)}")
    return DATASETS[name]
