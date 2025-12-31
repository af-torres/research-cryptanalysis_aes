import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pickle
import argparse
import os

from utils import get_dataset

parser = argparse.ArgumentParser(
    prog='ProgramName',
    description='What the program does',
    epilog='Text at the bottom of help'
)
parser.add_argument('-id', '--run_id',
    type=str,
    required=True,
)
parser.add_argument('-m', '--metric',
    choices=[
        "accuracy",
        "precision",
        "recall",
        "f1",
    ],
    default="accuracy",
)
args = parser.parse_args()

RESULTS_DIR = "./results"
FIGURES_DIR = "./fig"
LOG_FILE = "./training_log.txt"

run_id = args.run_id

def plot_metric(tr, vl, run_id, metric_name):
    tr_arr = np.array(tr)
    vl_arr = np.array(vl)
    assert len(tr_arr) == len(vl_arr)

    x = np.arange(len(tr_arr)) + 1

    plt.plot(x, tr_arr, label="Training")
    plt.plot(x, vl_arr, label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"Training {get_dataset(LOG_FILE, run_id)} Epoch vs. {metric_name}")
    plt.legend()
    plt.grid(True)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_name = f"{FIGURES_DIR}/{run_id}-{metric_name}.png"
    plt.savefig(fig_name, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"wrote image file: {fig_name}")


training_results_file = f"{RESULTS_DIR}/{run_id}.pkl"
with open(training_results_file, "rb") as f:
    run_data = pickle.load(f)

tr_metrics = pd.DataFrame(run_data.get("tr_metrics")).map(lambda x: x.item() if hasattr(x, 'item') else x)
vl_metrics = pd.DataFrame(run_data.get("vl_metrics")).map(lambda x: x.item() if hasattr(x, 'item') else x)
metric = args.metric

plot_metric(tr_metrics[metric], vl_metrics[metric], run_id, metric)

