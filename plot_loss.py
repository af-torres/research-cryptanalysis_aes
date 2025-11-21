import matplotlib.pyplot as plt
import numpy as np

import pickle
import argparse
import os

from utils import parse_run_log

parser = argparse.ArgumentParser(
    prog='ProgramName',
    description='What the program does',
    epilog='Text at the bottom of help'
)
parser.add_argument('-id', '--run_id',
    type=str,
    required=True,
)
args = parser.parse_args()

RESULTS_DIR = "./results"
FIGURES_DIR = "./fig"
LOG_FILE = "./training_log.txt"

run_id = args.run_id

def get_dataset(run_id):
    runs = parse_run_log(LOG_FILE)
    run_config = runs.get(run_id, None)
    assert run_config is not None
    
    dataset = run_config.get("dataset", "")
    assert dataset != ""
    
    return dataset

def plot_loss(tr, vl, run_id):
    tr_arr = np.array(tr)
    vl_arr = np.array(vl)
    assert len(tr_arr) == len(vl_arr)

    x = np.arange(len(tr_arr)) + 1

    plt.plot(x, tr_arr, label="Training")
    plt.plot(x, vl_arr, label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training {get_dataset(run_id)} Epoch vs. Loss")
    plt.legend()
    plt.grid(True)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_name = f"{FIGURES_DIR}/{run_id}.png"
    plt.savefig(fig_name, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"wrote image file: {fig_name}")

training_results_file = f"{RESULTS_DIR}/{run_id}.pkl"
with open(training_results_file, "rb") as f:
    run_data = pickle.load(f)

tr, vl = run_data.get("tr_loss"), run_data.get("vl_loss")
plot_loss(tr, vl, run_id)
