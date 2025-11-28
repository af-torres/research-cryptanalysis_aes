import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import build_model
from dataset import PAD_IDX, SOS_IDX, EOS_IDX
from datasets import Dataset

import pickle
import argparse
import random
import math
import uuid
import os

parser = argparse.ArgumentParser(
    prog='ProgramName',
    description='What the program does',
    epilog='Text at the bottom of help'
)
parser.add_argument('-d', '--dataset', choices = [
        'short_128',
        'short_192',
        'short_256',
        'short_rand_iv_128',
        'short_rand_iv_192',
        'short_rand_iv_256',
        'wiki_128',
        'wiki_192',
        'wiki_256',
    ],
    required=True,
)
parser.add_argument('-e', '--epochs',
    type=int,
    default=500
)
parser.add_argument('-bs', '--batch_size',
    type=int,
    default=500
)
args = parser.parse_args()

DATASET_DIR = dict(
    short_128 = "./data/tokens/engSentences/128-bytes",
    short_192 = "./data/tokens/engSentences/192-bytes",
    short_256 = "./data/tokens/engSentences/256-bytes",
    short_rand_iv_128 = "./data/tokens/engSentences/128-bytes-rand-iv",
    short_rand_iv_192 = "./data/tokens/engSentences/192-bytes-rand-iv",
    short_rand_iv_256 = "./data/tokens/engSentences/256-bytes-rand-iv",
    wiki_128 = "./data/tokens/wikipedia/128-bytes",
    wiki_192 = "./data/tokens/wikipedia/192-bytes",
    wiki_256 = "./data/tokens/wikipedia/256-bytes",
)
RESULTS_DIR = "./results"
LOG_FILE = "./training_log.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_id = uuid.uuid4().hex

dataset = args.dataset
ds_dir = DATASET_DIR.get(dataset, "")
assert ds_dir != ""

#!important: there might be some issue with the final shape of these objects
#! Or if every item here has the same shape then hf with format torch has a compatibility bug
ds = Dataset.load_from_disk(ds_dir).with_format("torch", device=device)
n = len(ds)

random.seed(42)
shuffle_idx = random.sample(np.arange(n).tolist(), n)
ds = ds.select(shuffle_idx)

tr_ptr = 0
vl_ptr = math.floor(.8 * n)
ts_ptr = math.floor(.9 * n)

ds_tr = ds.select(range(tr_ptr, vl_ptr))
ds_vl = ds.select(range(vl_ptr, ts_ptr))
ds_ts = ds.select(range(ts_ptr, n))

batch_size = args.batch_size
epochs = args.epochs
dataloader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True) # type: ignore

model_config = dict(
    input_dim = EOS_IDX + 1,
    embed_dim = 500,
    hidden_dim = 500,
    output_dim = EOS_IDX + 1,
    pad_idx = PAD_IDX,
    device = device
)
model = build_model(**model_config)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.AdamW(model.parameters())

tr_loss = []
vl_loss = []
for e in range(1, epochs + 1):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        X, Y = batch["c_tokens"], batch["p_tokens"]
        breakpoint()
        Y_hat = model(X, Y)        
        loss = loss_fn(
            Y_hat.reshape(-1, Y_hat.size(-1)),
            Y.reshape(-1)
        )
        loss.backward()
        
        optimizer.step()
    
    # model.eval()
    # with torch.no_grad():
    #     Y_hat = model(C_tr, P_tr, 1)
    #     tr_loss_item = loss_fn(
    #         Y_hat.reshape(-1, Y_hat.size(-1)),
    #         P_tr.reshape(-1)
    #     ).item()
    #     tr_loss.append(tr_loss_item)

    #     Y_hat = model(C_vl, P_vl, 1)
    #     vl_loss_item = loss_fn(
    #         Y_hat.reshape(-1, Y_hat.size(-1)), 
    #         P_vl.reshape(-1)
    #     ).item()
    #     vl_loss.append(vl_loss_item)

    # if e % 50 == 0: print(f"epoch {e}: training_loss={tr_loss_item}; validation_loss={vl_loss_item};")

# os.makedirs(RESULTS_DIR, exist_ok=True)
# training_results_file = f"{RESULTS_DIR}/{run_id}.pkl"
# with open(training_results_file, "wb") as f:
#     pickle.dump(dict(
#         tr_loss = tr_loss,
#         vl_loss = vl_loss,
#         model_config = model_config,
#     ), f)

# with open(LOG_FILE, "a") as f:
#     f.write(f"{run_id}: dataset={dataset}\n")
