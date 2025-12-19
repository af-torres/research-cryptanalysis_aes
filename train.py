import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import build_model
from dataset import PAD_IDX, EOS_IDX
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
parser.add_argument('-ms', '--ds_max_size',
    type=int, default=30000
)
args = parser.parse_args()

print(f"training model {args}")

DATASET_DIR = dict(
    wiki_128 = dict(
        plain_text = "./data/tokens/wikipedia/plain_text",
        encrypted_text = "./data/tokens/wikipedia/encrypted/128-bytes",
    ),
    wiki_192 = dict(
        plain_text = "./data/tokens/wikipedia/plain_text",
        encrypted_text = "./data/tokens/wikipedia/encrypted/192-bytes",
    ),
    wiki_256 = dict(
        plain_text = "./data/tokens/wikipedia/plain_text",
        encrypted_text = "./data/tokens/wikipedia/encrypted/256-bytes",
    ),
)
RESULTS_DIR = "./results"
LOG_FILE = "./training_log.txt"
C_COLUMN = "c_tokens"
P_COLUMN = "p_tokens"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_id = uuid.uuid4().hex

dataset = args.dataset
ds_config = DATASET_DIR.get(dataset)
assert ds_config

plain_text_ds_dir = ds_config.get("plain_text")
encrypted_text_ds_dir = ds_config.get("encrypted_text")
assert plain_text_ds_dir and encrypted_text_ds_dir

ds_max_size = args.ds_max_size
ds_c = Dataset.load_from_disk(encrypted_text_ds_dir).sort("_idx").with_format("torch", device=device)
ds_p = Dataset.load_from_disk(plain_text_ds_dir).sort("_idx").with_format("torch", device=device)
assert len(ds_c) == len(ds_p)

n = len(ds_p)
ds_size = n if n < ds_max_size else ds_max_size

random.seed(42)
shuffle_idx = random.sample(np.arange(n).tolist(), ds_size)
ds_c = ds_c.select(shuffle_idx)
ds_p = ds_p.select(shuffle_idx)
ds = Dataset.from_dict({
    C_COLUMN: ds_c["tokens"],
    P_COLUMN: ds_p["tokens"]
})

tr_ptr = 0
vl_ptr = math.floor(.8 * ds_size)
ts_ptr = math.floor(.9 * ds_size)

ds_tr = ds.select(range(tr_ptr, vl_ptr))
ds_vl = ds.select(range(vl_ptr, ts_ptr))
ds_ts = ds.select(range(ts_ptr, ds_size))

batch_size = args.batch_size
epochs = args.epochs
dataloader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True) # type: ignore

model_config = dict(
    input_dim = EOS_IDX + 1,
    embed_dim = 500,
    hidden_dim = 500,
    output_dim = EOS_IDX + 1,
    pad_idx = PAD_IDX,
    dropout = 0.5,
    device = device
)
model = build_model(**model_config)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.AdamW(model.parameters())

def eval(model, loss_fn, C_ds, P_ds):
    MAX_EVAL_SUBSET_SIZE = 10000
    
    n = len(C_ds)
    e_size = MAX_EVAL_SUBSET_SIZE if n > MAX_EVAL_SUBSET_SIZE else n
    
    C = C_ds[:e_size]
    P = P_ds[:e_size]
    Y_hat = model(C, P, 1)
    loss_item = loss_fn(
            Y_hat.reshape(-1, Y_hat.size(-1)),
            P.reshape(-1)
        ).item()
    
    return loss_item

tr_loss = []
vl_loss = []
for e in range(1, epochs + 1):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        X, Y = batch[C_COLUMN], batch[P_COLUMN]
        Y_hat = model(X, Y)
        loss = loss_fn(
            Y_hat.reshape(-1, Y_hat.size(-1)),
            Y.reshape(-1)
        )
        loss.backward()
        
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        C_tr, P_tr = ds_tr[C_COLUMN], ds_tr[P_COLUMN]
        C_vl, P_vl = ds_vl[C_COLUMN], ds_vl[P_COLUMN]
        tr_loss_item = eval(model, loss_fn, C_tr, P_tr)
        tr_loss.append(tr_loss_item)
        vl_loss_item = eval(model, loss_fn, C_vl, P_vl)
        vl_loss.append(vl_loss_item)

    if e % 10 == 0: print(f"epoch {e}: training_loss={tr_loss_item}; validation_loss={vl_loss_item};")

os.makedirs(RESULTS_DIR, exist_ok=True)
training_results_file = f"{RESULTS_DIR}/{run_id}.pkl"
with open(training_results_file, "wb") as f:
    pickle.dump(dict(
        tr_loss = tr_loss,
        vl_loss = vl_loss,
        model_config = model_config,
    ), f)

with open(LOG_FILE, "a") as f:
    f.write(f"{run_id}: dataset={dataset}\n")
