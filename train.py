import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import build_model
from dataset import PAD_IDX, SOS_IDX, EOS_IDX

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
    ],
    required=True,
)
parser.add_argument('-e', '--epochs',
    type=int,
    default=500
)
args = parser.parse_args()

C_DATASET = dict(
    short_128 = "./data/128-byte-tokens.pkl",
    short_192 = "./data/192-byte-tokens.pkl",
    short_256 = "./data/256-byte-tokens.pkl",
    short_rand_iv_128 = "./data/128-byte-rand-iv-tokens.pkl",
    short_rand_iv_192 = "./data/192-byte-rand-iv-tokens.pkl",
    short_rand_iv_256 = "./data/256-byte-rand-iv-tokens.pkl",
)
P_DATASET = dict(
    short_128 = "./data/englishSentence-tokens.pkl",
    short_192 = "./data/englishSentence-tokens.pkl",
    short_256 = "./data/englishSentence-tokens.pkl",
    short_rand_iv_128 = "./data/englishSentence-tokens.pkl",
    short_rand_iv_192 = "./data/englishSentence-tokens.pkl",
    short_rand_iv_256 = "./data/englishSentence-tokens.pkl",
)
RESULTS_DIR = "./results"
LOG_FILE = "./training_log.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_id = uuid.uuid4().hex

dataset = args.dataset
c_file = C_DATASET.get(dataset, "")
p_file = P_DATASET.get(dataset, "")
assert c_file != "" and p_file != ""

with open(c_file, "rb") as cf:
    C: np.ndarray | torch.Tensor = pickle.load(cf)
with open(p_file, "rb") as pf:
    P: np.ndarray | torch.Tensor = pickle.load(pf)
assert len(C) == len(P)

random.seed(42)
shuffle_idx = random.sample(np.arange(len(P)).tolist(), len(P))
P = torch.tensor(P[shuffle_idx, :], dtype=torch.long, device=device)
C = torch.tensor(C[shuffle_idx, :], dtype=torch.long, device=device)

tr_ptr = 0
vl_ptr = math.floor(.8 * len(P))
ts_ptr = math.floor(.9 * len(P))

P_tr = P[:vl_ptr, :]
C_tr = C[:vl_ptr, :]

P_vl = P[vl_ptr: ts_ptr, :]
C_vl = C[vl_ptr: ts_ptr, :]

P_ts = P[ts_ptr:, :]
C_ts = C[ts_ptr:, :]

batch_size = 250
epochs = args.epochs
train_dataset = TensorDataset(C_tr, P_tr)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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
    for X, Y in dataloader:
        optimizer.zero_grad()

        Y_hat = model(X, Y)        
        loss = loss_fn(
            Y_hat.reshape(-1, Y_hat.size(-1)),
            Y.reshape(-1)
        )
        loss.backward()
        
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        Y_hat = model(C_tr, P_tr, 1)
        tr_loss_item = loss_fn(
            Y_hat.reshape(-1, Y_hat.size(-1)),
            P_tr.reshape(-1)
        ).item()
        tr_loss.append(tr_loss_item)

        Y_hat = model(C_vl, P_vl, 1)
        vl_loss_item = loss_fn(
            Y_hat.reshape(-1, Y_hat.size(-1)), 
            P_vl.reshape(-1)
        ).item()
        vl_loss.append(vl_loss_item)

    if e % 50 == 0: print(f"epoch {e}: training_loss={tr_loss_item}; validation_loss={vl_loss_item};")

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
