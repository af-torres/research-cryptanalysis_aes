import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score

from methods.lstm import build_model as lstm
from methods.gru import build_model as gru
from methods.rnn import build_model as rnn
from dataset import PAD_IDX, EOS_IDX

from datasets import Dataset

import pickle
import argparse
import random
import math
import uuid
import os
import time

parser = argparse.ArgumentParser(
    prog='ProgramName',
    description='What the program does',
    epilog='Text at the bottom of help'
)
parser.add_argument('-m', '--model', choices=[
        'lstm',
        'gru',
        'rnn',
    ],
    default='lstm'
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
        'wiki_128_rc',
        'wiki_192_rc',
        'wiki_256_rc',
        'wiki_rand_iv_128',
        'wiki_rand_iv_192',
        'wiki_rand_iv_256',
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
    wiki_128_rc = dict(
        plain_text = "./data/tokens/wikipedia-reduced_char_set/plain_text",
        meta_data = "./data/tokens/wikipedia-reduced_char_set/meta_data.pkl",
        encrypted_text = "./data/tokens/wikipedia-reduced_char_set/encrypted/128-bytes",
        reduced_char_set = True
    ),
    wiki_192_rc = dict(
        plain_text = "./data/tokens/wikipedia-reduced_char_set/plain_text",
        meta_data = "./data/tokens/wikipedia-reduced_char_set/meta_data.pkl",
        encrypted_text = "./data/tokens/wikipedia-reduced_char_set/encrypted/192-bytes",
        reduced_char_set = True
    ),
    wiki_256_rc = dict(
        plain_text = "./data/tokens/wikipedia-reduced_char_set/plain_text",
        meta_data = "./data/tokens/wikipedia-reduced_char_set/meta_data.pkl",
        encrypted_text = "./data/tokens/wikipedia-reduced_char_set/encrypted/256-bytes",
        reduced_char_set = True
    ),
    wiki_rand_iv_128 = dict(
        plain_text = "./data/tokens/wikipedia/plain_text",
        encrypted_text = "./data/tokens/wikipedia-rand-iv/encrypted/128-bytes",
    ),
    wiki_rand_iv_192 = dict(
        plain_text = "./data/tokens/wikipedia/plain_text",
        encrypted_text = "./data/tokens/wikipedia-rand-iv/encrypted/192-bytes",
    ),
    wiki_rand_iv_256 = dict(
        plain_text = "./data/tokens/wikipedia/plain_text",
        encrypted_text = "./data/tokens/wikipedia-rand-iv/encrypted/256-bytes",
    ),
)

MODELS = dict(
    lstm = lstm,
    gru = gru,
    rnn = rnn
)

RESULTS_DIR = "./results"
LOG_FILE = "./training_log.txt"
TOKENS_COLUMN = "tokens"
INDEX_COLUMN = "_idx"

INPUT_DIM = EOS_IDX + 1
OUTPUT_DIM = EOS_IDX + 1
PAD_IDX_OUT = PAD_IDX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_id = uuid.uuid4().hex
metrics = MetricCollection({
    'accuracy': Accuracy(task='multiclass', num_classes=EOS_IDX + 1, ignore_index=PAD_IDX),
    'precision': Precision(task='multiclass', num_classes=EOS_IDX + 1, ignore_index=PAD_IDX),
    'recall': Recall(task='multiclass', num_classes=EOS_IDX + 1, ignore_index=PAD_IDX),
    'f1': F1Score(task='multiclass', num_classes=EOS_IDX + 1, ignore_index=PAD_IDX)
})
metrics = metrics.to(device)

dataset = args.dataset
ds_config = DATASET_DIR.get(dataset)
assert ds_config

plain_text_ds_dir = ds_config.get("plain_text")
encrypted_text_ds_dir = ds_config.get("encrypted_text")
assert plain_text_ds_dir and encrypted_text_ds_dir

ds_max_size = args.ds_max_size
ds_c = Dataset.load_from_disk(encrypted_text_ds_dir).sort(INDEX_COLUMN).with_format("torch")
ds_p = Dataset.load_from_disk(plain_text_ds_dir).sort(INDEX_COLUMN).with_format("torch")
assert len(ds_c) == len(ds_p)

reduced_char_set = ds_config.get("reduced_char_set", False)
if reduced_char_set:
    """
    The reduced character dataset has a different vocab size (OUTPUT_DIM) and padding index
    for the cross-entropoy loss (PAD_IDX_OUT), and we need to adjust the program 
    to correctly match this dataset
    """
    meta_data_file = ds_config.get("meta_data")
    assert meta_data_file

    with open(meta_data_file, "rb") as f:
        meta_data = pickle.load(f)

    OUTPUT_DIM = len(meta_data["reduced_vocab"])
    PAD_IDX_OUT = meta_data["token_to_idx"][PAD_IDX]

n = len(ds_p)
ds_size = n if n < ds_max_size else ds_max_size

random.seed(42)
shuffle_idx = random.sample(np.arange(n).tolist(), ds_size)
ds = Dataset.from_dict({INDEX_COLUMN: shuffle_idx})

tr_ptr = 0
vl_ptr = math.floor(.8 * ds_size)
ts_ptr = math.floor(.9 * ds_size)

ds_tr = ds.select(range(tr_ptr, vl_ptr)).with_format("numpy")
ds_vl = ds.select(range(vl_ptr, ts_ptr)).with_format("numpy")
ds_ts = ds.select(range(ts_ptr, ds_size)).with_format("numpy")

batch_size = args.batch_size
epochs = args.epochs
dataloader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True) # type: ignore

model_config = dict(
    input_dim = INPUT_DIM,
    embed_dim = 500,
    hidden_dim = 250,
    output_dim = OUTPUT_DIM,
    pad_idx = PAD_IDX,
    pad_idx_out = PAD_IDX_OUT,
    n_layers = 6,
    dropout = 0.2,
)
model = MODELS.get(args.model)(**model_config, device = device) # type: ignore
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX_OUT)
optimizer = torch.optim.AdamW(model.parameters())

def eval(model, loss_fn, eval_idx):
    # this will force the model to only use it's predictions during sequence decoding
    # otherwise it will mix-in target values during the prediction process
    TEACHER_FORCE = False
    ds_e = Dataset.from_dict({INDEX_COLUMN: eval_idx.tolist()})
    dataloader = DataLoader(ds_e, batch_size=batch_size) # type: ignore
    
    loss_sum = 0.0
    total_tokens = 0
    for batch in dataloader:
        idx = batch[INDEX_COLUMN].numpy()
        X, Y = ds_c.select(idx)[:][TOKENS_COLUMN].to(device), \
            ds_p.select(idx)[:][TOKENS_COLUMN].to(device)
        Y_hat = model(X, Y, TEACHER_FORCE)

        Y_hat = Y_hat.reshape(-1, Y_hat.size(-1))
        Y = Y.reshape(-1)
        
        metrics.update(Y_hat.argmax(dim=-1), Y)
        n_tokens = (Y != PAD_IDX).sum().item()
        loss_sum += loss_fn(
            Y_hat,
            Y
        ).item() * n_tokens
        total_tokens += n_tokens

    metrics_results = metrics.compute()
    metrics.reset()
    
    return loss_sum / total_tokens, metrics_results

start_t = time.perf_counter()
print("start model training")

tr_loss = []
tr_metrics = []
vl_loss = []
vl_metrics = []
for e in range(1, epochs + 1):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        
        idx = batch[INDEX_COLUMN].numpy()
        X, Y = ds_c.select(idx)[:][TOKENS_COLUMN].to(device), \
            ds_p.select(idx)[:][TOKENS_COLUMN].to(device)
        
        Y_hat = model(X, Y)
        loss = loss_fn(
            Y_hat.reshape(-1, Y_hat.size(-1)),
            Y.reshape(-1)
        )
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        idx_tr = ds_tr[INDEX_COLUMN][:]
        idx_vl = ds_vl[INDEX_COLUMN][:]

        tr_loss_item, tr_metrics_item = eval(model, loss_fn, idx_tr)
        tr_loss.append(tr_loss_item)
        tr_metrics.append(tr_metrics_item)
        
        vl_loss_item, vl_metrics_item = eval(model, loss_fn, idx_vl)
        vl_loss.append(vl_loss_item)
        vl_metrics.append(vl_metrics_item)

    if e % 10 == 0: print(f"epoch {e}: training_loss={tr_loss_item}; validation_loss={vl_loss_item};")

end_t = time.perf_counter()
elapsed = end_t - start_t
print(f"trained model successfully in {elapsed:.3f} s")

# get clean metrics format
tr_metrics = pd.DataFrame(tr_metrics).map(lambda x: x.item() if hasattr(x, 'item') else x)
vl_metrics = pd.DataFrame(vl_metrics).map(lambda x: x.item() if hasattr(x, 'item') else x)

os.makedirs(RESULTS_DIR, exist_ok=True)
training_results_file = f"{RESULTS_DIR}/{run_id}.pkl"
with open(training_results_file, "wb") as f:
    pickle.dump(dict(
        tr_loss = tr_loss,
        tr_metrics = tr_metrics,
        vl_loss = vl_loss,
        vl_metrics = vl_metrics,
        model_config = model_config,
    ), f)
print(f"wrote results to {training_results_file}")

with open(LOG_FILE, "a") as f:
    f.write(
        f"{run_id}: "
        f"{"; ".join([f"{k}={v}" for k, v in args.__dict__.items()])};\n"
    )
print(f"append run info to {LOG_FILE}")
