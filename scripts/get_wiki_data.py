from datasets import load_dataset
from scripts.utils import shard_into_files
import re
import argparse
import os
import numpy as np
import random

parser = argparse.ArgumentParser(
    prog='ProgramName',
    description='What the program does',
    epilog='Text at the bottom of help'
)
parser.add_argument('-rc', '--reduced_char_set', action='store_true')
parser.add_argument('-ms', "--max_size", default=-1, type=int)
args = parser.parse_args()

rc = args.reduced_char_set
ms = args.max_size

DATA_DIR = "./data/plain_text/wikipedia"
MAX_SHARD_SIZE = 15000
MIN_SENTENCE_SIZE = 30
os.makedirs(DATA_DIR, exist_ok=True)

def split_to_chunks(batch, max_size=500):
    new_text = []
    for text in batch["text"]:
        chunks = [text[i:i+max_size] for i in range(0, len(text), max_size)]
        new_text.extend(chunks)
    return {"text": new_text}

ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
ds = ds.filter(lambda sentence: sentence["text"] is not None and sentence["text"].strip() != "") # remove blank lines
ds = ds.map(lambda sentence: {
    "text": sentence["text"].strip().replace("<unk>", " ")
})

if rc:
    DATA_DIR += "-reduced_char_set"
    ds = ds.map(lambda sentence: {
        "text": re.sub(r"[^a-z ]", "", sentence["text"].lower())
    })

ds = ds.map(split_to_chunks, batched=True)
ds = ds.filter(lambda sentence: len(sentence["text"]) > MIN_SENTENCE_SIZE)

if ms > 0:
    random.seed(42)
    select_idx = random.sample(np.arange(len(ds)).tolist(), ms)
    ds = ds.select(select_idx)

ds = ds.map(lambda sentence, idx: {
    "_idx": idx,
    "text": sentence["text"],
}, with_indices=True)
shard_into_files(ds, DATA_DIR, len(ds) // MAX_SHARD_SIZE) # type: ignore

