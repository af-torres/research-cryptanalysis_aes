import numpy as np
import pandas as pd
from datasets import load_dataset

DATA_DIR = "./data/plain_text/wikipedia"

def shard_into_files(ds, max_shard_len=3000):
    n = len(ds)
    num_shards = n // max_shard_len

    breakpoint()
    for i in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=i)
        shard.to_csv(f"{DATA_DIR}/shard_{i}.csv")

def split_to_chunks(batch, max_size=500):
    new_text = []
    for text in batch["text"]:
        chunks = [text[i:i+max_size] for i in range(0, len(text), max_size)]
        new_text.extend(chunks)
    return {"text": new_text}

ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
ds = ds.filter(lambda sentence: sentence["text"] is not None and sentence["text"] != "") # remove blank lines
ds = ds.map(lambda sentence: {
    "text": sentence["text"].strip().replace("<unk>", " ")
})
ds = ds.map(split_to_chunks, batched=True)
shard_into_files(ds)
