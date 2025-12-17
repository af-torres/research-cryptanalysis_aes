from datasets import load_dataset
from scripts.utils import shard_into_files
import os

DATA_DIR = "./data/plain_text/wikipedia"
MAX_SHARD_SIZE = 15000
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
ds = ds.map(split_to_chunks, batched=True)
ds = ds.map(lambda text, idx: {
    "_idx": idx,
    "text": text,
}, with_indices=True)

shard_into_files(ds, DATA_DIR, len(ds) // MAX_SHARD_SIZE) # type: ignore
