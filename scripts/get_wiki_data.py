import numpy as np
import pandas as pd
from datasets import load_dataset

DATA_FILE = "./data/wikipediaSentences.csv"

ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
ds = ds.filter(lambda sentence: sentence["text"] != "") # remove blank lines
ds = ds.map(lambda sentence: {
    "text": sentence["text"].strip().replace("<unk>", " ")
})
ds.to_csv(DATA_FILE) # type: ignore
    