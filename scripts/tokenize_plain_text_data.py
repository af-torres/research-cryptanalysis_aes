import argparse
import os
from glob import glob
import pickle

from datasets import load_dataset, Dataset
from scripts.utils import get_max_len, byte_tokenize

parser = argparse.ArgumentParser(
    prog='ProgramName',
    description='What the program does',
    epilog='Text at the bottom of help'
)
parser.add_argument('-d', '--dataset', 
    type=str,
    required=True,
    choices=[
        "eng_sentences",
        "wikipedia_text"
    ]
)
parser.add_argument('-n', '--n_proc', type=int, default=4)
parser.add_argument('-rc', '--reduced_char_set', action='store_true')
args = parser.parse_args()

d_config = dict(
    eng_sentences = dict(
        plain_text_dir = "./data/plain_text/engSentences",
        data_name = "engSentences",
    ),
    wikipedia_text = dict(
        plain_text_dir = "./data/plain_text/wikipedia",
        data_name = "wikipedia"
    ),
)
dataset = d_config.get(args.dataset, None)
assert dataset


PLAIN_TEXT_DATA_DIR = dataset.get("plain_text_dir")
DATA_NAME = dataset.get("data_name")
assert PLAIN_TEXT_DATA_DIR and DATA_NAME

if args.reduced_char_set:
    PLAIN_TEXT_DATA_DIR += "-reduced_char_set"
    DATA_NAME += "-reduced_char_set"

OUT_DIR = f"./data/tokens/{DATA_NAME}/plain_text"
METADATA_FILE = f"./data/tokens/{DATA_NAME}/meta_data.pkl"

def get_unique_labels(ds, col="tokens"):
    def collect_uniques(batch):
        c = batch[col]
        uniques = set()
        for item in c:
            if isinstance(item, str):
                uniques.update(item)   # add characters
            else:
                uniques.update(item)   # add list elements
        return {"_uniques": list(uniques)}

    tmp = ds.map(
        collect_uniques,
        batched=True,
        batch_size=1000,
        remove_columns=ds.column_names,
    )
    unique_vals = set().union(tmp["_uniques"])

    return unique_vals

def encode_to_reduced_vocab(sentence, token_to_idx):
    sentence["tokens"] = [
        token_to_idx[t]
        for t in sentence["tokens"]
    ]
    return sentence

p_set_files = glob(os.path.join(PLAIN_TEXT_DATA_DIR, "**"))
p_set = load_dataset(
    "csv", data_files=p_set_files, split="train",
)
unique_letters = get_unique_labels(p_set, col="text")
print("loaded plain text dataset")

max_len = get_max_len(p_set)
p_tokens = p_set.map(
    lambda sentence: {"tokens": byte_tokenize(sentence["text"], max_len=max_len)},
    remove_columns=["text"],
    num_proc=args.n_proc
)
print("mapped plain text sentences to tokens")

if args.reduced_char_set:
    reduced_vocab = get_unique_labels(p_tokens)
    idx_to_token = dict(zip(range(len(reduced_vocab)), reduced_vocab))
    token_to_idx = dict(zip(reduced_vocab, range(len(reduced_vocab))))
    p_tokens = p_tokens.map(lambda sentence: encode_to_reduced_vocab(sentence, token_to_idx))
    
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(dict(
            unique_letters = unique_letters,
            reduced_vocab = reduced_vocab,
            idx_to_token = idx_to_token,
            token_to_idx = token_to_idx,
        ), f)
    print(f"wrote metadata file to {METADATA_FILE}")

p_tokens.save_to_disk(OUT_DIR)
print(f"wrote tokenized dataset: {OUT_DIR}")
