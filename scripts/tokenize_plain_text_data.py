import argparse
import os
from glob import glob
import pickle

from datasets import load_dataset
from scripts.utils import get_max_len, byte_tokenize, get_unique_tokens

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
        "wikipedia_text",
        "mini_aes"
    ]
)
parser.add_argument('--mini_aes', action='store_true')
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
    mini_aes = dict(
        plain_text_dir = "./data/plain_text/mini_aes",
        data_name = "mini_aes"
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

USE_MINI_AES_ENC = args.mini_aes

OUT_DIR = f"./data/tokens/{DATA_NAME}/plain_text"
METADATA_FILE = f"./data/tokens/{DATA_NAME}/meta_data.pkl"



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
unique_letters = get_unique_tokens(p_set, col="text")
print("loaded plain text dataset")

max_len = get_max_len(p_set)
p_tokens = p_set.map(
    lambda sentence: {"tokens": byte_tokenize(sentence["text"], max_len=max_len, maes_ec=USE_MINI_AES_ENC)},
    remove_columns=["text"],
    num_proc=args.n_proc
)
print("mapped plain text sentences to tokens")

if args.reduced_char_set or USE_MINI_AES_ENC:
    reduced_vocab = get_unique_tokens(p_tokens)
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
