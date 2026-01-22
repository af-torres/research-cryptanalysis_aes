import os
import argparse
from datasets import load_dataset
from glob import glob
import pickle
from scripts.utils import get_max_len, byte_tokenize, get_unique_tokens, encode_to_reduced_vocab

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
parser.add_argument('-k', '--key', 
    type=str,
    choices=[
        "128-bytes",
        "192-bytes",
        "256-bytes",
        "mini_aes.bin"
    ], required=True
)
parser.add_argument('-r', '--random_iv',
    action='store_true',
)
parser.add_argument('--mini_aes', action='store_true')
parser.add_argument('-rc', '--reduced_char_set', action='store_true')
parser.add_argument('-n', '--n_proc', type=int, default=4)
args = parser.parse_args()

k = args.key
d_config = dict(
    eng_sentences = dict(
        encrypted_text_dir = "./data/encrypted/engSentences",
        data_name = "engSentences",
    ),
    wikipedia_text = dict(
        encrypted_text_dir = "./data/encrypted/wikipedia",
        data_name = "wikipedia"
    ),
    mini_aes = dict(
        encrypted_text_dir = "./data/encrypted/mini_aes",
        data_name = "mini_aes"
    )
)
dataset = d_config.get(args.dataset, None)
assert dataset

ENCRYPTED_TEXT_DATA_DIR = dataset.get("encrypted_text_dir")
DATA_NAME = dataset.get("data_name")
assert ENCRYPTED_TEXT_DATA_DIR and DATA_NAME

if args.reduced_char_set:
    ENCRYPTED_TEXT_DATA_DIR += "-reduced_char_set"
    DATA_NAME += "-reduced_char_set"

USE_MINI_AES_ENC = args.mini_aes

random_iv: bool = args.random_iv
ENCRYPTED_TEXT_DATA_DIR = f"{ENCRYPTED_TEXT_DATA_DIR}{"-rand-iv" if random_iv else ""}"
DATA_NAME = f"{DATA_NAME}{"-rand-iv" if random_iv else ""}"
OUT_DIR = f"./data/tokens/{DATA_NAME}/encrypted"

METADATA_FILE = f"./data/tokens/{DATA_NAME}/meta_data_encrypted.pkl"

print("mapping encrypted sentences to tokens")
c_set_dir = f"{ENCRYPTED_TEXT_DATA_DIR}/{k}"
c_set_files = glob(os.path.join(c_set_dir, "**"))
c_set = load_dataset("csv", data_files=c_set_files, split="train")
max_len = get_max_len(c_set)
c_tokens = c_set.map(
    lambda sentence: {"tokens": byte_tokenize(sentence["text"], max_len=max_len, maes_ec=USE_MINI_AES_ENC)},
    remove_columns=["text"],
    num_proc=args.n_proc
)

if USE_MINI_AES_ENC:
    reduced_vocab = get_unique_tokens(c_tokens)
    idx_to_token = dict(zip(range(len(reduced_vocab)), reduced_vocab))
    token_to_idx = dict(zip(reduced_vocab, range(len(reduced_vocab))))
    c_tokens = c_tokens.map(lambda sentence: encode_to_reduced_vocab(sentence, token_to_idx))

    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(dict(
            unique_letters = reduced_vocab,
            reduced_vocab = reduced_vocab,
            idx_to_token = idx_to_token,
            token_to_idx = token_to_idx,
        ), f)
    print(f"wrote metadata file to {METADATA_FILE}")


print("wiriting tokenized dataset")
baseName = f"{OUT_DIR}/{k}"
os.makedirs(baseName, exist_ok=True)
c_tokens.save_to_disk(baseName)
