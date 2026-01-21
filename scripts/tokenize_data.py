import os
import argparse
from datasets import load_dataset
from glob import glob
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

print("wiriting tokenized dataset")
baseName = f"{OUT_DIR}/{k}"
os.makedirs(baseName, exist_ok=True)
c_tokens.save_to_disk(baseName)
