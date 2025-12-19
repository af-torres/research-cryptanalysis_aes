import os
import argparse
from datasets import load_dataset, Dataset
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
        "wikipedia_text"
    ]
)
parser.add_argument('-n', '--n_proc', type=int, default=4)
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

OUT_DIR = f"./data/tokens/{DATA_NAME}/plain_text"

p_set_files = glob(os.path.join(PLAIN_TEXT_DATA_DIR, "**"))
p_set = load_dataset(
    "csv", data_files=p_set_files, split="train",
)
print("loaded plain text dataset")

max_len = get_max_len(p_set)
p_tokens = p_set.map(
    lambda sentence: {"tokens": byte_tokenize(sentence["text"], max_len=max_len)},
    remove_columns=["text"],
    num_proc=args.n_proc
)
print("mapped plain text sentences to tokens")

os.makedirs(OUT_DIR, exist_ok=True)
p_tokens.save_to_disk(OUT_DIR)
print(f"wrote tokenized dataset: {OUT_DIR}")

