import os
import argparse
import base64
from datasets import load_dataset, Dataset
from glob import glob

PAD_IDX = 256
SOS_IDX = 257
EOS_IDX = 258

KEYS = [
    "128-bytes",
    "192-bytes",
    "256-bytes"
]

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
parser.add_argument('-r', '--random_iv',
    action='store_true',
)
args = parser.parse_args()

d_config = dict(
    eng_sentences = dict(
        plain_text_dir = "./data/plain_text/engSentences",
        encrypted_text_dir = "./data/encrypted/engSentences",
        data_name = "engSentences",
    ),
    wikipedia_text = dict(
        plain_text_dir = "./data/plain_text/wikipedia",
        encrypted_text_dir = "./data/encrypted/wikipedia",
        data_name = "wikipedia"
    ),
)
dataset = d_config.get(args.dataset, None)
assert dataset

PLAIN_TEXT_DATA_DIR = dataset.get("plain_text_dir")
ENCRYPTED_TEXT_DATA_DIR = dataset.get("encrypted_text_dir")
DATA_NAME = dataset.get("data_name")
assert PLAIN_TEXT_DATA_DIR and ENCRYPTED_TEXT_DATA_DIR and DATA_NAME

random_iv: bool = args.random_iv
ENCRYPTED_TEXT_DATA_DIR = f"{ENCRYPTED_TEXT_DATA_DIR}{"-rand-iv" if random_iv else ""}"
DATA_NAME = f"{DATA_NAME}{"-rand-iv" if random_iv else ""}"
OUT_DIR = f"./data/tokens/{DATA_NAME}"

def byte_tokenize(sentence, add_sos=True, add_eos=True, max_len=None, b64_enc=False):
    if b64_enc:
        decoded_sentence = base64.b64decode(sentence)
    else:
        decoded_sentence = sentence.encode("utf-8")
    byte_ids = list(decoded_sentence) 
    
    if add_sos:
        byte_ids = [SOS_IDX] + byte_ids
    if add_eos:
        byte_ids = byte_ids + [EOS_IDX]
    
    if max_len is not None:
        padding = [PAD_IDX] * max(0, max_len - len(byte_ids))
        byte_ids = byte_ids + padding
    
    return byte_ids

def get_max_len(ds):
    max_bytes = 0
    for i in range(len(ds)):
        text = ds[i]["text"]
        if text is None: continue

        b = len(text.encode("utf-8"))
        if b > max_bytes:
            max_bytes = b
    return max_bytes + 2  # SOS + EOS

p_set_files = sorted(glob(os.path.join(PLAIN_TEXT_DATA_DIR, "**")))
p_set = load_dataset(
    "csv", data_files=p_set_files, split="train",
    download_mode="force_redownload", verification_mode="no_checks"
)
max_len = get_max_len(p_set)
p_tokens = p_set.map(
    lambda sentence: {"tokens": byte_tokenize(sentence["text"], max_len=max_len)},
    remove_columns="text"
)

for k in KEYS:
    c_set_dir = f"{ENCRYPTED_TEXT_DATA_DIR}/{k}"
    c_set_files = sorted(glob(os.path.join(c_set_dir, "**")))
    c_set = load_dataset(
        "csv", data_files=c_set_files, split="train",
        download_mode="force_redownload", verification_mode="no_checks"
    )
    max_len = get_max_len(c_set)
    c_tokens = c_set.map(
        lambda sentence: {"tokens": byte_tokenize(sentence["text"], max_len=max_len)},
        remove_columns="text"
    )
    merged = Dataset.from_dict({
        "p_tokens": p_tokens["tokens"], # type: ignore
        "c_tokens": c_tokens["tokens"]  # type: ignore
    })
    
    baseName = f"{OUT_DIR}/{k}"
    os.makedirs(baseName, exist_ok=True)
    merged.save_to_disk(baseName)
