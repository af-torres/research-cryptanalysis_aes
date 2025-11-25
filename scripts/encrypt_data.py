import subprocess
import numpy as np
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser(
    prog='ProgramName',
    description='What the program does',
    epilog='Text at the bottom of help'
)
parser.add_argument('--dataset', '-d',
    type=str,
    required=True,
    choices=[
        "eng_sentences",
        "wikipedia_text"
    ]
)
parser.add_argument('--random_iv', '-r',
    action='store_true',
)
args = parser.parse_args()

d_config = dict(
    eng_sentences = dict(
        data_file = "./data/englishSentences.csv",
        key_files = [
            "./data/keys/128-bytes.hex",
            "./data/keys/192-bytes.hex",
            "./data/keys/256-bytes.hex",
        ],
        data_name = None,
    ),
    wikipedia_text = dict(
        data_file = "./data/wikipediaSentences.csv",
        key_files = [
            "./data/keys/128-bytes.hex",
            "./data/keys/192-bytes.hex",
            "./data/keys/256-bytes.hex",
        ],
        data_name = "wiki"
    )
)
dataset = d_config.get(args.dataset, None)
assert dataset

ENCRYPT_SCRIPT = "./scripts/encrypt.sh"

DATA_FILE = dataset.get("data_file", None)
KEY_FILES = dataset.get("key_files", None)
DATA_NAME = dataset.get("data_name", None)
assert DATA_FILE and KEY_FILES 

OUT_DIR = "./data/encrypted"

random_iv: bool = args.random_iv

def encrypt(s, key, iv):
    p_args = [ENCRYPT_SCRIPT, f"--key={key}"]
    if iv is not None: p_args.append(f"--iv={iv}")
    
    p = subprocess.Popen(
        p_args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    assert p.stdin and p.stdout

    p.stdin.write(s)
    p.stdin.close()
    output = p.stdout.read().strip()

    return output

sentences = pd.read_csv(DATA_FILE).iloc[:, 0].to_numpy(dtype=str) # type: ignore
for key in KEY_FILES:
    iv = None
    if not random_iv: iv = f"{key}.iv"
    keyName = os.path.basename(key).removesuffix(".hex")
    print(f"encrypting sentences with {keyName} key")
    
    enc = []
    for s in sentences:
        e = encrypt(str(s), key, iv)
        enc.append(e)
    
    fname = f"{OUT_DIR}/{keyName}{"-" + DATA_NAME if DATA_NAME else ""}{"-rand-iv" if random_iv else ""}.csv" # type: ignore
    np.savetxt(fname, np.array(enc, dtype=str), delimiter=",", fmt="%s")
    print(f"wrote file {fname}")
