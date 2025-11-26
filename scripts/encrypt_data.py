import subprocess
import argparse
from datasets import load_dataset
from glob import glob
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
parser.add_argument("--n_proc", "-n", type=int, default=4)
args = parser.parse_args()

d_config = dict(
    eng_sentences = dict(
        data_dir = "./data/plain_text/engSentences",
        key_files = [
            "./data/keys/128-bytes.hex",
            "./data/keys/192-bytes.hex",
            "./data/keys/256-bytes.hex",
        ],
        data_name = "engSentences",
    ),
    wikipedia_text = dict(
        data_dir = "./data/plain_text/wikipedia",
        key_files = [
            "./data/keys/128-bytes.hex",
            "./data/keys/192-bytes.hex",
            "./data/keys/256-bytes.hex",
        ],
        data_name = "wikipedia"
    )
)
dataset = d_config.get(args.dataset, None)
assert dataset

ENCRYPT_SCRIPT = "./scripts/encrypt.sh"

DATA_DIR = dataset.get("data_dir", None)
KEY_FILES = dataset.get("key_files", None)
DATA_NAME = dataset.get("data_name", None)
assert  DATA_DIR and KEY_FILES and DATA_NAME

OUT_DIR = f"./data/encrypted/{DATA_NAME}"
os.makedirs(OUT_DIR, exist_ok=True)

data_files = glob(os.path.join(DATA_DIR, "**")) # type: ignore
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

ds = load_dataset("csv", data_files=data_files, split="train")
for key in KEY_FILES:
    iv = None
    if not random_iv: iv = f"{key}.iv"
    keyName = os.path.basename(key).removesuffix(".hex")
    print(f"encrypting sentences with {keyName} key")
    
    enc = ds.map(lambda sentence: {
        "text": encrypt(str(sentence["text"]), key, iv)
    }, num_proc=args.n_proc) # type: ignore
    
    baseName = f"{OUT_DIR}/{keyName}-{DATA_NAME}{"-rand-iv" if random_iv else ""}" # type: ignore
    num_shards = len(data_files)
    for i in range(0, num_shards):
        fname = f"{baseName}-shard_{i}.csv"
        shard = enc.shard(num_shards=num_shards, index=i) # type: ignore
        shard.to_csv(fname)
        print(f"wrote file {fname}")
