import subprocess
import argparse
from datasets import load_dataset
from glob import glob
import os

ENCRYPT_SCRIPT = "./scripts/encrypt.sh"
KEY_FILES = [
    "./data/keys/128-bytes.hex",
    "./data/keys/192-bytes.hex",
    "./data/keys/256-bytes.hex",
]

def get_iv(key, random_iv=False):
    iv = None
    if not random_iv: iv = f"{key}.iv"
    return iv

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

    out, err = p.communicate(s)

    if p.returncode != 0:
        raise RuntimeError(f"Encryption failed: {err}")

    return out.strip()

if __name__ == "__main__":
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
            data_name = "engSentences",
        ),
        wikipedia_text = dict(
            data_dir = "./data/plain_text/wikipedia",
            data_name = "wikipedia"
        )
    )
    dataset = d_config.get(args.dataset, None)
    assert dataset

    DATA_DIR = dataset.get("data_dir", None)
    DATA_NAME = dataset.get("data_name", None)
    assert  DATA_DIR and DATA_NAME

    random_iv: bool = args.random_iv

    OUT_DIR = f"./data/encrypted/{DATA_NAME}{"-rand-iv" if random_iv else ""}"

    data_files = sorted(glob(os.path.join(DATA_DIR, "**"))) # type: ignore
    ds = load_dataset(
        "csv", 
        data_files=data_files, split="train"
    )
    for key in KEY_FILES:
        iv = get_iv(key, random_iv)
        keyName = os.path.basename(key).removesuffix(".hex")
        print(f"encrypting sentences with {keyName} key")

        enc = ds.map(
            lambda sentence: {
                "text": encrypt(sentence["text"], key, iv)
            },
            num_proc=args.n_proc,
        )

        baseName = f"{OUT_DIR}/{keyName}" # type: ignore
        os.makedirs(baseName, exist_ok=True)

        num_shards = len(data_files)
        n = len(enc)
        rows_per_shard = n // num_shards
        for i in range(num_shards):
            start = i * rows_per_shard
            end = (i+1) * rows_per_shard if i < num_shards - 1 else n
            shard = enc.select(range(start, end)) # type: ignore

            fname = f"{baseName}/shard_{i}.csv"
            shard.to_csv(fname)
            print(f"wrote file {fname}")
