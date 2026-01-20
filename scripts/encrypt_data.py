import subprocess
import argparse
from datasets import load_dataset
from glob import glob
from scripts.utils import shard_into_files
import os

ENCRYPT_SCRIPT = ["./scripts/encrypt.sh"]
MINI_AES_ENCRYPT_SCRIPT = ["python", "./scripts/mini_aes.py", "encrypt"]
USE_MINI_AES = False

KEY_FILES = {
    "128_bytes": "./data/keys/128-bytes.hex",
    "192_bytes": "./data/keys/192-bytes.hex",
    "256_bytes": "./data/keys/256-bytes.hex",
    "mini_aes": "./data/keys/mini_aes.bin",
}

def get_iv(key, random_iv=False):
    iv = None
    if not random_iv: iv = f"{key}.iv"
    return iv

def encrypt(s, key, iv):
    p_args = (MINI_AES_ENCRYPT_SCRIPT if USE_MINI_AES else ENCRYPT_SCRIPT).copy()
    p_args.append(f"--key={key}")
    if iv is not None:
        p_args.append(f"--iv={iv}")

    result = subprocess.run(
        p_args,
        input=s,
        text=True,
        capture_output=True,
        check=False
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Encryption failed (rc={result.returncode})\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout.strip()

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
            "wikipedia_text",
            "mini_aes"
        ]
    )
    parser.add_argument('--random_iv', '-r',
        action='store_true',
    )
    parser.add_argument('--key', '-k',
        type=str,
        required=True,
        choices=list(KEY_FILES.keys())
    )
    parser.add_argument('--mini_aes', action='store_true')
    parser.add_argument('-rc', '--reduced_char_set', action='store_true')
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
        ),
        mini_aes = dict(
            data_dir = "./data/plain_text/mini_aes",
            data_name = "mini_aes"
        )
    )
    dataset = d_config.get(args.dataset, None)
    assert dataset

    DATA_DIR = dataset.get("data_dir", None)        
    DATA_NAME = dataset.get("data_name", None)
    assert  DATA_DIR and DATA_NAME
    
    if args.reduced_char_set:
        DATA_DIR += "-reduced_char_set"
        DATA_NAME += "-reduced_char_set"

    random_iv: bool = args.random_iv
    USE_MINI_AES = args.mini_aes

    OUT_DIR = f"./data/encrypted/{DATA_NAME}{"-rand-iv" if random_iv else ""}"

    data_files = glob(os.path.join(DATA_DIR, "**"))
    ds = load_dataset(
        "csv", 
        data_files=data_files, split="train"
    )

    key = KEY_FILES[args.key]
    iv = get_iv(key, random_iv)
    keyName = os.path.basename(key).removesuffix(".hex")
    print(f"encrypting sentences with {keyName} key")

    enc = ds.map(
        lambda sentence: {
            "_idx": sentence["_idx"],
            "text": encrypt(sentence["text"], key, iv)
        },
        num_proc=args.n_proc, # type: ignore
    )

    baseName = f"{OUT_DIR}/{keyName}" # type: ignore
    os.makedirs(baseName, exist_ok=True)

    num_shards = len(data_files)
    shard_into_files(enc, baseName, num_shards)
