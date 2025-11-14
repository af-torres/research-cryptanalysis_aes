import subprocess
import numpy as np
import os

ENCRYPT_SCRIPT = "./scripts/encrypt.sh"

DATA_FILE = "./data/englishSentences.csv"
KEY_FILES = [
    "./data/keys/128-bytes.hex",
    "./data/keys/192-bytes.hex",
    "./data/keys/256-bytes.hex",
]

OUT_DIR = "./data/encrypted"

def encrypt(s, key, iv):
    p = subprocess.Popen(
        [ENCRYPT_SCRIPT, f"--key={key}", f"--iv={iv}"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    p.stdin.write(s)
    p.stdin.close()

    output = p.stdout.read().strip()
    return output

sentences = np.loadtxt(DATA_FILE, delimiter=",", dtype=str)
for key in KEY_FILES:
    iv = f"{key}.iv"
    keyName = os.path.basename(key).rstrip(".hex")
    print(f"encrypting sentences with {keyName} key")
    
    enc = []
    for s in sentences:
        e = encrypt(str(s), key, iv)
        enc.append(e)
    
    fname = f"{OUT_DIR}/{keyName}.csv"
    np.savetxt(fname, np.array(enc, dtype=str), delimiter=",", fmt="%s")
    print(f"wrote file {fname}")

