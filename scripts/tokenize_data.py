import os
import numpy as np
import pickle
import base64

PAD_IDX = 256
SOS_IDX = 257
EOS_IDX = 258

PLAIN_TEXT_DATA_FILE = "./data/englishSentences.csv"
ENCRYPTED_DATA_FILES = [
    "./data/encrypted/128-bytes.csv",
    "./data/encrypted/192-bytes.csv",
    "./data/encrypted/256-bytes.csv",
]

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

def get_max_len(arr):
    return len(max(arr, key=len)) + 2 # we add two here as SOS AND EOS are added to original sentences

p_set = np.loadtxt(PLAIN_TEXT_DATA_FILE, delimiter=",", dtype=str).tolist()
max_len = get_max_len(p_set)
p_tokens = []
for p in p_set:
    p_tokens.append(byte_tokenize(p, max_len=max_len))

basename = PLAIN_TEXT_DATA_FILE.rstrip(".csv")
p_tokenized_file = f"{basename}-tokens.pkl"
with open(p_tokenized_file, "wb") as f:
    pickle.dump(np.array(p_tokens, dtype=np.uint16), f)

print(f"wrote tokenized file {p_tokenized_file}")

for c_file in ENCRYPTED_DATA_FILES:
    basename = os.path.basename(c_file).rstrip(".csv")

    c = np.loadtxt(c_file, delimiter=",", dtype=str).tolist()
    max_len = get_max_len(c)

    c_tokens = []
    for c_i in c:
        c_tokens.append(byte_tokenize(c_i, max_len=max_len, b64_enc=True))
    
    pkl_file = f"./data/{basename}-tokens.pkl"
    with open(pkl_file, "wb") as f:
        pickle.dump(np.array(c_tokens, dtype=np.uint16), f)
    print(f"wrote tokenized encrypted sentences file {pkl_file}")
