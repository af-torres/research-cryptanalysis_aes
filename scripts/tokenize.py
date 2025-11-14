import numpy as np

PAD_IDX = 256
SOS_IDX = 257
EOS_IDX = 258

PLAIN_TEXT_DATA_FILE = "./data/englishSentences.csv"
ENCRYPTED_DATA_FILES = [
    "./data/encrypted/128-bytes.csv",
    "./data/encrypted/192-bytes.csv ",
    "./data/encrypted/256-bytes.csv",
]

def byte_tokenize(sentence, add_sos=True, add_eos=True, max_len=None):
    byte_ids = list(sentence.encode('utf-8'))    
    
    if add_sos:
        byte_ids = [SOS_IDX] + byte_ids
    if add_eos:
        byte_ids = byte_ids + [EOS_IDX]
    
    if max_len is not None:
        padding = [PAD_IDX] * max(0, max_len - len(byte_ids))
        byte_ids = byte_ids + padding
    
    return byte_ids


