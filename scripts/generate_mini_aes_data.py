from datasets import Dataset
from scripts.utils import shard_into_files
import random

MAX_SEQUENCE_LEN = 512
MIN_SEQUENCE_LEN = 16
TOTAL_SEQUENCES = 30000

DATA_DIR = "./data/plain_text/mini_aes"
MAX_SHARD_SIZE = 15000

def random_sequence_len(min_value, max_value):
    k_min = min_value // 16
    k_max = max_value // 16
    return 16 * random.randint(k_min, k_max)

def random_sequence():
    length = random_sequence_len(MIN_SEQUENCE_LEN, MAX_SEQUENCE_LEN)
    return ",".join([str(random.randint(0, 15)) for _ in range(length)])

idx = []
sequences = []
for i in range(TOTAL_SEQUENCES):
    idx.append(i)
    sequences.append(random_sequence())

ds = Dataset.from_dict({
    "text": sequences,
    "_idx": idx,
})
shard_into_files(ds, DATA_DIR, len(ds) // MAX_SHARD_SIZE) # type: ignore
