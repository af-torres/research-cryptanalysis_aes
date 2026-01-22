import base64

def shard_into_files(enc, baseName, num_shards):
    n = len(enc)
    rows_per_shard = n // num_shards
    for i in range(num_shards):
        start = i * rows_per_shard
        end = (i+1) * rows_per_shard if i < num_shards - 1 else n
        shard = enc.select(range(start, end)) # type: ignore

        fname = f"{baseName}/shard_{i}.csv"
        shard.to_csv(fname)
        print(f"wrote file {fname}")

def byte_tokenize(sentence, add_sos=True, add_eos=True, max_len=None, b64_enc=False, maes_ec=False):
    PAD_IDX = 256
    SOS_IDX = 257
    EOS_IDX = 258

    if b64_enc:
        decoded_sentence = base64.b64decode(sentence)
    elif maes_ec:
        decoded_sentence = [int(char) for char in sentence.split(",")]
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
        if text is None: raise ValueError("sentence text must not be None") 

        b = len(text.encode("utf-8"))
        if b > max_bytes:
            max_bytes = b
    return max_bytes + 2  # + [SOS, EOS]

def get_unique_tokens(ds, col="tokens"):
    def collect_uniques(batch):
        c = batch[col]
        uniques = set()
        for item in c:
            if isinstance(item, str):
                uniques.update(item)   # add characters
            else:
                uniques.update(item)   # add list elements
        return {"_uniques": list(uniques)}

    tmp = ds.map(
        collect_uniques,
        batched=True,
        batch_size=1000,
        remove_columns=ds.column_names,
    )
    unique_vals = set().union(tmp["_uniques"])

    return unique_vals

def encode_to_reduced_vocab(sentence, token_to_idx):
    sentence["tokens"] = [
        token_to_idx[t]
        for t in sentence["tokens"]
    ]
    return sentence