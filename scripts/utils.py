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

def byte_tokenize(sentence, add_sos=True, add_eos=True, max_len=None, b64_enc=False):
    PAD_IDX = 256
    SOS_IDX = 257
    EOS_IDX = 258

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
        if text is None: raise ValueError("sentence text must not be None") 

        b = len(text.encode("utf-8"))
        if b > max_bytes:
            max_bytes = b
    return max_bytes + 2  # + [SOS, EOS]

