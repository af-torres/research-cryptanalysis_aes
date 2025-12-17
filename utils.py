
def parse_run_log(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
    
    runs = dict()
    for line in lines:
        id, args_str = line.split(":")
        args = dict()
        for a in args_str.split(";"):
            a = a.strip()
            k, v = a.split("=")
            args[k] = v
        
        runs[id] = args

    return runs

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