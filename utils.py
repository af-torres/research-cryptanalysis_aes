def parse_run_log(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
    runs = dict()
    for line in lines:
        id, args_str = line.split(":")
        args = dict()
        for a in args_str.split(";"):
            a = a.strip()
            kv = a.split("=")
            if len(kv) != 2: continue
            
            args[kv[0]] = kv[1]
        
        runs[id] = args

    return runs
