#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -G 1
#SBATCH --nodes=1
#SBATCH -t 24:00:00
#SBATCH --partition=uri-gpu

#0 "-d short_128"
#1 "-d short_192"
#2 "-d short_256"
#3 "-d short_rand_iv_128"
#4 "-d short_rand_iv_192"
#5 "-d short_rand_iv_256"

TASK_ARGS=("-d short_128" "-d short_192" "-d short_256" "-d short_rand_iv_128" "-d short_rand_iv_192" "-d short_rand_iv_256")
WORKING_DIR=/home/andres_torres_uri_edu/ondemand/data/sys/myjobs/projects/research-cryptanalysis_aes

cd "$WORKING_DIR"

source .venv/bin/activate

python train.py ${TASK_ARGS[$SLURM_ARRAY_TASK_ID]}
