#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=120G
#SBATCH -G 1
#SBATCH --nodes=1
#SBATCH -t 32:00:00
#SBATCH --partition=uri-gpu

#0 "-d short_128"
#1 "-d short_192"
#2 "-d short_256"
#3 "-d short_rand_iv_128"
#4 "-d short_rand_iv_192"
#5 "-d short_rand_iv_256"
#6 "-d wiki_128"
#7 "-d wiki_192"
#8 "-d wiki_256"

TASK_ARGS=("-d short_128" "-d short_192" "-d short_256" "-d short_rand_iv_128" "-d short_rand_iv_192" "-d short_rand_iv_256" "-d wiki_128" "-d wiki_192" "-d wiki_256")
WORKING_DIR=/work/pi_kelum_gajamannage_uri_edu/research-cryptanalysis_aes

cd "$WORKING_DIR"

source ./start_venv.sh

python train.py ${TASK_ARGS[$SLURM_ARRAY_TASK_ID]} -e 100 -bs 500 -ms 30000

