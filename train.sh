#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=120G
#SBATCH -G 1
#SBATCH --nodes=1
#SBATCH -t 32:00:00
#SBATCH --partition=uri-gpu

##### LSTM
#0 "-d short_128"
#1 "-d short_192"
#2 "-d short_256"
#3 "-d short_rand_iv_128"
#4 "-d short_rand_iv_192"
#5 "-d short_rand_iv_256"
#6 "-d wiki_128"
#7 "-d wiki_192"
#8 "-d wiki_256"
#9 "-d wiki_rand_iv_128"
#10 "-d wiki_rand_iv_192"
#11 "-d wiki_rand_iv_256"

##### GRU
#12 "-d wiki_128"
#13 "-d wiki_192"
#14 "-d wiki_256"
#15 "-d wiki_rand_iv_128"
#16 "-d wiki_rand_iv_192"
#17 "-d wiki_rand_iv_256"

##### RNN
#18 "-d wiki_128"
#19 "-d wiki_192"
#20 "-d wiki_256"
#21 "-d wiki_rand_iv_128"
#22 "-d wiki_rand_iv_192"
#23 "-d wiki_rand_iv_256"

####################
# Reduced Char Set
####################
##### LSTM
#24 "-d wiki_128_rc"
#25 "-d wiki_192_rc"
#26 "-d wiki_256_rc"

TASK_ARGS=("-d short_128" "-d short_192" "-d short_256" \
    "-d short_rand_iv_128" "-d short_rand_iv_192" "-d short_rand_iv_256" \
    "-d wiki_128 -m lstm" "-d wiki_192 -m lstm" "-d wiki_256 -m lstm" \
    "-d wiki_rand_iv_128 -m lstm" "-d wiki_rand_iv_192 -m lstm" "-d wiki_rand_iv_256 -m lstm" \
    "-d wiki_128 -m gru" "-d wiki_192 -m gru" "-d wiki_256 -m gru" \
    "-d wiki_rand_iv_128 -m gru" "-d wiki_rand_iv_192 -m gru" "-d wiki_rand_iv_256 -m gru" \
    "-d wiki_128 -m rnn" "-d wiki_192 -m rnn" "-d wiki_256 -m rnn" \
    "-d wiki_rand_iv_128 -m rnn" "-d wiki_rand_iv_192 -m rnn" "-d wiki_rand_iv_256 -m rnn"\
    "-d wiki_128_rc" "-d wiki_192_rc" "-d wiki_256_rc")
WORKING_DIR=/work/pi_kelum_gajamannage_uri_edu/research-cryptanalysis_aes

cd "$WORKING_DIR"

source ./start_venv.sh

python train.py ${TASK_ARGS[$SLURM_ARRAY_TASK_ID]} -e 500 -bs 500 -ms 30000
