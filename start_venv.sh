#!/bin/bash

export HF_HOME="/work/pi_kelum_gajamannage_uri_edu/.cache/huggingface/datasets"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/modules/opt/linux-ubuntu24.04-x86_64/miniforge3/24.7.1/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/modules/opt/linux-ubuntu24.04-x86_64/miniforge3/24.7.1/etc/profile.d/conda.sh" ]; then
        . "/modules/opt/linux-ubuntu24.04-x86_64/miniforge3/24.7.1/etc/profile.d/conda.sh"
    else
        export PATH="/modules/opt/linux-ubuntu24.04-x86_64/miniforge3/24.7.1/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda init
conda activate sage
