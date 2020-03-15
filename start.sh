#!/usr/bin/env bash
export PATH="/opt/anaconda3/bin:$PATH"
conda init bash
conda activate ./condaenv

# to go back to normal $PATH
# source ~/.bash_profile