#!/bin/bash

export OMP_NUM_THREADS=4

# CONFIG="./config/config_T5-base_en.json"
CONFIG="./config/config_ace05_mT5copy-base_en.json"
# CONFIG="./config/config_ace05_mT5copy-base_ar.json"
# CONFIG="./config/config_ace05_mT5copy-base_zh.json"

python ./xgear/train.py -c $CONFIG

