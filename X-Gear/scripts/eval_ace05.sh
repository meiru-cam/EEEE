#!/bin/bash

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

MODEL="./output/ace05_mT5base_mT5_trig_arg_notemp/20220621_020534/best_model.mdl"
OUTPUT_DIR="./predictions/ace05_mT5_trig_arg_notemp_mix"

# CONFIG_EN="./config/config_T5-base_en.json"

CONFIG_EN="./config/config_ace05_mT5copy-base_en.json"
# CONFIG_AR="./config/config_ace05_mT5copy-base_ar.json"
# CONFIG_ZH="./config/config_ace05_mT5copy-base_zh.json"

echo "======================"
echo "Predicting for English"
echo "======================"
python ./xgear/evaluate_mix.py -c $CONFIG_EN -m $MODEL -o $OUTPUT_DIR/en

# echo "======================"
# echo "Predicting for Arabic"
# echo "======================"
# python ./xgear/evaluate.py -c $CONFIG_AR -m $MODEL -o $OUTPUT_DIR/ar

# echo "======================"
# echo "Predicting for Chinese"
# echo "======================"
# python ./xgear/evaluate.py -c $CONFIG_ZH -m $MODEL -o $OUTPUT_DIR/zh
