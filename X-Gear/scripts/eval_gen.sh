#!/bin/bash

export OMP_NUM_THREADS=4
# export CUDA_VISIBLE_DEVICES=1

# MODEL="./output/ace05_mT5base_mT5_trig_arg_notemp/20220621_020534/best_model.mdl"
# MODEL="./output/ace05_mT5base_et_trig_arg_notemp_notype/20220622_131545/best_model.mdl"
MODEL="./output/ace05_mT5base_mT5_trig_arg_notemp_notype/20220621_141357/best_model.mdl"
# MODEL="./output/ace05_mT5base_mT5_trig_arg_withtemp/20220707_232715/best_model.mdl"
# MODEL="./output/ace05_mT5base_mT5_e2e_et_trig_arg_notemp_notype/20220712_225238/best_model.mdl"
# MODEL="./output/ace05_mT5base_mT5_e2e_et_trig_et_arg_notemp/20220713_120940/best_model.mdl"
# OUTPUT_DIR="./ace05rank/diverse10-single_bleu/"
OUTPUT_DIR="./ace05rank/diverse10-single-trig_bleu/"

# CONFIG_EN="./config/config_T5-base_en.json"
CONFIG_EN="./config/config_ace05_mT5copy-base_en2.json"
# CONFIG_AR="./config/config_ace05_mT5copy-base_ar.json"
# CONFIG_ZH="./config/config_ace05_mT5copy-base_zh.json"

echo "======================"
echo "Predicting for English"
echo "======================"
python ./xgear/evaluate_candidates.py -c $CONFIG_EN -m $MODEL -o $OUTPUT_DIR --beam 10 --beam_group 10 --num_return 10 --type ranktrig --single_only

# echo "======================"
# echo "Predicting for Arabic"
# echo "======================"
# python ./xgear/evaluate.py -c $CONFIG_AR -m $MODEL -o $OUTPUT_DIR/ar

# echo "======================"
# echo "Predicting for Chinese"
# echo "======================"
# python ./xgear/evaluate.py -c $CONFIG_ZH -m $MODEL -o $OUTPUT_DIR/zh
