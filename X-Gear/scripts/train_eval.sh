#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# python ./xgear/cl_eval.py --cuda --log --batch_size 1 --epoch 50 --accumulate_step 8 --warmup_steps 500 --max_lr 2e-3 --datatype diverse5-single_bleu

python ./xgear/cl_eval.py --cuda --log --evaluate --model_pt 22-07-18-1/scorer.bin --datatype diverse5-single