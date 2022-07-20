#!/bin/bash

# for lr in 2e-3 4e-3
#   do for accu in 4 8 12
#       do python ./xgear/cl_eval.py --cuda --log --batch_size 4 --epoch 50 --accumulate_step $accu --warmup_steps 500 --max_lr $lr --datatype diverse5-single_bleu
#       done
#   done

accu=4
lr=2e-3
python ./xgear/cl_eval.py --cuda --log --batch_size 4 --epoch 50 --accumulate_step $accu --warmup_steps 500 --max_lr $lr --datatype diverse10-single_bleu

# python ./xgear/cl_eval.py --cuda --log --evaluate --model_pt 22-07-18-1/scorer.bin --datatype diverse5-single

# python ./scripts/trail.py