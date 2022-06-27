#!/bin/bash
CUDA_VISIBLE_DEVICES="0,3" python main_t5.py\
    --model_name t5-base \
    --train_path ./processed_data/toddata_all_new_sep/train_cor.txt\
    --dev_path ./processed_data/toddata_all_new_sep/dev_cor.txt\
    --test_path ./processed_data/toddata_all_new_sep/test_cor.txt\
    --max_length 512\
    --max_output_length 256\
    --n_gpu 2\
    --cuda \
    --batch_size_per_gpu 1\
    --gradient_accumulation_steps 8\
    --effective_batch_size 16\
    --total_steps 10000\
    --print_every 20\
    --save_every 200\
    --learning_rate 1e-5\
    --save_path_prefix ./trained_model/ace_T5_all_tod_shifted_template_aw2/ \
    --random_seed 33
