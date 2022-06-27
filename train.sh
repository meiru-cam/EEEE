#!/bin/bash
python main.py\
    --model_name gpt2\
    --train_path ./processed_data/toddata_all_sep_template/train_aw_temp.txt\
    --dev_path ./processed_data/toddata_all_sep_template/dev_aw_temp.txt\
    --test_path ./processed_data/toddata_all_sep_template/test_aw_temp.txt\
    --max_len 256\
    --n_gpu 2\
    --cuda \
    --batch_size_per_gpu 1\
    --gradient_accumulation_steps 8\
    --effective_batch_size 16\
    --total_steps 10000\
    --print_every 20\
    --save_every 200\
    --learning_rate 1e-5\
    --save_path_prefix ./trained_model/ace_all_tod_shifted_template_aw2/ \
    --random_seed 33
