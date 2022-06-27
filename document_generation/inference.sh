CUDA_VISIBLE_DEVICES="0" python inference.py\
    --ckpt_path ../trained_model/ace_all_simctg_margin0_pad_eval_sep_cor/training_step_9200_train_mle_loss_0.85873_train_cl_loss_0.0_dev_loss_2.46514_dev_score_40.98/ \
    --dev_path ../processed_data/toddata_all_new_sep/dev_cor.txt \
    --test_path ../processed_data/toddata_all_new_sep/test_cor.txt \
    --prefix_len 256\
    --decoding_len 512\
    --num_per_instance 1\
    --k 8\
    --alpha 0.6\
    --save_path ../generated/simctg_shifted_margin0_pad_greedy_sep_cor.json
