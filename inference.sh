
# for num_beam in 8 12 16 20 24 28 32 36 40
# # for num_beam in 4
#   do CUDA_VISIBLE_DEVICES='1' python inference.py \
#     --ckpt_path ./trained_model/ace_all_tod_cor_shifted_eval/training_step_6800_train_mle_loss_0.67366_dev_loss_1.01129_dev_score_53.335/ \
#     --dev_path ./processed_data/toddata_all/dev_cor.txt \
#     --test_path ./processed_data/toddata_all/test_cor.txt \
#     --k 20 \
#     --num_beam $num_beam \
#     --top_p 0.9 \
#     --max_length 150 \
#     --decode_method 'beam' \
#     --oracle False \
#     --save_path 'generated/ace_all_tod_cor_shifted_eval_'$num_beam
#   done

#trained_model/ace_all_tod_cor_shifted_reverse/training_step_1200_train_mle_loss_1.0466_dev_loss_1.07923_dev_score_0

CUDA_VISIBLE_DEVICES='1' python inference.py \
  --ckpt_path ./trained_model/ace_all_tod_shifted_template_aw2/training_step_2000_train_mle_loss_0.83126_dev_loss_0.94997_dev_score_0/ \
  --dev_path ./processed_data/toddata_all_sep_template/dev_aw_temp.txt\
  --test_path ./processed_data/toddata_all_sep_template/test_aw_temp.txt\
  --k 20 \
  --num_beam 4\
  --top_p 0.9 \
  --max_length 400 \
  --decode_method 'greedy' \
  --oracle False \
  --save_path 'generated/ace_all_tod_shifted_template_aw2_'