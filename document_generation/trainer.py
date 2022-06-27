# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import argparse, os
import random
import numpy as np
import time
import logging
import progressbar
# import evaluation2 as evaluation
import evaluation
import logging
import json
logging.getLogger('transformers.generation_utils').disabled = True


def get_inference_data(data):
    dev_tokens = data.dev_token_id_list
    inf_data=[]
    p = progressbar.ProgressBar(data.dev_num)
    p.start()
    for idx in range(data.dev_num):
        p.update(idx)
        text = data.tokenizer.decode(dev_tokens[idx])
        one_dict = {}
        one_dict['data_type'] = "dev"
        one_dict['reference_text'] = text
        # TODO: change back
        # one_dict['prefix_text'] = text.split("<|endofcontent|>")[0] + "<|endofcontent|>"
        # one_dict['prefix_text'] = text.split("] <")[0] + "]"
        one_dict['prefix_text'] = text.split("<|trigger|>")[0] + "<|trigger|>"
        # one_dict['reference_continuation_text'] = text.split("<|endofcontent|>")[1]
        # one_dict['reference_continuation_text'] = " <" + text.split("] <")[1]
        one_dict['reference_continuation_text'] = text.split("<|trigger|>")[1]

        one_dict['input_ids'] = data.tokenizer.convert_tokens_to_ids(data.tokenizer.tokenize(one_dict['prefix_text']))
        inf_data.append(one_dict)
    p.finish()
    return inf_data


def inference(model, data, inf_data, cuda_available, device):
    print('start inference')
    model.eval()
    result_list = []
    with torch.no_grad():
        pro = progressbar.ProgressBar(data.dev_num)
        pro.start()
        for idx in range(data.dev_num):
        # for idx in range(5):
            pro.update(idx)
            one_res_dict = {}
            one_res_dict['data_type'] = inf_data[idx]['data_type']
            one_res_dict['prefix_text'] = inf_data[idx]['prefix_text']
            one_res_dict['reference_text'] = inf_data[idx]['reference_text']
            one_res_dict['reference_continuation_text'] = inf_data[idx]['reference_continuation_text']
            generated_dict = {}
            input_ids = inf_data[idx]['input_ids']

            # input_tensor = torch.tensor(input_ids).unsqueeze(0)
            input_tensor = torch.LongTensor(input_ids).view(1, -1)
            if cuda_available:
                input_tensor = input_tensor.cuda(device)
            
            break_tokens = data.tokenizer.encode("<|endoftext|>")
            output_tokens = []
            predicted_index = input_ids[-1]
            # TODO: provide oracle
            while predicted_index not in break_tokens:
                if cuda_available and torch.cuda.device_count() > 1:
                    output = model.module.model(input_tensor, output_hidden_states=True, output_attentions=True)
                else:
                    output = model.model(input_tensor, output_hidden_states=True, output_attentions=True)

                logits = output.logits
                # last_hidden_states = output.hidden_states[-1]
                # if cuda_available and torch.cuda.device_count() > 1:
                #     p_copy = torch.sigmoid(model.module.linear_copy(last_hidden_states)) # or torch.tanh [bsz, seqlen, 1]
                # else:
                #     p_copy = torch.sigmoid(model.linear_copy(last_hidden_states)) # or torch.tanh [bsz, seqlen, 1]
                # original_word_pro = logits * (1 - p_copy)  # [bsz, seqlen, vocab_size]
                # attentions = output.attentions[-1]  # batch x head x decoder_length x encoder_length
                # attentions = torch.mean(attentions, dim=1)  # batch x decoder_length x encoder_length
                # copy_words = input_tensor.unsqueeze(1).repeat(1, attentions.size(1), 1)  # [bsz, seqlen, seqlen]
                # logits = torch.scatter_add(original_word_pro, 2, copy_words,
                #                             attentions * p_copy)  # in the vocab dimension, copy_words is index, add original_pro with attention weightedly

                # predictions = output[0]
                predictions = logits
                predicted_index = torch.argmax(predictions[0, -1, :]).item()
                output_tokens += [predicted_index]
                input_tensor = torch.cat((input_tensor, torch.LongTensor([predicted_index]).view(1,-1).cuda(device)), 1)
                # tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
                if len(output_tokens) > 160:
                    break
            output_continuation = data.tokenizer.decode(output_tokens)
            output_text = one_res_dict['prefix_text']+data.tokenizer.decode(output_tokens)
            generated_dict[0] = {'full_text': output_text, 'continuation': output_continuation}
            
            # output = model.model.generate(
            #     input_ids=input_tensor,
            #     do_sample=True,
            #     max_length=150,
            #     top_k=20,
            #     pad_token_id=data.tokenizer.eos_token_id
            # )
            # add special token <|sep|>
            # special_tokens = {'pad_token':'<|pad|>','sep_token':'<|sep|>'}
            # output_text = data.tokenizer.decode(output[0])
            # output_continuation = data.tokenizer.decode(output[0]).split("<|endofcontent|>")[1]
            # generated_dict[0] = {'full_text': output_text, 'continuation': output_continuation}
            # if data.tokenizer.decode(indexed_tokens).endswith('<|endoftrigger|>'):
            #     break

            one_res_dict['generated_result'] = generated_dict
            result_list.append(one_res_dict)
        pro.finish()
    
    
    
    gt_data = [i['reference_continuation_text'] for i in result_list]
    pred_data = [t_i['generated_result'][0]['continuation'] for i, t_i in enumerate(result_list)]
    all_results = evaluation.compute_f1(gt_data, pred_data)
    # conpute trigger related F1
    # print('correct format {} over {}'.format(sum(all_results['format']), len(all_results['gt_triggerwords'])))
    trig_idf, trig_cls = evaluation.compute_trigger_f1(all_results, 'dev')
    argu_idf, argu_cls = evaluation.compute_argument_f1(all_results, 'dev')
    
    if trig_idf > 30:
        with open('intermediate_dev_results_sep_pad.json', 'w') as outfile:
            print('save intermediate dev')
            json.dump(result_list, outfile, indent=4)
    
    # overall_s = trig_idf*0.2+trig_cls*0.2+argu_idf*0.3+argu_cls*0.3
    overall_s = argu_idf*0.5+argu_cls*0.5
    model.train()
    return overall_s


def eval_model(args, model, data, cuda_available, device):
    dataset_batch_size = args.batch_size_per_gpu * args.number_of_gpu
    eval_step = int(data.test_num / dataset_batch_size) + 1
    val_loss, token_sum = 0., 0.
    model.eval()
    with torch.no_grad():
        p = progressbar.ProgressBar(eval_step)
        p.start()
        for idx in range(eval_step):
            p.update(idx)
            batch_input_tensor, batch_labels, _ = \
            data.get_next_validation_batch(batch_size=dataset_batch_size, mode='test')
            if cuda_available:
                batch_input_tensor = batch_input_tensor.cuda(device)
                batch_labels = batch_labels.cuda(device)
            
            if cuda_available and torch.cuda.device_count() > 1:
                one_val_loss, one_val_token_sum = model.module.eval_loss(batch_input_tensor, batch_labels)
            else:
                one_val_loss, one_val_token_sum = model.eval_loss(batch_input_tensor, batch_labels)
            one_val_loss = torch.sum(one_val_loss)
            one_val_token_sum = torch.sum(one_val_token_sum)
            if cuda_available and torch.cuda.device_count() > 1:
                val_loss += one_val_loss.mean().item()
            else:
                val_loss += one_val_loss.item()
            token_sum += one_val_token_sum.item()
        p.finish()
    model.train()
    val_loss = val_loss / token_sum
    return val_loss


def model_training(args, data, model, total_steps, print_every, save_every, ckpt_save_path, cuda_available, device):
    import os
    if os.path.exists(ckpt_save_path):
        pass
    else: # recursively construct directory
        os.makedirs(ckpt_save_path, exist_ok=True)

    max_save_num = 1

    batch_size_per_gpu, gradient_accumulation_steps, number_of_gpu, effective_batch_size = \
    args.batch_size_per_gpu, args.gradient_accumulation_steps, args.number_of_gpu, args.effective_batch_size
    assert effective_batch_size == batch_size_per_gpu * gradient_accumulation_steps * number_of_gpu

    warmup_steps = int(0.1 * total_steps) # 10% of training steps are used for warmup
    print ('total training steps is {}, warmup steps is {}'.format(total_steps, warmup_steps))
    from transformers.optimization import AdamW, get_linear_schedule_with_warmup, Adafactor
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    print(model.parameters())
    # optimizer = Adafactor(model.parameters(), relative_step=False, warmup_init=False, lr=args.learning_rate, scale_parameter=False)
    # scheduler = AdafactorSchedule(optimizer)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    optimizer.zero_grad()

    effective_batch_acm = 0
    all_batch_step = 1
    print_valid, save_valid = False, False
    train_loss, train_cl_loss, min_val_loss, max_val_score = 0., 0., 1e10, -100
    train_ave_bleu = 0.

    inf_data = get_inference_data(data)

    print ('--------------------------------------------------------------------------')
    print ('Start Training:')
    model.train()
    number_of_saves = 0

    while effective_batch_acm < total_steps:
        all_batch_step += 1
        train_batch_input_tensor, train_batch_labels, _ = data.get_next_train_batch(batch_size_per_gpu * number_of_gpu)
        if cuda_available:
            train_batch_input_tensor = train_batch_input_tensor.cuda(device)
            train_batch_labels = train_batch_labels.cuda(device)
        mle_loss, cl_loss = model(train_batch_input_tensor, train_batch_labels, args.margin)

        loss = mle_loss + cl_loss
        loss = loss.mean()
        loss.backward()
        if cuda_available and torch.cuda.device_count() > 1:
            train_loss += mle_loss.mean().item()
            train_cl_loss = cl_loss.mean().item()
        else:
            train_loss += mle_loss.item()
            train_cl_loss += cl_loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # parameter update
        if all_batch_step % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            effective_batch_acm += 1
            print_valid, save_valid = True, True

        # print intermediate result
        if effective_batch_acm % print_every == 0 and print_valid:
            denominator = (effective_batch_acm - (number_of_saves * save_every)) * gradient_accumulation_steps
            one_train_loss = train_loss / denominator
            one_train_cl_loss = train_cl_loss / denominator
            print ('At training steps {}, training MLE loss is {}, train CL loss is {}'.format(effective_batch_acm, 
                one_train_loss, one_train_cl_loss))
            print_valid = False

        # saving result
        if effective_batch_acm % save_every == 0 and save_valid:
            number_of_saves += 1

            save_valid = False
            one_train_loss = train_loss / (save_every * gradient_accumulation_steps)
            one_train_cl_loss = train_cl_loss / (save_every * gradient_accumulation_steps)

            model.eval()
            one_val_loss = eval_model(args, model, data, cuda_available, device)
            score = inference(model, data, inf_data, cuda_available, device)

            model.train()

            print ('At training steps {}, training MLE loss is {}, train CL loss is {}, validation loss is {}, score is {}'.format(effective_batch_acm, 
                one_train_loss, one_train_cl_loss, one_val_loss, score))

            train_loss, train_cl_loss = 0., 0.

            # if one_val_loss < min_val_loss:
            if score > max_val_score:
                # in finetuning stage, we always save the model
                # min_val_loss = min(one_val_loss, min_val_loss)
                max_val_score = max(score, max_val_score)
                print ('Saving model...')
                one_val_ppl = np.exp(one_val_loss)
                one_val_ppl = round(one_val_ppl, 3)
                save_name = 'training_step_{}_train_mle_loss_{}_train_cl_loss_{}_dev_loss_{}_dev_score_{}'.format(effective_batch_acm,
                round(one_train_loss,5), round(one_train_cl_loss,5), round(one_val_loss,5), round(max_val_score,3))

                model_save_path = ckpt_save_path + '/' + save_name
                import os
                if os.path.exists(model_save_path):
                    pass
                else: # recursively construct directory
                    os.makedirs(model_save_path, exist_ok=True)
                if cuda_available and torch.cuda.device_count() > 1:
                    model.module.save_model(model_save_path)
                else:
                    model.save_model(model_save_path)
                print ('Model Saved!')

                # --------------------------------------------------------------------------------------------- #
                # removing extra checkpoints...
                import os
                from operator import itemgetter
                fileData = {}
                test_output_dir = ckpt_save_path
                for fname in os.listdir(test_output_dir):
                    if fname.startswith('training_step'):
                        fileData[fname] = os.stat(test_output_dir + '/' + fname).st_mtime
                    else:
                        pass
                sortedFiles = sorted(fileData.items(), key=itemgetter(1))

                if len(sortedFiles) < max_save_num:
                    pass
                else:
                    delete = len(sortedFiles) - max_save_num
                    for x in range(0, delete):
                        one_folder_name = test_output_dir + '/' + sortedFiles[x][0]
                        os.system('rm -r ' + one_folder_name)
                print ('-----------------------------------')
                # --------------------------------------------------------------------------------------------- #
    return model

