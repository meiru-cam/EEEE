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
import json
import logging
logging.getLogger('transformers.generation_utils').disabled = True

def inference_one_instance(args, data, model, index, cuda_available, device, data_type):
    one_res_dict = {}
    one_res_dict['data_type'] = data_type
    one_res_dict['prefix_text'] = data.prefix_text_list[data_type][index]
    one_res_dict['reference_text'] = data.reference_text_list[data_type][index]
    one_res_dict['reference_continuation_text'] = data.reference_continuation_text_list[data_type][index]

    generated_dict = {}

    input_ids = data.prefix_token_id_list[data_type][index]
    len_content = len(input_ids)

    input_tensor = torch.LongTensor(input_ids).view(1,-1)
    if cuda_available:
        input_tensor = input_tensor.cuda(device)

    # num_per_instance = args.num_per_instance
    # for idx in range(num_per_instance):
    #     # output = model.fast_contrastive_search(input_tensor, k, alpha, args.decoding_len)
    #     output = model.beam_search(input_tensor, k, args.decoding_len)
    #     output_text = data.tokenizer.decode(output)
    #     output_continuation = data.tokenizer.decode(output).split("<|endofcontent|>")[1]
    #     generated_dict[idx] = {'full_text': output_text, 'continuation':output_continuation}
    # one_res_dict['generated_result'] = generated_dict

    break_tokens = data.tokenizer.encode("<|endoftext|>")
    output_tokens = []
    predicted_index = input_ids[-1]
    # TODO: provide oracle
    while predicted_index not in break_tokens:
        output = model.model(input_tensor, output_hidden_states=True, output_attentions=True)

        logits = output.logits
        last_hidden_states = output.hidden_states[-1]
        p_copy = torch.sigmoid(model.linear_copy(last_hidden_states)) # or torch.tanh [bsz, seqlen, 1]
        original_word_pro = logits * (1 - p_copy)  # [bsz, seqlen, vocab_size]
        attentions = output.attentions[-1]  # batch x head x decoder_length x encoder_length
        attentions = torch.mean(attentions, dim=1)  # batch x decoder_length x encoder_length
        copy_words = input_tensor.unsqueeze(1).repeat(1, attentions.size(1), 1)  # [bsz, seqlen, seqlen]
        logits = torch.scatter_add(original_word_pro, 2, copy_words,
                                      attentions * p_copy)  # in the vocab dimension, copy_words is index, add original_pro with attention weightedly
        
        predictions = logits
        predicted_index = torch.argmax(predictions[0, -1, :]).item()
        output_tokens += [predicted_index]
        input_tensor = torch.cat((input_tensor, torch.LongTensor([predicted_index]).view(1,-1).cuda(device)), 1)
        # tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
        if len(output_tokens) > args.decoding_len:
            break
    
    output_continuation = data.tokenizer.decode(output_tokens)
    # output_text = data.prefix_text_list[index]+data.tokenizer.decode(output_tokens)
    output_text = data.prefix_text_list[data_type][index]+data.tokenizer.decode(output_tokens)
    generated_dict[0] = {'full_text': output_text, 'continuation': output_continuation}

    one_res_dict['generated_result'] = generated_dict
    return one_res_dict

def inference_one_file(args, data, model, cuda_available, device):
    print ('----------------------------------------------------------------')
    print ('Start inference...')
    # save_path = args.save_path + args.decode_method + '.json'
    save_path = args.save_path
    result_list = []

    # inference dev data
    for data_type in ['dev', 'test']:
        data_num = len(data.prefix_token_id_list[data_type])
        p = progressbar.ProgressBar(data_num)
        p.start()
        with torch.no_grad():
            for index in range(data_num):
                try:
                    p.update(index)
                except:
                    print('current index ', index)
                one_res_dict = inference_one_instance(args, data, model, index,
                    cuda_available, device, data_type=data_type)
                result_list.append(one_res_dict)
            p.finish()
    with open(save_path, 'w') as outfile:
        print('save')
        json.dump(result_list, outfile, indent=4)

    print ('Inference completed.')

def parse_config():
    parser = argparse.ArgumentParser()
    # model and data configuration
    parser.add_argument("--ckpt_path", type=str, help="path of the pre-trained checkpoint")
    parser.add_argument("--dev_path", type=str)
    parser.add_argument("--test_path", type=str)
    # evaluation configuration
    parser.add_argument("--prefix_len", type=int, default=32)
    parser.add_argument("--decoding_len", type=int, default=128)
    parser.add_argument("--num_per_instance", type=int, help="how many samples to generate per instance.")
    # save configuration
    parser.add_argument("--k", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--save_path", type=str)
    return parser.parse_args()

import argparse
if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            print ('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
        else:
            print ('Using single GPU.')
    else:
        pass
    args = parse_config()
    device = torch.device('cuda')

    print ('Loading data...')
    from inference_dataclass import Data
    data = Data(args)
    print ('Data loaded.')

    print ('Loading pre-trained model...')
    from simctg import SimCTG
    # model = SimCTG(args.ckpt_path, data.pad_token_id)
    model = SimCTG('gpt2', 20256)
    model.load_state_dict(torch.load(args.ckpt_path+'model.mdl'))
    # model = SimCTG('gpt2', data.pad_token_id)
    if cuda_available:
        model = model.to(device)
    model.eval()
    print ('Model loaded') 

    with torch.no_grad():
        inference_one_file(args, data, model, cuda_available, device)
