# coding=utf-8
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
import argparse
import tqdm
import json
import sys
import os
import random
import numpy as np
import progressbar
from inference_dataclass import Data
from torch.nn.utils import rnn

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
    # input_tensor = torch.tensor(input_ids).unsqueeze(0)
    input_tensor = torch.LongTensor(input_ids).view(1, -1)
    if cuda_available:
        input_tensor = input_tensor.cuda(device)

    if args.decode_method == 'beam':
        output = model.generate(input_ids=input_tensor,
                                  num_beams=args.num_beam,
                                  num_return_sequences=3,
                                  max_length=args.max_length,
                                  no_repeat_ngram_size=3, # may harm the modal because multi-event cases
                                  early_stopping=True)
        for i in range(3):
            output_text = data.tokenizer.decode(output[i])
            output_continuation = data.tokenizer.decode(output[i]).split("<|endofcontent|>")[1]
            generated_dict[i] = {'full_text': output_text, 'continuation': output_continuation}
    elif args.decode_method == 'top_k':
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        output = model.generate(
            input_ids=input_tensor,
            do_sample=True,
            max_length=args.max_length,
            top_k=args.k,
        )
        output_text = data.tokenizer.decode(output[0])
        output_continuation = data.tokenizer.decode(output[0]).split("<|endofcontent|>")[1]
        generated_dict[0] = {'full_text': output_text, 'continuation': output_continuation}
    elif args.decode_method == 'top_p':
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        output = model.generate(
            input_ids=input_tensor,
            do_sample=True,
            max_length=args.max_length,
            top_p=args.top_p,
            top_k=0,
            # num_return_sequences=3,
        )
        output_text = data.tokenizer.decode(output[0])
        output_continuation = data.tokenizer.decode(output[0]).split("<|endofcontent|>")[1]
        generated_dict[0] = {'full_text': output_text, 'continuation': output_continuation}
    elif args.decode_method == 'greedy':
        break_tokens = data.tokenizer.encode("<|endoftext|>")
        output_tokens = []
        predicted_index = input_ids[-1]
        # TODO: provide oracle
        while predicted_index not in break_tokens:
            output = model(input_tensor)
            predictions = output[0]
            predicted_index = torch.argmax(predictions[0, -1, :]).item()
            output_tokens += [predicted_index]
            input_tensor = torch.cat((input_tensor, torch.LongTensor([predicted_index]).view(1,-1).cuda(device)), 1)
            # tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
            if len(output_tokens) > args.max_length-len(input_ids):
                break
        output_continuation = data.tokenizer.decode(output_tokens)
        output_text = data.prefix_text_list[data_type][index]+data.tokenizer.decode(output_tokens)
        generated_dict[0] = {'full_text': output_text, 'continuation': output_continuation}
    elif args.decode_method == 'constrained':
        break_tokens = data.tokenizer.encode("<|endoftext|>")
        output_tokens = []
        predicted_index = input_ids[-1]
        # TODO: provide oracle
        flag_1_1 = True
        flag_1 = True
        flag_2 = True
        while predicted_index not in break_tokens:
            output = model(input_tensor)
            predictions = output[0]

            predicted_index = torch.argmax(predictions[0, -1, :]).item()

            if predicted_index == 1279:
                flag_1_1 = False
            if flag_1_1 and predicted_index == 91:
                flag_1 = False
            if flag_1 and predicted_index == 91:
                flag_1_1 = True
            if not flag_1_1 and predicted_index == 29:
                flag_1 = True
            
            if predicted_index == 685:
                flag_2 = False
            if predicted_index == 60:
                flag_2 = True

            if predicted_index not in [685, 60, 91, 29, 1279]:
                if (flag_1 and flag_2):
                    orders = torch.argsort(predictions[0,-1,:], dim=0, descending=True)
                    for i in orders:
                        if i in input_ids:
                            predicted_index = i
                            break

            output_tokens += [predicted_index]
            input_tensor = torch.cat((input_tensor, torch.LongTensor([predicted_index]).view(1,-1).cuda(device)), 1)
            # tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
            if len(output_tokens) > args.max_length-len(input_ids):
                break
        output_continuation = data.tokenizer.decode(output_tokens)
        output_text = data.prefix_text_list[data_type][index]+data.tokenizer.decode(output_tokens)
        generated_dict[0] = {'full_text': output_text, 'continuation': output_continuation}


    # if data.tokenizer.decode(indexed_tokens).endswith('<|endoftrigger|>'):
    #     break
    one_res_dict['generated_result'] = generated_dict
    return one_res_dict


def inference_one_file(args, data, model, cuda_available, device):
    print ('----------------------------------------------------------------')
    print ('Start inference...')
    save_path = args.save_path + args.decode_method + '.json'
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
    parser.add_argument("--k", type=int)
    parser.add_argument("--num_beam", type=int)
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--decode_method", type=str)
    parser.add_argument("--oracle", type=bool)
    # save configuration
    parser.add_argument("--save_path", type=str)
    return parser.parse_args()


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
    data = Data(args)
    print ('Data loaded.')

    print ('Loading pre-trained model...')
    # model = SimCTG(args.ckpt_path, data.pad_token_id)
    # model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained(args.ckpt_path)
    if cuda_available:
        model = model.to(device)
    model.eval()
    print ('Model loaded')

    with torch.no_grad():
        inference_one_file(args, data, model, cuda_available, device)

