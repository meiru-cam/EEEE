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

import logging
logging.getLogger('transformers.generation_utils').disabled = True

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_config():
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument("--model_name", type=str, default='gpt2')
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--dev_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--max_len", type=int, default=256)
    # mini-batch training configuration
    parser.add_argument("--number_of_gpu", type=int, help="Number of available GPUs.")  
    parser.add_argument("--batch_size_per_gpu", type=int, help='batch size for each gpu.') 
    parser.add_argument("--gradient_accumulation_steps", type=int, help="gradient accumulation step.")
    parser.add_argument("--effective_batch_size", type=int, 
        help="effective_bsz = batch_size_per_gpu x number_of_gpu x gradient_accumulation_steps")
    # pre-training configuration
    parser.add_argument("--total_steps", type=int, 
        help="total effective training steps")
    parser.add_argument("--print_every", type=int, 
        help="how many update steps to print one intermediate result")
    parser.add_argument("--save_every", type=int, 
        help="how many update steps to save one model")
    # learning configuration
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--margin", type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--save_path_prefix", type=str, help="directory to save the model parameters.")
    return parser.parse_args()

def load_previous_best_model(path):
    import os
    filenames = os.listdir(path)
    for file in filenames:
        if file.startswith('training_step'):
            return path + '/' + file
    raise Exception('No best model found!')

def add_tokenizers(tokenizer):
    special_tokens = []
    sep_tokens = ["<|trigger|>", "<|endoftrigger|>", '<|content|>', '<|endofcontent|>']
    role_list = [
        'Person', 'Entity', 'Defendant', 'Prosecutor', 'Plaintiff', 'Buyer', 'Artifact', 'Seller', 'Destination',
        'Origin', 'Vehicle', 'Agent', 'Attacker', 'Target', 'Victim', 'Instrument', 'Giver', 'Recipient',
        'Org', 'Place', 'Adjudicator', 'Beneficiary'
    ]
    event_list = ['[Contact_Phone-Write]', '[Personnel_Elect]', '[Justice_Sentence]', '[Life_Die]',
                  '[Life_Be-Born]',
                  '[Transaction_Transfer-Ownership]', '[Business_End-Org]', '[Life_Divorce]', '[Justice_Acquit]',
                  '[Justice_Sue]', '[Justice_Appeal]', '[Justice_Charge-Indict]', '[Business_Declare-Bankruptcy]',
                  '[Contact_Meet]', '[Personnel_Start-Position]', '[Business_Merge-Org]', '[Conflict_Attack]',
                  '[Personnel_End-Position]', '[Conflict_Demonstrate]', '[Justice_Execute]',
                  '[Transaction_Transfer-Money]',
                  '[Justice_Pardon]', '[Personnel_Nominate]', '[Justice_Arrest-Jail]', '[Justice_Release-Parole]',
                  '[Justice_Trial-Hearing]', '[Justice_Convict]', '[Business_Start-Org]', '[Life_Injure]',
                  '[Justice_Extradite]', '[Justice_Fine]', '[Life_Marry]', '[Movement_Transport]']

    special_tokens += [f"<|{r}|>" for r in role_list]
    special_tokens += [f"<|endof{r}|>" for r in role_list]
    special_tokens += ["[None]", "[and]"]
    tokenizer.add_tokens(event_list + sep_tokens + special_tokens, special_tokens=True)
    return tokenizer

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
            print ('Using single GPU training.')
    else:
        pass
    args = parse_config()
    device = torch.device('cuda')
    model_name = args.model_name

    print ('Loading data...')
    from dataclass import Data
    data = Data(model_name, args.train_path, args.dev_path, args.test_path, args.max_len)
    print ('Data loaded.')

    from trainer import model_training
    print ('############################################################')
    print ('Start Training...')
    from simctg import SimCTG
    print ('Initializaing SimCTG model...')
    model = SimCTG(model_name, data.pad_token_id)
    # model.tokenizer = add_tokenizers(model.tokenizer)

    if cuda_available:
        if multi_gpu_training:
            model = nn.DataParallel(model) # multi-gpu training
        else:
            pass
        model = model.to(device)
    else:
        pass
    print ('Model loaded') 
    total_steps, print_every, save_every = args.total_steps, args.print_every, args.save_every
    ckpt_save_path = args.save_path_prefix
    model = model_training(args, data, model, total_steps, print_every, save_every, 
        ckpt_save_path, cuda_available, device)
    print ('Training stage completed!')
    print ('############################################################')