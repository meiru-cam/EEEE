"""
Adjust GPT-2 model for unified event extraction
"""
import glob
import logging
import os
import pickle
import random
import re

import sys
import operator
from operator import itemgetter
import torch
from torch import nn
import argparse
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from loss_func import contrastive_loss

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# import shutil
# from typing import Dict, List, Tuple

# from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm, trange

# from transformers import (
#     WEIGHTS_NAME,
#     # AdamW,
#     GPT2Tokenizer,
#     PreTrainedModel,
#     PreTrainedTokenizer,
#     # get_linear_schedule_with_warmup,
# )

from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

from dataclass.dataset.language_model import *
from utils.model import *
from utils.language_model import get_optimizer_scheduler
from utils.gpt2_args_parser import ArgsParser

train_fct = CrossEntropyLoss()
val_fct = CrossEntropyLoss(reduction='none')
class EEModel(nn.Module):
    def __init__(self, model_name, pad_token_id):
        super(EEModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.embed_dim = self.model.config.hidden_size
        self.pad_token_id = pad_token_id

        special_tokens = []
        sep_tokens = ["<--triggerword-->"]

        role_list = [
            'Person', 'Entity', 'Defendant', 'Prosecutor', 'Plaintiff', 'Buyer', 'Artifact', 'Seller', 'Destination', 
            'Origin', 'Vehicle', 'Agent', 'Attacker', 'Target', 'Victim', 'Instrument', 'Giver', 'Recipient', 
            'Org', 'Place', 'Adjudicator', 'Beneficiary'
        ]
        special_tokens += [f"<--{r}-->" for r in role_list]
        special_tokens += [f"</--{r}-->" for r in role_list]
        # TODO: add event type special tokens
        special_tokens += ["[None]", "[and]"]

        self.tokenizer.add_tokens(sep_tokens+special_tokens)


    def trigger_beam_search():
        # how to allow multiple triggers for one sentence?
        # given the context, K triggers are selected; *beam_width
        # after trigger been generated, 
        # how to measure the confidence of the generated trigger -> threshold 
        return


def get_model_tokenizer(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_class(config=config)

    model.to(args.device)

    if args.model_name_or_path == 'openai-gpt':
        tokenizer.add_special_tokens({'bos_token': '<|endoftext|>'})
        tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
    elif args.model_name_or_path == 'gpt2':
        pass

    return model, tokenizer, model_class, args

