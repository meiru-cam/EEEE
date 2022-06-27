import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    Adafactor,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
import os
from torch.nn import CrossEntropyLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set random seed
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
set_seed(42)
import json

tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

train_fct = CrossEntropyLoss(ignore_index=-100)

# optimizer
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in t5_model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in t5_model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
# optimizer = AdamW(optimizer_grouped_parameters, lr=3e-4, eps=1e-8)
optimizer = Adafactor(
    optimizer_grouped_parameters,
    lr=3e-4,
    eps=(1e-30, 2e-5),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False
)
# load train_data
# [(input, target), (input, target), ..., (input, target)]
# t5_model.resize_token_embeddings(len(tokenizer))
t5_model.cuda()
t5_model.train()

# dataset preparation

true_false_adjective_tuples = [
                               ("The cat is alive","The cat is dead"),
                               ("The old woman is beautiful","The old woman is ugly"),
                               ("The purse is cheap","The purse is expensive"),
                               ("Her hair is curly","Her hair is straight"),
                               ("The bathroom is clean","The bathroom is dirty"),
                               ("The exam was easy","The exam was difficult"),
                               ("The house is big","The house is small"),
                               ("The house owner is good","The house owner is bad"),
                               ("The little kid is fat","The little kid is thin"),
                               ("She arrived early","She arrived late."),
                               ("John is very hardworking","John is very lazy"),
                               ("The fridge is empty","The fridge is full")

]

epochs = 100
for epoch in range(epochs):
  print ("epoch ",epoch)
  for input,output in true_false_adjective_tuples:
    input_sent = "Event Extract: "+input
    ouput_sent = output

    tokenized_inp = tokenizer(input_sent,  max_length=96, return_tensors="pt", padding=True, truncation=True)
    tokenized_output = tokenizer(ouput_sent, max_length=96, return_tensors="pt", padding=True, truncation=True)


    input_ids  = tokenized_inp["input_ids"].cuda()
    attention_mask = tokenized_inp["attention_mask"].cuda()

    lm_labels= tokenized_output["input_ids"].cuda()
    decoder_attention_mask=  tokenized_output["attention_mask"].cuda()

    # the forward function automatically creates the correct decoder_input_ids
    output = t5_model(input_ids=input_ids, labels=lm_labels,decoder_attention_mask=decoder_attention_mask,attention_mask=attention_mask)
    logits = output.logits

    # loss = train_fct(logits.view(-1, t5_model.config.vocab_size), lm_labels.view(-1))
    """
    NLL loss, change vocab_size
    The proaspat was ugly
    The was ugly
    The was ugly

    default loss, change vocab_size
    The seaman was sad
    The seaman was unhappy
    The seaman was bad

    default loss, default vocab_size
    The sailor was unhappy
    The sailor was happy
    The sailor was sad

    NLL loss, default vocab_size
    The proaspat was ugly
    The<extra_id_-1> was ugly
    The<extra_id_-4> taiat senzati senzati
    """
    loss = output[0]

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

test_sent = 'falsify: The sailor was happy and joyful. </s>'
test_tokenized = tokenizer.encode_plus(test_sent, return_tensors="pt")

test_input_ids  = test_tokenized["input_ids"].cuda()
test_attention_mask = test_tokenized["attention_mask"].cuda()

t5_model.eval()
beam_outputs = t5_model.generate(
    input_ids=test_input_ids,attention_mask=test_attention_mask,
    max_length=64,
    early_stopping=True,
    num_beams=10,
    num_return_sequences=3,
    no_repeat_ngram_size=2
)

for beam_output in beam_outputs:
    sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    print (sent)