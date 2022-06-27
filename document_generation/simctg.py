import sys
import os
import operator
from operator import itemgetter
import torch
from torch import nn
import random
import argparse
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from loss_func import contrastive_loss

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

train_fct = CrossEntropyLoss()
val_fct = CrossEntropyLoss(reduction='none')
class SimCTG(nn.Module):
    # _keys_to_ignore_on_load_missing = ["linear_copy.weight", "linear_copy.bias"]

    def __init__(self, model_name, pad_token_id):
        super(SimCTG, self).__init__()
        from transformers import AutoTokenizer, GPT2LMHeadModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        '''
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
        self.tokenizer.add_special_tokens({"additional_special_tokens": event_list + sep_tokens + special_tokens})
        '''
        self.vocab_size = len(self.tokenizer)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        # self.model.resize_token_embeddings(self.vocab_size)
        self.embed_dim = self.model.config.hidden_size
        self.pad_token_id = pad_token_id
        self.linear_copy = nn.Linear(self.embed_dim, 1)

    def compute_logits_and_hidden_states(self, input_ids):
        # used for advanced decoding
        # input_ids: 1 x seqlen
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits
        return last_hidden_states, logits

    def forward(self, input_ids, labels, margin):
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, output_hidden_states=True, output_attentions=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])

        last_hidden_states = outputs.hidden_states[-1]
        assert last_hidden_states.size() == torch.Size([bsz, seqlen, self.embed_dim])

        # p_copy = torch.sigmoid(self.linear_copy(last_hidden_states)) # or torch.tanh [bsz, seqlen, 1]
        # # Merge distribution
        # original_word_pro = logits * (1 - p_copy)  # [bsz, seqlen, vocab_size]
        
        # # Copy distribution
        # attentions = outputs.attentions[-1]  # batch x head x decoder_length x encoder_length
        # attentions = torch.mean(attentions, dim=1)  # batch x decoder_length x encoder_length

        # # input_ids has shape [bsz, seq]
        # copy_words = input_ids.unsqueeze(1).repeat(1, attentions.size(1), 1)  # [bsz, seqlen, seqlen]

        # # attention # batch x decoder_length x encoder_length
        # # scatter_add((Tensor input, int dim, Tensor index, Tensor src))
        # logits = torch.scatter_add(original_word_pro, 2, copy_words,
        #                               attentions * p_copy)  # in the vocab dimension, copy_words is index, add original_pro with attention weightedly

        mle_loss = train_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        norm_rep = last_hidden_states / last_hidden_states.norm(dim=2, keepdim=True)
        cosine_scores = torch.matmul(norm_rep, norm_rep.transpose(1,2)) 
        assert cosine_scores.size() == torch.Size([bsz, seqlen, seqlen])
        cl_loss = contrastive_loss(margin, cosine_scores, input_ids, self.pad_token_id, prefix_len=0)
        return mle_loss, cl_loss

    def eval_loss(self, input_ids, labels):
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        last_hidden_states = outputs.hidden_states[-1]
        assert last_hidden_states.size() == torch.Size([bsz, seqlen, self.embed_dim])
        mle_loss = val_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        assert mle_loss.size() == torch.Size([bsz * seqlen])
        mask_tmp = labels.masked_fill(~labels.eq(-100), 1.0)
        mask = mask_tmp.masked_fill(mask_tmp.eq(-100), 0.0)
        # sum 
        mle_loss_sum = torch.sum(mle_loss)
        token_num_sum = torch.sum(mask)
        return mle_loss_sum, token_num_sum

    def save_model(self, ckpt_save_path):
        import os
        if os.path.exists(ckpt_save_path):
            pass
        else: # recursively construct directory
            os.makedirs(ckpt_save_path, exist_ok=True)
        # save model
        print(ckpt_save_path+'/model.mdl')
        torch.save(self.state_dict(), ckpt_save_path+'/model.mdl')
        # self.model.save_pretrained(ckpt_save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(ckpt_save_path)

    # decoding functions
    # ------------------------------------------------------- #
    def slow_contrastive_search(self, input_ids, beam_width, alpha, decoding_len):
        '''
           input_ids: prefix input; 1 x prefix_len
           decoding_len: how many tokens to generate
           beam_width: size of candidate pool during decoding
           alpha: regulates importance of model confidence and degeneration penalty
        '''
        # sanity check
        # sanity check
        assert alpha >= 0. and alpha <= 1.0

        from utlis import ContrastiveDecodingOneStep
        for step in range(decoding_len):
            input_ids = ContrastiveDecodingOneStep(self, input_ids, beam_width, alpha)
        return input_ids[0]

    def fast_contrastive_search(self, input_ids, beam_width, alpha, decoding_len):
        '''
           input_ids: prefix input; 1 x prefix_len
           decoding_len: how many tokens to generate
           beam_width: size of candidate pool during decoding
           alpha: regulates importance of model confidence and degeneration penalty
        '''
        self.model.eval()
        from utlis import ContrastiveDecodingOneStepFast
        # sanity check
        assert alpha >= 0. and alpha <= 1.0
        
        # fast mode
        batch_size, seqlen = input_ids.size()
        #generated = [[] for _ in range(batch_size)]
        generated = [item for item in input_ids.tolist()]
        past_key_values = None
        last_hidden_states = None
        logits = None
        for step in range(decoding_len):
            input_ids, past_key_values, last_hidden_states, logits = ContrastiveDecodingOneStepFast(
                self.model,
                input_ids,
                beam_width,
                alpha,
                past_key_values,
                last_hidden_states,
                self.tokenizer,
                logits,
                first_step=step == 0,
            )
            tokens = input_ids.squeeze(dim=-1).tolist()
            for idx, t in enumerate(tokens):
                generated[idx].append(t)
        return generated[0]

    def diverse_contrastive_search(self, input_ids, sample_step, nucleus_p, beam_width, alpha, decoding_len):
        '''
            sample_step: 
                number of steps to decode with nucleus sampling, 
                for the remaining steps we use contrastive search
            decoding_len: 
                the total number of generated tokens
            beam_width: 
                size of candidate pool during decoding
            alpha: 
                regulates importance of model confidence and degeneration penalty

        '''
        contrastive_step = decoding_len - sample_step
        _, prefix_len = input_ids.size()
        # first do sample
        input_ids = self.model.generate(
                            input_ids, 
                            do_sample=True, 
                            max_length=prefix_len+sample_step, 
                            top_p=nucleus_p,
                            top_k=0)
        # then do contrastive search
        output = self.fast_contrastive_search(input_ids, beam_width, alpha, contrastive_step)
        return output

    def greedy_search(self, input_ids, decoding_len):
        _, prefix_len = input_ids.size()
        output = self.model.generate(
                            input_ids, 
                            max_length=prefix_len+decoding_len)
        return output[0]

    def beam_search(self, input_ids, beam_width, decoding_len):
        _, prefix_len = input_ids.size()
        output = self.model.generate(
                            input_ids, 
                            max_length=prefix_len+decoding_len, 
                            num_beams=beam_width)
        return output[0]


    def nucleus_sampling(self, input_ids, nucleus_p, decoding_len):
        _, prefix_len = input_ids.size()
        output = self.model.generate(
                            input_ids, 
                            do_sample=True, 
                            max_length=prefix_len+decoding_len, 
                            top_p=nucleus_p,
                            top_k=0)
        return output[0]
    # ------------------------------------------------------- #

    def compute_correlation_matrix(self, input_ids):        
        _, seq_len = input_ids.size()
        hidden = self.model.base_model(input_ids).last_hidden_state
        #print (hidden)
        norm_hidden = hidden / hidden.norm(dim=2, keepdim=True)
        correlation_matrix = torch.matmul(norm_hidden, norm_hidden.transpose(1,2)).view(seq_len, seq_len)
        return correlation_matrix.detach().numpy()

    # to produce similarity matrix heatmap
    def save_token_similarity_map(self, input_ids, save_name):
        input_ids = torch.LongTensor(input_ids).view(1, -1)
        correlation_matrix = self.compute_correlation_matrix(input_ids)
        df = pd.DataFrame(correlation_matrix)
        df.to_string(index=False)
        df.style.hide_index()
        df.style.hide_index()
        sns.heatmap(df, cmap="Blues", xticklabels=False, yticklabels=False)
        plt.savefig(save_name, format='png', dpi=500, bbox_inches = 'tight')
        plt.show()

