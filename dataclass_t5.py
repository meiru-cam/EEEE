### Dataset should return input_ids, attention_masks, labels

### to add tokens
# tokenizer.add_tokens(list_of_new_tokens)
# # resize the embeddings
# model.resize_token_embeddings(len(tokenizer))

### task prefix is optional
### don't need to register the task
### tokenizer will handle the <eos> tokens

### use AdaFactor
### Set both max length and min length to None for model.generate(), and then the model will stop only when EOS token is the most probable output.
### modify some of these config parameters at load, e.g. by model.config.max_length = new_value


### TODO: forward function need input_ids, attention_mask, decoder_ids, decoder_attention_mask, lm_labels
# lm_labels is the target_ids

import torch
# from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
# from torch.utils.data.distributed import DistributedSampler
import os
import logging
import random
import progressbar
from torch.nn.utils import rnn

class TextDataset():
    def __init__(self, args, tokenizer):
        assert os.path.isfile(args.train_path)
        assert os.path.isfile(args.dev_path)
        assert os.path.isfile(args.test_path)

        self.tokenizer = tokenizer
        self.max_length = self.max_output_length = args.max_length

        if args.max_output_length is not None:
            self.max_output_length = args.max_output_length
        
        # these are specific to mT5
        self.in_start_code = None                      # no BOS token
        self.out_start_code = tokenizer.pad_token_id   # output starts with PAD token

        self.train_enc_id_list, self.train_attn_list, self.train_dec_id_list, self.train_dec_attn_list = self.process_one_file(args.train_path)
        self.dev_enc_id_list, self.dev_attn_list, self.dev_dec_id_list, self.dev_dec_attn_list = self.process_one_file(args.dev_path)
        self.test_enc_id_list, self.test_attn_list, self.test_dec_id_list, self.test_dec_attn_list = self.process_one_file(args.test_path)
        
        self.train_num, self.dev_num, self.test_num = len(self.train_enc_id_list), len(self.dev_enc_id_list), len(self.test_enc_id_list)
        print('train number:{}, dev number:{}, test number:{}'.format(self.train_num, self.dev_num, self.test_num))
        
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token_id])[0]
        # print('padding token is {}, padding token id {}'.format(self.tokenizer.bos_token, self.pad_token_id))
        self.train_idx_list = [i for i in range(self.train_num)]
        random.shuffle(self.train_idx_list)
        self.dev_idx_list = [j for j in range(self.dev_num)]
        self.test_idx_list = [j for j in range(self.test_num)]
        self.dev_current_idx, self.test_current_idx = 0, 0


    def process_one_file(self, path):
        print('Processing {}'.format(path))
        input_ids, attention_mask, decoder_ids, decoder_attention_mask = [], [], [], []

        with open(path, 'r', encoding = 'utf-8') as i:
            lines = i.readlines()

        n = len(lines)
        p = progressbar.ProgressBar(n)
        p.start()
        for i in range(n):
            p.update(i)
            text = lines[i].strip('\n')
            if len(text) > 0 and not text.isspace():
                self.process_one_text(text, input_ids, attention_mask, decoder_ids, decoder_attention_mask)
        p.finish()
        print('{} processed!'.format(path))
        return input_ids, attention_mask, decoder_ids, decoder_attention_mask


    def process_one_text(self, text, input_ids, attention_mask, decoder_ids, decoder_attention_mask):
        # separate input and target
        input_text = text.split("<|content|> ")[1].split(" <|endofcontent|>")[0]
        target_text = text.split("<|endofcontent|> ")[1].split(" <|endoftext|>")[0]

        # encoder inputs
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, max_length=self.max_length+2, truncation=True)
        enc_idxs = inputs['input_ids']
        enc_attn = inputs['attention_mask']
        assert enc_idxs.size(1) < self.max_length+2
        input_ids.append(enc_idxs)
        attention_mask.append(enc_attn)

        # decoder inputs
        targets = self.tokenizer(target_text, return_tensors='pt', padding=True, max_length=self.max_output_length+2, truncation=True)
        dec_idxs = targets['input_ids']
        dec_attn = targets['attention_mask']
        assert dec_idxs.size(1) < self.max_output_length+2
        decoder_ids.append(dec_idxs)
        decoder_attention_mask.append(dec_attn)

        return 


    def get_next_train_batch(self, batch_size):
        batch_idx_list = random.sample(self.train_idx_list, batch_size)
        batch_enc_idxs, batch_ori_dec_idxs, batch_attn_mask, batch_ori_dec_attn_mask = [], [], [], []

        for idx in batch_idx_list:
            batch_enc_idxs.append(self.train_enc_id_list[idx])
            batch_ori_dec_idxs.append(self.train_dec_id_list[idx])
            batch_attn_mask.append(self.train_attn_list[idx])
            batch_ori_dec_attn_mask.append(self.train_dec_attn_list[idx])
        
        # add PAD token as the start token
        tt = torch.ones((batch_size, 1), dtype=torch.long)
        tt[:] = self.tokenizer.pad_token_id   
        batch_dec_idxs = torch.cat((tt, batch_dec_idxs), dim=1)
        batch_dec_attn = torch.cat((torch.ones((batch_size, 1), dtype=torch.long), batch_ori_dec_attn_mask), dim=1)
        
        # labels
        tt = torch.ones((batch_size, 1), dtype=torch.long)
        tt[:] = self.tokenizer.pad_token_id
        raw_lbl_idxs = torch.cat((batch_dec_idxs[:, 1:], tt), dim=1)
        batch_lbl_attn = torch.cat((batch_dec_attn[:, 1:], torch.zeros((batch_size, 1), dtype=torch.long)), dim=1)
        batch_lbl_idxs = raw_lbl_idxs.masked_fill(batch_lbl_attn==0, -100) # ignore padding

        return batch_enc_idxs, batch_attn_mask, batch_dec_idxs, batch_dec_attn, batch_lbl_idxs

    def get_next_validation_batch(self, batch_size, mode):
        batch_enc_idxs, batch_ori_dec_idxs, batch_attn_mask, batch_ori_dec_attn_mask = [], [], [], []
        if mode == 'dev':
            curr_select_idx, instance_num = self.dev_current_idx, self.dev_num
            tgt_enc_idx_list, tgt_enc_attn_idx_list, tgt_dec_idx_list, tgt_dec_attn_list = \
                self.dev_enc_id_list, self.dev_attn_list, self.dev_dec_id_list, self.dev_dec_attn_list
        elif mode == 'test':
            curr_select_idx, instance_num = self.test_current_idx, self.test_num
            tgt_enc_idx_list, tgt_enc_attn_idx_list, tgt_dec_idx_list, tgt_dec_attn_list = \
                self.test_enc_id_list, self.test_attn_list, self.test_dec_id_list, self.test_dec_attn_list
        else:
            raise Exception('Wrong Validation Mode!!!')

        if curr_select_idx + batch_size < instance_num:
            for i in range(batch_size):
                curr_idx = curr_select_idx + i
                batch_enc_idxs.append(tgt_enc_idx_list[curr_idx])
                batch_ori_dec_idxs.append(tgt_dec_idx_list[curr_idx])
                batch_attn_mask.append(tgt_enc_attn_idx_list[curr_idx])
                batch_ori_dec_attn_mask.append(tgt_dec_attn_list[curr_idx])
            if mode == 'dev':
                self.dev_current_idx += batch_size
            else:
                self.test_current_idx += batch_size
        else:
            for i in range(batch_size):
                curr_idx = curr_select_idx + i
                if curr_idx > instance_num - 1:
                    curr_idx = 0
                    if mode == 'dev':
                        self.dev_current_idx = 0
                    else:
                        self.test_current_idx = 0
                batch_enc_idxs.append(tgt_enc_idx_list[curr_idx])
                batch_ori_dec_idxs.append(tgt_dec_idx_list[curr_idx])
                batch_attn_mask.append(tgt_enc_attn_idx_list[curr_idx])
                batch_ori_dec_attn_mask.append(tgt_dec_attn_list[curr_idx])
            if mode == 'dev':
                self.dev_current_idx = 0
            else:
                self.test_current_idx = 0
        
        # add PAD token as the start token
        tt = torch.ones((batch_size, 1), dtype=torch.long)
        tt[:] = self.tokenizer.pad_token_id   
        batch_dec_idxs = torch.cat((tt, batch_dec_idxs), dim=1)
        batch_dec_attn = torch.cat((torch.ones((batch_size, 1), dtype=torch.long), batch_ori_dec_attn_mask), dim=1)
        
        # labels
        tt = torch.ones((batch_size, 1), dtype=torch.long)
        tt[:] = self.tokenizer.pad_token_id
        raw_lbl_idxs = torch.cat((batch_dec_idxs[:, 1:], tt), dim=1)
        batch_lbl_attn = torch.cat((batch_dec_attn[:, 1:], torch.zeros((batch_size, 1), dtype=torch.long)), dim=1)
        batch_lbl_idxs = raw_lbl_idxs.masked_fill(batch_lbl_attn==0, -100) # ignore padding

        return batch_enc_idxs, batch_attn_mask, batch_dec_idxs, batch_dec_attn, batch_lbl_idxs





#     def __len__(self):
#         return len(self.examples)
#
#     def __getitem__(self, i):
#         return torch.tensor(self.examples[i], dtype=torch.long)
#
#
# def get_dataloader(dataset, tokenizer, args, split='train'):
#     def collate(examples):
#         if tokenizer._pad_token is None:
#             return pad_sequence(examples, batch_first=True)
#         return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
#
#     if split == 'train':
#         args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
#         batch_size = args.train_batch_size
#         sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
#     else:
#         args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
#         batch_size = args.eval_batch_size
#         sampler = SequentialSampler(dataset)
#
#     dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate)
#     return dataloader, args
