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
        self.max_len = args.max_len
        self.train_token_list, self.train_token_id_list = self.process_one_file(args.train_path)
        self.dev_token_list, self.dev_token_id_list = self.process_one_file(args.dev_path)
        self.test_token_list, self.test_token_id_list = self.process_one_file(args.test_path)
        self.train_num, self.dev_num, self.test_num = len(self.train_token_list), len(self.dev_token_list), \
        len(self.test_token_list)
        print('train number:{}, dev number:{}, test number:{}'.format(self.train_num, self.dev_num, self.test_num))
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.bos_token])[0]
        # print('padding token is {}, padding token id {}'.format(self.tokenizer.bos_token, self.pad_token_id))

        self.train_idx_list = [i for i in range(self.train_num)]
        random.shuffle(self.train_idx_list)
        self.dev_idx_list = [j for j in range(self.dev_num)]
        self.test_idx_list = [j for j in range(self.test_num)]
        self.dev_current_idx, self.test_current_idx = 0, 0

    def process_one_file(self, path):
        print('Processing {}'.format(path))
        res_token_list, res_token_id_list = [], []
        with open(path, 'r', encoding = 'utf-8') as i:
            lines = i.readlines()
        n = len(lines)
        p = progressbar.ProgressBar(n)
        p.start()
        for i in range(n):
            p.update(i)
            text = lines[i].strip('\n')
            if len(text) > 0 and not text.isspace():
                self.process_one_text(text, res_token_list, res_token_id_list)
        p.finish()
        print('{} processed!'.format(path))
        return res_token_list, res_token_id_list

    def process_one_text(self, text, res_token_list, res_token_id_list):
        tokens = self.tokenizer.tokenize(text, max_length=self.max_len, truncation=True)
        if len(tokens) <= 1: # filter out too short sequence
            return
        tokens = tokens[:self.max_len]
        res_token_list.append(tokens)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        res_token_id_list.append(token_ids)
        return

    def pad_batch(self, batch_id_list):
        batch_id_list = [torch.LongTensor(item) for item in batch_id_list]
        batch_tensor = rnn.pad_sequence(batch_id_list, batch_first=True)
        # batch_tensor = rnn.pad_sequence(batch_id_list, batch_first=True, padding_value=self.pad_token_id)
        batch_mask = torch.ones_like(batch_tensor)
        # batch_mask = batch_mask.masked_fill(batch_tensor.eq(self.pad_token_id), 0.0).type(torch.FloatTensor)
        return batch_tensor, None #batch_mask

    def process_output(self, batch_tgt_id_list):
        batch_tgt_id_list = [torch.LongTensor(item) for item in batch_tgt_id_list]
        batch_tgt_tensor, _ = self.pad_batch(batch_tgt_id_list) # padded target sequence
        batch_tgt_input_tensor = batch_tgt_tensor[:, :-1].clone() # shift the input and output
        batch_tgt_output_tensor = batch_tgt_tensor[:, 1:].clone()
        # batch_tgt_input_tensor = batch_tgt_tensor.clone() # similar in SimpleTOD, the input and output are the same
        # batch_tgt_output_tensor = batch_tgt_tensor.clone()
        return batch_tgt_input_tensor, batch_tgt_output_tensor

    def parse_batch(self, batch_id_list):
        batch_input, batch_labels = self.process_output(batch_id_list)
        # batch_labels[batch_labels[:, :] == self.pad_token_id] = -100
        return batch_input, batch_labels

    def get_next_train_batch(self, batch_size):
        batch_idx_list = random.sample(self.train_idx_list, batch_size)
        batch_id_list, batch_token_list = [], []

        for idx in batch_idx_list:
            batch_id_list.append(self.train_token_id_list[idx])
            batch_token_list.append(self.train_token_list[idx])
        batch_input_tensor, batch_labels = self.parse_batch(batch_id_list)
        return batch_input_tensor, batch_labels, batch_token_list

    def get_next_validation_batch(self, batch_size, mode):
        batch_id_list, batch_token_list = [], []
        if mode == 'dev':
            curr_select_idx, instance_num = self.dev_current_idx, self.dev_num
            tgt_token_id_list, tgt_token_list = self.dev_token_id_list, self.dev_token_list
        elif mode == 'test':
            curr_select_idx, instance_num = self.test_current_idx, self.test_num
            tgt_token_id_list, tgt_token_list = self.test_token_id_list, self.test_token_list
        else:
            raise Exception('Wrong Validation Mode!!!')

        if curr_select_idx + batch_size < instance_num:
            for i in range(batch_size):
                curr_idx = curr_select_idx + i
                batch_id_list.append(tgt_token_id_list[curr_idx])
                batch_token_list.append(tgt_token_list[curr_idx])
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
                batch_id_list.append(tgt_token_id_list[curr_idx])
                batch_token_list.append(tgt_token_list[curr_idx])
            if mode == 'dev':
                self.dev_current_idx = 0
            else:
                self.test_current_idx = 0
        batch_input_tensor, batch_labels = self.parse_batch(batch_id_list)
        return batch_input_tensor, batch_labels, batch_token_list





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
