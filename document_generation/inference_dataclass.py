import random
import torch
import numpy as np
import progressbar
from torch.nn.utils import rnn
from transformers import GPT2TokenizerFast

class Data:
    def __init__(self, args): #model_name, dev_path, test_path, prefix_len, decoding_len):
        '''
            dev_path, test_path: data path to validate the result
            prefix_len: length of the human-written prefix
            decoding_len: length of generated text continuation
        '''
        self.tokenizer = GPT2TokenizerFast.from_pretrained(args.ckpt_path)
        # special_tokens = []
        # sep_tokens = ["<|trigger|>", "<|endoftrigger|>", '<|content|>', '<|endofcontent|>']
        # role_list = [
        #     'Person', 'Entity', 'Defendant', 'Prosecutor', 'Plaintiff', 'Buyer', 'Artifact', 'Seller', 'Destination',
        #     'Origin', 'Vehicle', 'Agent', 'Attacker', 'Target', 'Victim', 'Instrument', 'Giver', 'Recipient',
        #     'Org', 'Place', 'Adjudicator', 'Beneficiary'
        # ]
        # event_list = ['[Contact_Phone-Write]', '[Personnel_Elect]', '[Justice_Sentence]', '[Life_Die]',
        #               '[Life_Be-Born]',
        #               '[Transaction_Transfer-Ownership]', '[Business_End-Org]', '[Life_Divorce]', '[Justice_Acquit]',
        #               '[Justice_Sue]', '[Justice_Appeal]', '[Justice_Charge-Indict]', '[Business_Declare-Bankruptcy]',
        #               '[Contact_Meet]', '[Personnel_Start-Position]', '[Business_Merge-Org]', '[Conflict_Attack]',
        #               '[Personnel_End-Position]', '[Conflict_Demonstrate]', '[Justice_Execute]',
        #               '[Transaction_Transfer-Money]',
        #               '[Justice_Pardon]', '[Personnel_Nominate]', '[Justice_Arrest-Jail]', '[Justice_Release-Parole]',
        #               '[Justice_Trial-Hearing]', '[Justice_Convict]', '[Business_Start-Org]', '[Life_Injure]',
        #               '[Justice_Extradite]', '[Justice_Fine]', '[Life_Marry]', '[Movement_Transport]']
        #
        # special_tokens += [f"<|{r}|>" for r in role_list]
        # special_tokens += [f"<|endof{r}|>" for r in role_list]
        # special_tokens += ["[None]", "[and]"]
        # self.tokenizer.add_special_tokens({'additional_special_tokens': event_list + sep_tokens + special_tokens})
        # self.tokenizer.add_tokens(event_list + sep_tokens + special_tokens)
        # self.pad_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.bos_token])[0]
        # print ('padding token is {}, padding token id {}'.format(self.tokenizer.bos_token, self.pad_token_id))
        # self.prefix_len, self.decoding_len = prefix_len, decoding_len
        # self.min_len = self.prefix_len + self.decoding_len
        self.min_len = 5

        dev_prefix_token_id_list, dev_prefix_text_list, dev_reference_text_list, \
        dev_reference_continuation_text_list = self.process_one_file(args.dev_path)

        test_prefix_token_id_list, test_prefix_text_list, test_reference_text_list, \
        test_reference_continuation_text_list = self.process_one_file(args.test_path)

        # combine data
        self.prefix_token_id_list = {'dev': dev_prefix_token_id_list, 'test': test_prefix_token_id_list}
        self.prefix_text_list = {'dev': dev_prefix_text_list, 'test': test_prefix_text_list}
        self.reference_text_list = {'dev': dev_reference_text_list, 'test': test_reference_text_list}
        self.reference_continuation_text_list = {'dev': dev_reference_continuation_text_list, 'test': test_reference_continuation_text_list}
        print ('Evaluation dev number is {}, test number is {}'.format(len(self.prefix_token_id_list['dev']),len(self.prefix_token_id_list['test'])))

    def process_one_file(self, path):
        print ('Processing {}'.format(path))
        prefix_token_id_list, prefix_text_list, reference_text_list, \
        reference_continuation_text_list = [], [], [], []

        with open(path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
        n = len(lines)
        # n =10 
        print (n) 
        p = progressbar.ProgressBar(n)
        p.start()
        for i in range(n):
            p.update(i)
            text = lines[i].strip('\n')
            self.process_one_text(text, prefix_token_id_list, prefix_text_list, reference_text_list,
                                  reference_continuation_text_list)
        p.finish()
        print ('{} processed!'.format(path))
        return prefix_token_id_list, prefix_text_list, reference_text_list, reference_continuation_text_list

    def process_one_text(self, text, prefix_token_id_list, prefix_text_list, reference_text_list, \
        reference_continuation_text_list):
        # prefix_text = text.split("<|trigger|>")[0]+"<|trigger|>"
        # reference_continuation_text = text.split("<|trigger|>")[1]
        prefix_text = text.split("<|endofcontent|>")[0]+"<|endofcontent|>"
        reference_continuation_text = text.split("<|endofcontent|>")[1]
        
        # prefix_text = text.split("] <")[0] + "]"
        # reference_continuation_text = " <" + text.split("] <")[1]
        
        prefix_text = prefix_text.replace(" 's", "'s")
        reference_continuation_text = reference_continuation_text.replace(" 's", "'s")
        
        tokens = self.tokenizer.tokenize(prefix_text)
        prefix_tokens = self.tokenizer.tokenize(prefix_text)

        if len(tokens) < self.min_len:
            return

        token_id_list = self.tokenizer.convert_tokens_to_ids(tokens)
        prefix_id_list = self.tokenizer.convert_tokens_to_ids(prefix_tokens)
        prefix_token_id_list.append(prefix_id_list)
        # prefix_text = self.tokenizer.decode(prefix_id_list)
        prefix_text_list.append(prefix_text)
        reference_text_list.append(text)
        # reference_continuation_text = self.tokenizer.decode(token_id_list[self.prefix_len:])
        reference_continuation_text_list.append(reference_continuation_text)
        return
