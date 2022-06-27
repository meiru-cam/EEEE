# coding=utf-8
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import argparse
import os
import numpy as np
import logging
import progressbar
from transformers import GPT2LMHeadModel, AutoTokenizer, GPT2TokenizerFast

from transformers import T5ForConditionalGeneration, T5Tokenizer

# TODO: use GPT2-medium or GPT2-large
# TODO: do I need to add PAD? otherwise the position embedding may have different shape

from transformers.optimization import AdamW, get_linear_schedule_with_warmup, Adafactor
import random
import time
import sys
sys.stdout.flush()
sys.stderr.flush()

from dataclass import TextDataset
import evaluation2 as evaluation
# import evaluation
import json

LOGGER = logging.getLogger(__name__)

train_fct = CrossEntropyLoss()
val_fct = CrossEntropyLoss(reduction='none')

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    print("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s: [ %(message)s ]", "%m/%d/%Y %I:%M:%S %p")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)


def parse_config():
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--dev_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--max_len", type=int, default=256)
    # training configuration
    parser.add_argument("--n_gpu", type=int, help="Number of available GPUs")
    parser.add_argument('--cuda', action='store_true', help="use GPU or not", default=False)
    parser.add_argument("--batch_size_per_gpu", type=int, help="batch size for each gpu")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="gradient accumulation step.")
    parser.add_argument("--effective_batch_size", type=int,
        help="effective_bsz = batch_size_per_gpu x number_of_gpu x gradient_accumulation_steps")
    parser.add_argument("--total_steps", type=int, help="total training steps")
    parser.add_argument("--print_every", type=int)
    parser.add_argument("--save_every", type=int)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--save_path_prefix", type=str, help="directory to save the model parameters.")
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def eval_model(args, model, data, cuda_available, device, vocab_size):
    print('start evaluation')
    dataset_batch_size = args.batch_size_per_gpu * args.n_gpu
    eval_step = int(data.dev_num / dataset_batch_size) + 1
    val_loss =  0.
    model.eval()
    with torch.no_grad():
        p = progressbar.ProgressBar(eval_step)
        p.start()
        for idx in range(eval_step):
            p.update(idx)
            batch_input_tensor, batch_labels, _ = \
            data.get_next_validation_batch(batch_size=dataset_batch_size, mode='dev')
            if cuda_available:
                batch_input_tensor = batch_input_tensor.cuda(device)
                batch_labels = batch_labels.cuda(device)
            # outputs = model(batch_input_tensor, labels=batch_labels)
            # one_val_loss = outputs[0] # LM loss

            bsz, seqlen = batch_input_tensor.size()
            outputs = model(input_ids=batch_input_tensor, output_hidden_states=True)
            logits = outputs.logits
            assert logits.size() == torch.Size([bsz, seqlen, vocab_size])
            last_hidden_states = outputs.hidden_states[-1]
            if cuda_available and args.n_gpu>1 and torch.cuda.device_count() > 1:
                assert last_hidden_states.size() == torch.Size([bsz, seqlen, model.module.config.hidden_size])
            else:
                assert last_hidden_states.size() == torch.Size([bsz, seqlen, model.config.hidden_size])
            one_val_loss = val_fct(logits.view(-1, vocab_size), batch_labels.view(-1))
            one_val_loss = one_val_loss.mean()
            val_loss += one_val_loss.item()
        p.finish()
    model.train()
    val_loss = val_loss / eval_step
    return val_loss


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
        # one_dict['prefix_text'] = text.split("<|content|>")[0] + "<|content|>"
        # one_dict['prefix_text'] = text.split("] <")[0] + "]"
        # one_dict['prefix_text'] = text.split("<|trigger|>")[0] + "<|trigger|>"
        one_dict['prefix_text'] = text.split("<|endoftemplate|>")[0] + "<|endoftemplate|>"
        # one_dict['reference_continuation_text'] = text.split("<|endofcontent|>")[1]
        # one_dict['reference_continuation_text'] = text.split("<|content|>")[1]
        # one_dict['reference_continuation_text'] = " <" + text.split("] <")[1]
        # one_dict['reference_continuation_text'] = text.split("<|trigger|>")[1]
        one_dict['reference_continuation_text'] = text.split("<|endoftemplate|>")[1]

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
            
            # output = model.generate(
            #     input_ids=input_tensor,
            #     do_sample=True,
            #     max_length=512,
            #     top_k=20,
            #     pad_token_id=data.tokenizer.eos_token_id
            # )
            # # add special token <|sep|>
            # # special_tokens = {'pad_token':'<|pad|>','sep_token':'<|sep|>'}
            # output_text = data.tokenizer.decode(output[0])
            # output_continuation = data.tokenizer.decode(output[0]).split("<|endofcontent|>")[1]
            # generated_dict[0] = {'full_text': output_text, 'continuation': output_continuation}
            # # if data.tokenizer.decode(indexed_tokens).endswith('<|endoftrigger|>'):
            # #     break
            # one_res_dict['generated_result'] = generated_dict

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
                if len(output_tokens) > 512:
                    break
            output_continuation = data.tokenizer.decode(output_tokens)
            output_text = one_res_dict['prefix_text']+data.tokenizer.decode(output_tokens)
            generated_dict[0] = {'full_text': output_text, 'continuation': output_continuation}
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

    # overall_s = trig_idf*0.2+trig_cls*0.2+argu_idf*0.3+argu_cls*0.3
    overall_s = argu_idf*0.5+argu_cls*0.5
    
    if argu_idf > 10:
        with open('intermediate_dev_results_template_aw.json', 'w') as outfile:
            print('save intermediate dev')
            json.dump(result_list, outfile, indent=4)
    model.train()
    return overall_s


def model_training(args, data, model, tokenizer, total_steps, print_every, save_every, ckpt_save_path,
                   cuda_available, device, vocab_size):
    if os.path.exists(ckpt_save_path):
        pass
    else: # recursively construct directory
        os.makedirs(ckpt_save_path, exist_ok=True)

    max_save_num = 1

    batch_size_per_gpu, gradient_accumulation_steps, number_of_gpu, effective_batch_size = \
    args.batch_size_per_gpu, args.gradient_accumulation_steps, args.n_gpu, args.effective_batch_size
    assert effective_batch_size == batch_size_per_gpu * gradient_accumulation_steps * number_of_gpu

    warmup_steps = int(0.1 * total_steps) # 10% of training steps are used for warmup
    LOGGER.info('total training steps is {}, warmup steps is {}'.format(total_steps, warmup_steps))
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    # optimizer = Adafactor(
    #     model.parameters(),
    #     lr=args.learning_rate,
    #     eps=(1e-30, 2e-5),
    #     clip_threshold=1.0,
    #     decay_rate=-0.8,
    #     beta1=None,
    #     weight_decay=0.0,
    #     relative_step=False,
    #     scale_parameter=False,
    #     warmup_init=False
    # )
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    optimizer.zero_grad()

    effective_batch_acm = 0
    all_batch_step = 1
    print_valid, save_valid = False, False
    train_loss,min_val_loss, max_val_score = 0., 1e10, 0
    train_ave_bleu = 0.

    inf_data = get_inference_data(data)
    print(inf_data[0])

    LOGGER.info('--------------------------------------------------------------------------')
    LOGGER.info('Start Training:')
    model.train()
    number_of_saves = 0

    while effective_batch_acm < total_steps:
        all_batch_step += 1
        train_batch_input_tensor, train_batch_labels, _ = data.get_next_train_batch(batch_size_per_gpu * number_of_gpu)
        if cuda_available:
            train_batch_input_tensor = train_batch_input_tensor.cuda(device)
            train_batch_labels = train_batch_labels.cuda(device)
        # outputs = model(train_batch_input_tensor, labels=train_batch_labels)
        # loss = outputs[0] # MLE loss

        bsz, seqlen = train_batch_input_tensor.size()
        outputs = model(input_ids=train_batch_input_tensor, output_hidden_states=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, vocab_size])
        last_hidden_states = outputs.hidden_states[-1]
        if cuda_available and args.n_gpu>1 and torch.cuda.device_count() > 1:
            assert last_hidden_states.size() == torch.Size([bsz, seqlen, model.module.config.hidden_size])
        else:
            assert last_hidden_states.size() == torch.Size([bsz, seqlen, model.config.hidden_size])
        loss = train_fct(logits.view(-1, vocab_size), train_batch_labels.view(-1))


        loss = loss.mean()
        loss.backward()
        train_loss += loss.item()
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
            LOGGER.info('At training steps {}, training loss is {}'.format(effective_batch_acm,
                one_train_loss))
            print_valid = False

        # saving result
        if effective_batch_acm % save_every == 0 and save_valid:
            number_of_saves += 1

            save_valid = False
            one_train_loss = train_loss / (save_every * gradient_accumulation_steps)

            model.eval()
            one_val_loss = eval_model(args, model, data, cuda_available, device, vocab_size)
            # score = inference(model, data, inf_data, cuda_available, device)
            score=0

            model.train()

            LOGGER.info('At training steps {}, training loss is {}, validation loss is {}, score is {}'.format(effective_batch_acm,
                one_train_loss, one_val_loss, score))

            train_loss = 0.

            if one_val_loss < min_val_loss:
            # if score > max_val_score:
                # in finetuning stage, we always save the model
                min_val_loss = min(one_val_loss, min_val_loss)
                max_val_score = max(score, max_val_score)
                LOGGER.info('Saving model...')
                one_val_ppl = np.exp(one_val_loss)
                one_val_ppl = round(one_val_ppl, 3)
                save_name = 'training_step_{}_train_mle_loss_{}_dev_loss_{}_dev_score_{}'.format(effective_batch_acm,
                round(one_train_loss,5), round(one_val_loss,5), round(max_val_score,3))

                model_save_path = ckpt_save_path + '/' + save_name
                if os.path.exists(model_save_path):
                    pass
                else: # recursively construct directory
                    os.makedirs(model_save_path, exist_ok=True)
                if cuda_available and args.n_gpu>1 and torch.cuda.device_count() > 1:
                    model.module.save_pretrained(model_save_path)
                else:
                    # save model
                    model.save_pretrained(model_save_path)
                # save tokenizer
                tokenizer.save_pretrained(model_save_path)
                LOGGER.info('Model Saved!')

                # --------------------------------------------------------------------------------------------- #
                # removing extra checkpoints...
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
                LOGGER.info('-----------------------------------')
                # --------------------------------------------------------------------------------------------- #
    return model


def add_tokenizers(tokenizer):
    special_tokens = []
    sep_tokens = ["<|trigger|>", "<|endoftrigger|>", '<|content|>', '<|endofcontent|>']
    role_list = [
        'Person', 'Entity', 'Defendant', 'Prosecutor', 'Plaintiff', 'Buyer', 'Artifact', 'Seller', 'Destination',
        'Origin', 'Vehicle', 'Agent', 'Attacker', 'Target', 'Victim', 'Instrument', 'Giver', 'Recipient',
        'Org', 'Place', 'Adjudicator', 'Beneficiary'
    ]
    event_list = ['[Contact_Phone-Write]', '[Personnel_Elect]', '[Justice_Sentence]', '[Life_Die]', '[Life_Be-Born]',
                  '[Transaction_Transfer-Ownership]', '[Business_End-Org]', '[Life_Divorce]', '[Justice_Acquit]',
                  '[Justice_Sue]', '[Justice_Appeal]', '[Justice_Charge-Indict]', '[Business_Declare-Bankruptcy]',
                  '[Contact_Meet]', '[Personnel_Start-Position]', '[Business_Merge-Org]', '[Conflict_Attack]',
                  '[Personnel_End-Position]', '[Conflict_Demonstrate]', '[Justice_Execute]', '[Transaction_Transfer-Money]',
                  '[Justice_Pardon]', '[Personnel_Nominate]', '[Justice_Arrest-Jail]', '[Justice_Release-Parole]',
                  '[Justice_Trial-Hearing]', '[Justice_Convict]', '[Business_Start-Org]', '[Life_Injure]',
                  '[Justice_Extradite]', '[Justice_Fine]', '[Life_Marry]', '[Movement_Transport]']

    special_tokens += [f"<|{r}|>" for r in role_list]
    special_tokens += [f"<|endof{r}|>" for r in role_list]
    special_tokens += ["[None]", "[and]"]
    tokenizer.add_special_tokens({'additional_special_tokens': event_list + sep_tokens + special_tokens})
    # tokenizer.add_tokens(event_list + sep_tokens + special_tokens)
    return tokenizer


def main():
    # set up device
    init_logging()
    args = parse_config()

    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    if cuda_available:
        if args.n_gpu > 1 and torch.cuda.device_count() > 1:
            multi_gpu_training = True
            print('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
        else:
            print('Using single GPU training.')
    else:
        pass

    device = torch.device('cuda')
    model_name = args.model_name
    init_seed(args.random_seed)

    # load model and tokenizer
    LOGGER.info('Initializaing model and tokenizer ...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('vocab_size ', len(tokenizer))
    # tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    # tokenizer = add_tokenizers(tokenizer)
    print('vocab_size ', len(tokenizer))
    vocab_size = len(tokenizer)

    # load data
    LOGGER.info('Loading data...')
    data = TextDataset(args, tokenizer)
    LOGGER.info('Data loaded.')

    if 'gpt2' in model_name:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        # model.resize_token_embeddings(len(tokenizer))
    elif model_name == 't5-base':
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
    embed_dim = model.config.hidden_size
    print('embed_dim ', embed_dim)

    # check multiple GPU
    if cuda_available:
        if multi_gpu_training:
            model = nn.DataParallel(model) # multi-gpu training
        else:
            pass
        model = model.to(device)
    else:
        pass

    # start train
    # if in same format as SimCTG: train on *_cor.txt, but input and labels are the same instead of [:-1] v.s. [1:]
    total_steps, print_every, save_every = args.total_steps, args.print_every, args.save_every
    ckpt_save_path = args.save_path_prefix
    model = model_training(args, data, model, tokenizer, total_steps, print_every, save_every,
        ckpt_save_path, cuda_available, device, vocab_size)
    LOGGER.info('Training stage completed!')
    LOGGER.info('############################################################')


if __name__ == "__main__":
    main()
