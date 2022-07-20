import os, sys, json, logging, pprint, tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import MT5Tokenizer, T5Tokenizer
from model import GenerativeModel, Prefix_fn_cls
from data import EEDataset
from utils import cal_scores, get_span_idxs, get_span_idxs_zh
from argparse import ArgumentParser, Namespace
import re
from sklearn.metrics import f1_score
from compare_mt.rouge.rouge_scorer import RougeScorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', required=True)
parser.add_argument('-m', '--model', required=True)
parser.add_argument('-o', '--output_dir', type=str, required=True)
parser.add_argument('--constrained_decode', default=False, action='store_true')
parser.add_argument('--beam', type=int, default=4)
parser.add_argument('--beam_group', type=int, default=4)
parser.add_argument('--num_return', type=int, default=1)
parser.add_argument('--type', type=str, default="sep")
parser.add_argument('--single_only', default=False, action="store_true")
parser.add_argument('--trig_style', type=str, default="content-et2trig")
parser.add_argument('--arg_style', type=str, default="content-trig2arg")
args = parser.parse_args()
with open(args.config) as fp:
    config = json.load(fp)
config = Namespace(**config)

# over write beam size
config.beam_size = args.beam

# import template file
if config.dataset == "ace05":
    from template_generate_ace import event_template_generator, IN_SEP, ROLE_LIST, NO_ROLE, AND
    TEMP_FILE = "template_generate_ace"
elif config.dataset == "ere":
    from template_generate_ere import event_template_generator, IN_SEP, ROLE_LIST, NO_ROLE, AND
    TEMP_FILE = "template_generate_ere"
else:
    raise NotImplementedError

# fix random seed
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.enabled = False

# set GPU device
torch.cuda.set_device(config.gpu_device)

assert torch.cuda.is_available()


# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
        
# check valid styles
assert np.all([style in ['triggerword', 'template'] for style in config.input_style])
assert np.all([style in ['argument:roletype'] for style in config.output_style])

# tokenizer
if config.model_name.startswith("google/mt5-"):
    tokenizer = MT5Tokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
elif config.model_name.startswith("copy+google/mt5-"):
    model_name = config.model_name.split('copy+', 1)[1]
    tokenizer = MT5Tokenizer.from_pretrained(model_name, cache_dir=config.cache_dir)
elif config.model_name.startswith("t5-"):
    tokenizer = T5Tokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
elif config.model_name.startswith("copy+t5-"):
    model_name = config.model_name.split('copy+', 1)[1]
    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=config.cache_dir)
else:
    raise NotImplementedError

special_tokens = []
sep_tokens = []
if "triggerword" in config.input_style:
    sep_tokens += [IN_SEP["triggerword"]]
if "template" in config.input_style:
    sep_tokens += [IN_SEP["template"]]
if "argument:roletype" in config.output_style:
    special_tokens += [f"<|{r}|>" for r in ROLE_LIST]
    special_tokens += [f"<|/{r}|>" for r in ROLE_LIST]
    special_tokens += [NO_ROLE, AND]

# special_tokens += ['[PER]', '[ORG]', '[FAC]', '[LOC]', '[WEA]', '[GPE]', '[VEH]']
# special_tokens += ['[\PER]', '[\ORG]', '[\FAC]', '[\LOC]', '[\WEA]', '[\GPE]', '[\VEH]']

# tokenizer.add_tokens(sep_tokens+special_tokens)

# load data
dev_set = EEDataset(tokenizer, config.dev_file, max_length=config.max_length)
test_set = EEDataset(tokenizer, config.test_file, max_length=config.max_length)
dev_batch_num = len(dev_set) // config.eval_batch_size + (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + (len(test_set) % config.eval_batch_size != 0)

train_set = EEDataset(tokenizer, config.train_file, max_length=config.max_length)
train_batch_num = len(train_set) // config.eval_batch_size + (len(train_set) % config.eval_batch_size != 0)

with open(config.vocab_file) as f:
    vocab = json.load(f)

# load model
logger.info(f"Loading model from {args.model}")
model = GenerativeModel(config, tokenizer)
model.load_state_dict(torch.load(args.model, map_location=f'cuda:{config.gpu_device}'))
model.cuda(device=config.gpu_device)
model.eval()

# output directory
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if args.type == 'sep':
    for data_set, batch_num, data_type in zip([dev_set, test_set], [dev_batch_num, test_batch_num], ['dev', 'test']):
        progress = tqdm.tqdm(total=batch_num, ncols=75, desc=data_type)
        gold_triggers, gold_roles, pred_roles = [], [], []
        pred_wnd_ids, gold_outputs, pred_outputs, inputs = [], [], [], []
        pred_trigs, pred_trig_event, gold_trigs, gold_trig_event = [], [], [], []
        count = 0
        for batch in DataLoader(data_set, batch_size=config.eval_batch_size, shuffle=False, collate_fn=data_set.collate_fn):
            progress.update(1)
            batch_pred_roles = [[] for _ in range(config.eval_batch_size)]
            batch_pred_outputs = [[] for _ in range(config.eval_batch_size)]
            batch_gold_outputs = [[] for _ in range(config.eval_batch_size)]
            batch_inputs = [[] for _ in range(config.eval_batch_size)]
            batch_event_templates = []
            for tokens, triggers, roles in zip(batch.tokens, batch.triggers, batch.roles):
                batch_event_templates.append(event_template_generator(tokens, triggers, roles, config.input_style, config.output_style, vocab, config.lang))
            
            ## Stage1: Extract Trigger and Event_type
            # convert EE instances to EAE instances
            trig_inputs, trig_gold_outputs, trig_events, trig_bids = [], [], [], []
            eae_inputs, eae_gold_outputs, eae_events, eae_bids = [], [], [], []
            # create data inputs and output for trigger extraction
            for i, event_temp in enumerate(batch_event_templates):
                for data in event_temp.get_training_data():
                    # eae_inputs.append(data[0].split('<|triggerword|>')[0]+'<|triggerword|> [None] <|template'+data[0].split('<|triggerword|>')[1].split('<|template')[1])
                    # eae_inputs.append(data[0].split(' <|template')[0])
                    trig_inputs.append("TriggerExtract: " + data[0].split(' <|triggerword|>')[0]) # + " [" +data[4].replace(":", "_")+ "]")
                    trig_gold_outputs.append('<|triggerword|> ' + data[0].split('<|triggerword|> ')[1].split(" <|template|")[0] + " [" +data[4].replace(":", "_")+ "]")
                    # trig_gold_outputs.append('<|triggerword|> ' + data[0].split('<|triggerword|> ')[1].split(" <|template|")[0])
                    trig_events.append(data[2])
                    trig_bids.append(i)
                    # batch_inputs[i].append(data[0].split(' <|template')[0])
                    batch_inputs[i].append("TriggerExtract: " + data[0].split(' <|triggerword|>')[0]) # + " [" +data[4].replace(":", "_")+ "]")
            
            # if there is triggers in this batch, predict triggerword and event type
            if len(trig_inputs) > 0:
                trig_inputs = tokenizer(trig_inputs, return_tensors='pt', padding=True, max_length=config.max_length+2)
                enc_idxs = trig_inputs['input_ids']
                enc_idxs = enc_idxs.cuda()
                enc_attn = trig_inputs['attention_mask'].cuda()

                if config.beam_size == 1:
                    model.model._cache_input_ids = enc_idxs
                else:
                    expanded_return_idx = (
                        torch.arange(enc_idxs.shape[0]).view(-1, 1).repeat(1, config.beam_size).view(-1).to(enc_idxs.device)
                    )
                    input_ids = enc_idxs.index_select(0, expanded_return_idx)
                    model.model._cache_input_ids = input_ids
                
                # inference
                with torch.no_grad():
                    if args.constrained_decode:
                        prefix_fn_obj = Prefix_fn_cls(tokenizer, ["[and]"], enc_idxs)
                        outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn, 
                                num_beams=config.beam_size, 
                                max_length=config.max_output_length,
                                forced_bos_token_id=None,
                                prefix_allowed_tokens_fn=lambda batch_id, sent: prefix_fn_obj.get(batch_id, sent)
                                )
                    else:
                        # outputs= model.model.predict(batch, num_beams=config.beam_size, max_length=config.max_output_length)
                        outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn,
                            num_beams=config.beam_size, max_length=config.max_output_length,
                            forced_bos_token_id=None, num_return_sequences=args.num_return, num_beam_groups=args.beam_group, diversity_penalty=1.0) # diverse beam search
                trig_pred_outputs = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
                trig_pred_outputs = np.reshape(trig_pred_outputs, (len(trig_gold_outputs), -1))
                # extract triggerword and event type from the generated outputs
                for p_texts, g_text in zip(trig_pred_outputs, trig_gold_outputs):
                    tag_ge = re.search('\[[^ />][^>]*\]', g_text)
                    gold_event_type = tag_ge.group()[1:-1]
                    gold_trigs.append(g_text[16:tag_ge.start()-1])
                    gold_trig_event.append(g_text[16:])
                    flag = False
                    # loop to check if the ground truth exists in the returned four beams
                    for p_text in p_texts:
                        if not p_text.startswith("<|triggerword|>"):
                            continue
                        tag_pe = re.search('\[[^ />][^>]*\]', p_text)
                        if not tag_pe:
                            continue
                        pred_event_type = tag_pe.group()[1:-1]
                        if p_text[16:] == g_text[16:]:
                            flag = True
                            pred_trigs.append(p_text[16:tag_pe.start()-1])
                            pred_trig_event.append(p_text[16:])
                            break
                    if not flag:
                        tag_pe = re.search('\[[^ />][^>]*\]', p_texts[0])
                        pred_event_type = tag_pe.group()[1:-1]
                        pred_trigs.append(p_texts[0][16:tag_pe.start()-1])
                        pred_trig_event.append(p_texts[0][16:])
                    

            # create data inputs and output for argument extraction
            for i, event_temp in enumerate(batch_event_templates):
                for data in event_temp.get_training_data():
                    # eae_inputs.append(data[0].split('<|triggerword|>')[0]+'<|triggerword|> [None] <|template'+data[0].split('<|triggerword|>')[1].split('<|template')[1])
                    # eae_inputs.append(data[0].split(' <|template')[0])
                    # eae_inputs.append("TriggerExtract: " + data[0].split(' <|triggerword|>')[0])
                    # eae_inputs.append("EventExtract: " + data[0].split(' <|template')[0]) # + " [" +data[4].replace(":", "_")+ "]")
                    eae_inputs.append("EventExtract: " + data[0].split(" <|template")[0] + " [" +data[4].replace(":", "_")+ "] <|template" + data[0].split(" <|template")[1])
                    eae_gold_outputs.append(data[1])
                    eae_events.append(data[2])
                    eae_bids.append(i)
                    # batch_inputs[i].append(data[0].split(' <|template')[0])
                    # batch_inputs[i].append("EventExtract: " + data[0].split(' <|template')[0]) # + " [" +data[4].replace(":", "_")+ "]")
                    batch_inputs[i].append("EventExtract: " + data[0].split(" <|template")[0] + " [" +data[4].replace(":", "_")+ "] <|template" + data[0].split(" <|template")[1])

            # if there are triggers in this batch, predict argument roles
            if len(eae_inputs) > 0:
                eae_inputs = tokenizer(eae_inputs, return_tensors='pt', padding=True, max_length=config.max_length+2)
                enc_idxs = eae_inputs['input_ids']
                enc_idxs = enc_idxs.cuda()
                enc_attn = eae_inputs['attention_mask'].cuda()

                if config.beam_size == 1:
                    model.model._cache_input_ids = enc_idxs
                else:
                    expanded_return_idx = (
                        torch.arange(enc_idxs.shape[0]).view(-1, 1).repeat(1, config.beam_size).view(-1).to(enc_idxs.device)
                    )
                    input_ids = enc_idxs.index_select(0, expanded_return_idx)
                    model.model._cache_input_ids = input_ids
                
                # inference
                with torch.no_grad():
                    if args.constrained_decode:
                        prefix_fn_obj = Prefix_fn_cls(tokenizer, ["[and]"], enc_idxs)
                        outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn, 
                                num_beams=config.beam_size, 
                                max_length=config.max_output_length,
                                forced_bos_token_id=None,
                                prefix_allowed_tokens_fn=lambda batch_id, sent: prefix_fn_obj.get(batch_id, sent)
                                )
                    else:
                        # outputs= model.model.predict(batch, num_beams=config.beam_size, max_length=config.max_output_length)
                        outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn,
                            num_beams=config.beam_size, max_length=config.max_output_length,
                            forced_bos_token_id=None)

                eae_pred_outputs = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
                # extract argument roles from the generated outputs
                for p_text, g_text, info, bid in zip(eae_pred_outputs, eae_gold_outputs, eae_events, eae_bids):
                    if config.model_name.split("copy+")[-1] == 't5-base':
                        p_text = p_text.replace(" |", " <|")
                        if p_text and p_text[0] != "<":
                            p_text = "<" + p_text
                    theclass = getattr(sys.modules[TEMP_FILE], info['event type'].replace(':', '_').replace('-', '_'), False)
                    assert theclass
                    template = theclass(config.input_style, config.output_style, info['tokens'], info['event type'], config.lang, info)            
                    pred_object = template.decode(p_text)

                    for span, role_type, _ in pred_object:
                        # convert the predicted span to the offsets in the passage
                        # Chinese uses a different function since there is no space between Chenise characters
                        if config.lang == "chinese":
                            sid, eid = get_span_idxs_zh(batch.tokens[bid], span, trigger_span=info['trigger span'])
                        else:
                            sid, eid = get_span_idxs(batch.piece_idxs[bid], batch.token_start_idxs[bid], span, tokenizer, trigger_span=info['trigger span'])

                        if sid == -1:
                            continue
                        batch_pred_roles[bid].append(((info['trigger span']+(info['event type'],)), (sid, eid, role_type)))

                    batch_gold_outputs[bid].append(g_text)
                    batch_pred_outputs[bid].append(p_text)

            batch_pred_roles = [sorted(set(role)) for role in batch_pred_roles]
            
            gold_triggers.extend(batch.triggers)
            gold_roles.extend(batch.roles)
            pred_roles.extend(batch_pred_roles)
            pred_wnd_ids.extend(batch.wnd_ids)
            gold_outputs.extend(batch_gold_outputs)
            pred_outputs.extend(batch_pred_outputs)
            inputs.extend(batch_inputs)

        progress.close()

        # calculate scores
        scores = cal_scores(gold_roles, pred_roles)

        print("---------------------------------------------------------------------")
        print('Trigger I {:6.2f}, Trigger C {:6.2f}'.format(f1_score(pred_trigs, gold_trigs, average='macro')*100, f1_score(pred_trig_event, gold_trig_event, average='macro')*100))
        print('Role I     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
            scores['arg_id'][3] * 100.0, scores['arg_id'][2], scores['arg_id'][1], 
            scores['arg_id'][4] * 100.0, scores['arg_id'][2], scores['arg_id'][0], scores['arg_id'][5] * 100.0))
        print('Role C     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
            scores['arg_cls'][3] * 100.0, scores['arg_cls'][2], scores['arg_cls'][1], 
            scores['arg_cls'][4] * 100.0, scores['arg_cls'][2], scores['arg_cls'][0], scores['arg_cls'][5] * 100.0))
        print("---------------------------------------------------------------------")


        # write outputs
        outputs = {}
        for (pred_wnd_id, gold_trigger, gold_role, pred_role, gold_output, pred_output, input) in zip(
            pred_wnd_ids, gold_triggers, gold_roles, pred_roles, gold_outputs, pred_outputs, inputs):
            outputs[pred_wnd_id] = {
                "input": input, 
                "triggers": gold_trigger,
                "gold_roles": gold_role,
                "pred_roles": pred_role,
                "gold_text": gold_output,
                "pred_text": pred_output,
            }

        with open(os.path.join(args.output_dir, f'{data_type}.pred.json'), 'w') as fp:
            json.dump(outputs, fp, indent=2)
elif args.type == 'e2e':
    # End-to-end inference
    for data_set, batch_num, data_type in zip([dev_set, test_set], [dev_batch_num, test_batch_num], ['dev', 'test']):
        progress = tqdm.tqdm(total=batch_num, ncols=75, desc=data_type)
        gold_triggers, gold_roles, pred_roles = [], [], []
        pred_wnd_ids, gold_outputs, pred_outputs, inputs = [], [], [], []
        pred_trigs, pred_trig_event, gold_trigs, gold_trig_event = [], [], [], []
        count = 0
        # evaluate batch
        for batch in DataLoader(data_set, batch_size=config.eval_batch_size, shuffle=False, collate_fn=data_set.collate_fn):
            progress.update(1)
            batch_pred_roles = [[] for _ in range(config.eval_batch_size)]
            batch_pred_outputs = [[] for _ in range(config.eval_batch_size)]
            batch_gold_outputs = [[] for _ in range(config.eval_batch_size)]
            batch_inputs = [[] for _ in range(config.eval_batch_size)]
            batch_event_templates = []
            for tokens, triggers, roles in zip(batch.tokens, batch.triggers, batch.roles):
                batch_event_templates.append(event_template_generator(tokens, triggers, roles, config.input_style, config.output_style, vocab, config.lang))
            
            ## Stage1: Extract Trigger and Event_type
            # convert EE instances to EAE instances
            trig_inputs, trig_gold_outputs, trig_events, trig_bids = [], [], [], []
            eae_inputs, eae_gold_outputs, eae_events, eae_bids = [], [], [], []
            # create data inputs and output for trigger extraction
            for i, event_temp in enumerate(batch_event_templates):
                for data in event_temp.get_training_data():
                    # eae_inputs.append(data[0].split('<|triggerword|>')[0]+'<|triggerword|> [None] <|template'+data[0].split('<|triggerword|>')[1].split('<|template')[1])
                    # eae_inputs.append(data[0].split(' <|template')[0])
                    trig_inputs.append("TriggerExtract: " + data[0].split(' <|triggerword|>')[0] + " [" +data[4].replace(":", "_")+ "]")
                    trig_gold_outputs.append('<|triggerword|> ' + data[0].split('<|triggerword|> ')[1].split(" <|template|")[0])# + " [" +data[4].replace(":", "_")+ "]")
                    # trig_gold_outputs.append('<|triggerword|> ' + data[0].split('<|triggerword|> ')[1].split(" <|template|")[0])
                    trig_events.append(data[2])
                    trig_bids.append(i)
                    # batch_inputs[i].append(data[0].split(' <|template')[0])
                    batch_inputs[i].append("TriggerExtract: " + data[0].split(' <|triggerword|>')[0] + " [" +data[4].replace(":", "_")+ "]")
            # if there is triggers in this batch, predict triggerword and event type
            if len(trig_inputs) > 0:
                trig_inputs = tokenizer(trig_inputs, return_tensors='pt', padding=True, max_length=config.max_length+2)
                enc_idxs = trig_inputs['input_ids']
                enc_idxs = enc_idxs.cuda()
                enc_attn = trig_inputs['attention_mask'].cuda()

                if config.beam_size == 1:
                    model.model._cache_input_ids = enc_idxs
                else:
                    expanded_return_idx = (
                        torch.arange(enc_idxs.shape[0]).view(-1, 1).repeat(1, config.beam_size).view(-1).to(enc_idxs.device)
                    )
                    input_ids = enc_idxs.index_select(0, expanded_return_idx)
                    model.model._cache_input_ids = input_ids
                
                # inference, generate outputs
                with torch.no_grad():
                    if args.constrained_decode:
                        prefix_fn_obj = Prefix_fn_cls(tokenizer, ["[and]"], enc_idxs)
                        outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn, 
                                num_beams=config.beam_size, 
                                max_length=config.max_output_length,
                                forced_bos_token_id=None,
                                prefix_allowed_tokens_fn=lambda batch_id, sent: prefix_fn_obj.get(batch_id, sent)
                                )
                    else:
                        # outputs= model.model.predict(batch, num_beams=config.beam_size, max_length=config.max_output_length)
                        outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn,
                            num_beams=config.beam_size, max_length=config.max_output_length,
                            forced_bos_token_id=None, num_return_sequences=args.num_return, num_beam_groups=args.beam_group, diversity_penalty=1.0) # diverse beam search
                
                trig_pred_outputs = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
                trig_pred_outputs = np.reshape(trig_pred_outputs, (len(trig_gold_outputs), -1))
                # extract triggerword and event type from the generated outputs
                for p_texts, g_text in zip(trig_pred_outputs, trig_gold_outputs):
                    # tag_ge = re.search('\[[^ />][^>]*\]', g_text)
                    # gold_event_type = tag_ge.group()[1:-1]
                    # gold_trigs.append(g_text[16:tag_ge.start()-1])
                    gold_trig_event.append(g_text[16:])
                    flag = False
                    # loop to check if the ground truth exists in the returned four beams
                    for p_text in p_texts:
                        if not p_text.startswith("<|triggerword|>"):
                            continue
                        # tag_pe = re.search('\[[^ />][^>]*\]', p_text)
                        # if not tag_pe:
                            # continue
                        # pred_event_type = tag_pe.group()[1:-1]
                        if p_text[16:] == g_text[16:]:
                            flag = True
                            # pred_trigs.append(p_text[16:tag_pe.start()-1])
                            pred_trig_event.append(p_text[16:])
                            break
                    if not flag:
                        # tag_pe = re.search('\[[^ />][^>]*\]', p_texts[0])
                        # pred_event_type = tag_pe.group()[1:-1]
                        # pred_trigs.append(p_texts[0][16:tag_pe.start()-1])
                        pred_trig_event.append(p_texts[0][16:])
                    

            # create data inputs and output for argument extraction
            # use the pred_trigs generated in the first stage to construct the EE input
            for i, event_temp in enumerate(batch_event_templates):
                for data in event_temp.get_training_data():
                    # eae_inputs.append(data[0].split('<|triggerword|>')[0]+'<|triggerword|> [None] <|template'+data[0].split('<|triggerword|>')[1].split('<|template')[1])
                    # eae_inputs.append(data[0].split(' <|template')[0])
                    # eae_inputs.append("TriggerExtract: " + data[0].split(' <|triggerword|>')[0])
                    # eae_inputs.append("EventExtract: " + data[0].split(' <|template')[0]) # + " [" +data[4].replace(":", "_")+ "]")
                    # eae_inputs.append("EventExtract: " + data[0].split(" <|template")[0] + " [" +data[4].replace(":", "_")+ "] <|template" + data[0].split(" <|template")[1])
                    eae_inputs.append("EventExtract: " + data[0].split(" <|template")[0] + " [" +data[4].replace(":", "_")+ "]" + pred_trig_event[count]) # + " <|template" + data[0].split(" <|template")[1])
                    eae_gold_outputs.append(data[1])
                    eae_events.append(data[2])
                    eae_bids.append(i)
                    # batch_inputs[i].append(data[0].split(' <|template')[0])
                    # batch_inputs[i].append("EventExtract: " + data[0].split(' <|template')[0]) # + " [" +data[4].replace(":", "_")+ "]")
                    # batch_inputs[i].append("EventExtract: " + data[0].split(" <|template")[0] + " [" +data[4].replace(":", "_")+ "] <|template" + data[0].split(" <|template")[1])
                    batch_inputs[i].append("EventExtract: " + data[0].split(" <|template")[0] + " [" +data[4].replace(":", "_")+ "]" + pred_trig_event[count]) # + " <|template" + data[0].split(" <|template")[1])
                    count += 1
            # if there are triggers in this batch, predict argument roles
            if len(eae_inputs) > 0:
                eae_inputs = tokenizer(eae_inputs, return_tensors='pt', padding=True, max_length=config.max_length+2)
                enc_idxs = eae_inputs['input_ids']
                enc_idxs = enc_idxs.cuda()
                enc_attn = eae_inputs['attention_mask'].cuda()

                if config.beam_size == 1:
                    model.model._cache_input_ids = enc_idxs
                else:
                    expanded_return_idx = (
                        torch.arange(enc_idxs.shape[0]).view(-1, 1).repeat(1, config.beam_size).view(-1).to(enc_idxs.device)
                    )
                    input_ids = enc_idxs.index_select(0, expanded_return_idx)
                    model.model._cache_input_ids = input_ids
                
                # inference
                with torch.no_grad():
                    if args.constrained_decode:
                        prefix_fn_obj = Prefix_fn_cls(tokenizer, ["[and]"], enc_idxs)
                        outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn, 
                                num_beams=config.beam_size, 
                                max_length=config.max_output_length,
                                forced_bos_token_id=None,
                                prefix_allowed_tokens_fn=lambda batch_id, sent: prefix_fn_obj.get(batch_id, sent)
                                )
                    else:
                        # outputs= model.model.predict(batch, num_beams=config.beam_size, max_length=config.max_output_length)
                        outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn,
                            num_beams=config.beam_size, max_length=config.max_output_length,
                            forced_bos_token_id=None)

                eae_pred_outputs = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
                # extract argument roles from the generated outputs
                for p_text, g_text, info, bid in zip(eae_pred_outputs, eae_gold_outputs, eae_events, eae_bids):
                    if config.model_name.split("copy+")[-1] == 't5-base':
                        p_text = p_text.replace(" |", " <|")
                        if p_text and p_text[0] != "<":
                            p_text = "<" + p_text
                    theclass = getattr(sys.modules[TEMP_FILE], info['event type'].replace(':', '_').replace('-', '_'), False)
                    assert theclass
                    template = theclass(config.input_style, config.output_style, info['tokens'], info['event type'], config.lang, info)            
                    pred_object = template.decode(p_text)

                    for span, role_type, _ in pred_object:
                        # convert the predicted span to the offsets in the passage
                        # Chinese uses a different function since there is no space between Chenise characters
                        if config.lang == "chinese":
                            sid, eid = get_span_idxs_zh(batch.tokens[bid], span, trigger_span=info['trigger span'])
                        else:
                            sid, eid = get_span_idxs(batch.piece_idxs[bid], batch.token_start_idxs[bid], span, tokenizer, trigger_span=info['trigger span'])

                        if sid == -1:
                            continue
                        batch_pred_roles[bid].append(((info['trigger span']+(info['event type'],)), (sid, eid, role_type)))

                    batch_gold_outputs[bid].append(g_text)
                    batch_pred_outputs[bid].append(p_text)

            batch_pred_roles = [sorted(set(role)) for role in batch_pred_roles]
            
            gold_triggers.extend(batch.triggers)
            gold_roles.extend(batch.roles)
            pred_roles.extend(batch_pred_roles)
            pred_wnd_ids.extend(batch.wnd_ids)
            gold_outputs.extend(batch_gold_outputs)
            pred_outputs.extend(batch_pred_outputs)
            inputs.extend(batch_inputs)

        progress.close()

        # calculate scores
        scores = cal_scores(gold_roles, pred_roles)
        print(f"num pred_trig {len(pred_trigs)}, num gold_trig {len(gold_trigs)}")
        print(f"num pred_trig_event {len(pred_trig_event)}, num gold_trig_event {len(gold_trig_event)}")
        print("first five predictions: ", pred_trig_event[:5], "first five gold: ", gold_trig_event[:5])

        print("---------------------------------------------------------------------")
        print('Trigger I {:6.2f}, Trigger C {:6.2f}'.format(f1_score(pred_trigs, gold_trigs, average='macro')*100, f1_score(pred_trig_event, gold_trig_event, average='macro')*100))
        print('Role I     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
            scores['arg_id'][3] * 100.0, scores['arg_id'][2], scores['arg_id'][1], 
            scores['arg_id'][4] * 100.0, scores['arg_id'][2], scores['arg_id'][0], scores['arg_id'][5] * 100.0))
        print('Role C     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
            scores['arg_cls'][3] * 100.0, scores['arg_cls'][2], scores['arg_cls'][1], 
            scores['arg_cls'][4] * 100.0, scores['arg_cls'][2], scores['arg_cls'][0], scores['arg_cls'][5] * 100.0))
        print("---------------------------------------------------------------------")


        # write outputs
        outputs = {}
        for (pred_wnd_id, gold_trigger, gold_role, pred_role, gold_output, pred_output, input) in zip(
            pred_wnd_ids, gold_triggers, gold_roles, pred_roles, gold_outputs, pred_outputs, inputs):
            outputs[pred_wnd_id] = {
                "input": input, 
                "triggers": gold_trigger,
                "gold_roles": gold_role,
                "pred_roles": pred_role,
                "gold_text": gold_output,
                "pred_text": pred_output,
            }

        with open(os.path.join(args.output_dir, f'{data_type}.pred.json'), 'w') as fp:
            json.dump(outputs, fp, indent=2)
elif args.type == "rank":
    # compute bleu
    def compute_bleu(can, ref):
        smoothie = SmoothingFunction().method2
        bleu_s = sentence_bleu([ref.split()], can.split(), weights=(1,0,0,0), smoothing_function=smoothie)
        return bleu_s
    
    # compute rouge score
    def compute_rouge(hyp, ref):
        score = all_scorer.score(ref, hyp)
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3

    all_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    # TODO: genearte train|dev|test files for ReRanking
    # in stage 1: DBS and return 4 triggers
    # in stage 2: DBS and return 4 candidates for each trigger -> in total 16 candidates
    for data_set, batch_num, data_type in zip([train_set, dev_set, test_set], [train_batch_num, dev_batch_num, test_batch_num], ['train', 'dev', 'test']):
        output_dir = args.output_dir + data_type
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        progress = tqdm.tqdm(total=batch_num, ncols=75, desc=data_type)
        gold_triggers, gold_roles, pred_roles = [], [], []
        pred_wnd_ids, gold_outputs, pred_outputs, inputs = [], [], [], []
        pred_trigs, pred_trig_event, gold_trigs, gold_trig_event = [], [], [], []
        count = 0
        ref_event, candidates, trig_candidates, content = [], [], [], []
        # only considering single event examples
        # content: with event_type and without event_type
        # candidates: <|triggerword|> triggerword <role> arg <role> arg # combine the prediction from TE and EE
        # ref_event: ground truth 

        # evaluate batch
        for batch in DataLoader(data_set, batch_size=config.eval_batch_size, shuffle=False, collate_fn=data_set.collate_fn):
            progress.update(1)
            # if count > 200: break
            batch_pred_roles = [[] for _ in range(config.eval_batch_size)]
            batch_pred_outputs = [[] for _ in range(config.eval_batch_size)]
            batch_gold_outputs = [[] for _ in range(config.eval_batch_size)]
            batch_inputs = [[] for _ in range(config.eval_batch_size)]
            batch_event_templates = []
            for tokens, triggers, roles in zip(batch.tokens, batch.triggers, batch.roles):
                batch_event_templates.append(event_template_generator(tokens, triggers, roles, config.input_style, config.output_style, vocab, config.lang))
            
            ## Stage1: Extract Trigger and Event_type
            # convert EE instances to EAE instances
            trig_inputs, trig_gold_outputs= [], []
            eae_inputs, eae_gold_outputs = [], []
            # create data inputs and output for trigger extraction
            for i, event_temp in enumerate(batch_event_templates):
                if args.single_only:
                    if len(event_temp.get_training_data()) > 1:
                        # only considering the single event examples
                        continue
                for data in event_temp.get_training_data():
                    trig_inputs.append("TriggerExtract: " + data[0].split(' <|triggerword|>')[0]) # + " [" +data[4].replace(":", "_")+ "]")
                    trig_gold_outputs.append('<|triggerword|> ' + data[0].split('<|triggerword|> ')[1].split(" <|template|")[0] + " [" +data[4].replace(":", "_")+ "]")

                    ## generate data for ranking
                    content.append(data[0].split(' <|triggerword|>')[0]) # without event_type
                    # content.append(data[0].split(' <|triggerword|>')[0] + " [" +data[4].replace(":", "_")+ "]") # with event_type

            # if there is triggers in this batch, predict triggerword and event type
            if len(trig_inputs) > 0:
                trig_inputs = tokenizer(trig_inputs, return_tensors='pt', padding=True, max_length=config.max_length+2)
                enc_idxs = trig_inputs['input_ids']
                enc_idxs = enc_idxs.cuda()
                enc_attn = trig_inputs['attention_mask'].cuda()

                if config.beam_size == 1:
                    model.model._cache_input_ids = enc_idxs
                else:
                    expanded_return_idx = (
                        torch.arange(enc_idxs.shape[0]).view(-1, 1).repeat(1, config.beam_size).view(-1).to(enc_idxs.device)
                    )
                    input_ids = enc_idxs.index_select(0, expanded_return_idx)
                    model.model._cache_input_ids = input_ids
                
                with torch.no_grad():
                    if args.constrained_decode:
                        prefix_fn_obj = Prefix_fn_cls(tokenizer, ["[and]"], enc_idxs)
                        outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn, 
                                num_beams=config.beam_size, 
                                max_length=config.max_output_length,
                                forced_bos_token_id=None,
                                prefix_allowed_tokens_fn=lambda batch_id, sent: prefix_fn_obj.get(batch_id, sent)
                                )
                    else:
                        # outputs= model.model.predict(batch, num_beams=config.beam_size, max_length=config.max_output_length)
                        outputs = model.model.generate(
                            input_ids=enc_idxs, 
                            attention_mask=enc_attn,
                            num_beams=config.beam_size, 
                            max_length=config.max_output_length,
                            forced_bos_token_id=None, 
                            num_return_sequences=args.num_return, 
                            num_beam_groups=args.beam_group, 
                            diversity_penalty=1.0) # diverse beam search
                
                trig_pred_outputs = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
                trig_pred_outputs = np.reshape(trig_pred_outputs, (len(trig_gold_outputs), -1))

                # extract triggerword and event type from the generated outputs
                for p_texts, g_text in zip(trig_pred_outputs, trig_gold_outputs):
                    # tag_ge = re.search('\[[^ />][^>]*\]', g_text)
                    # gold_event_type = tag_ge.group()[1:-1]
                    # gold_trigs.append(g_text[16:tag_ge.start()-1])
                    gold_trig_event.append(g_text[16:])
                    # ref_event: ground truth
                    ref_event.append(g_text)

                    cur_candidates = []
                    for i_p, p_text in enumerate(p_texts):
                        # if not p_text.startswith("<|triggerword|>"):
                        #     print("\n not start with <|triggerword|> :", p_text)
                        #     cur_candidates.append(cur_candidates[-1])
                        #     continue
                        all_words = re.findall("[^ <|\[>]\w*-*\w*[^ >|\]]", p_text)

                        tag_all = re.search("\<\|triggerword\|\> \w* \[[^ />][^>]*\]", p_text)
                        tag_trig = re.search("\<\|triggerword\|\> \w* ", p_text)
                        tag_pe = re.search('\[[^ />][^>]*\]', p_text)
                        tag_trigpe = re.search("\w* \[[^ />][^>]*\]",p_text)
                        
                        if tag_all:
                            p_text = tag_all.group()
                        elif tag_trig:
                            if tag_pe:
                                p_text = tag_trig.group() + tag_pe.group()
                            else:
                                idx_ind = all_words.index("triggerword")
                                if len(all_words) > idx_ind+2:
                                    p_text = tag_trig.group() + " [" + all_words[idx_ind+2] + "]"
                                    p_text.replace("[[", "[").replace("]]","]")
                                else:
                                    p_text = tag_trig.group() + " [None]"
                        elif tag_trigpe:
                            p_text = "<|triggerword|> " + tag_trigpe.group()
                        elif tag_pe:
                            if "triggerword" in p_text:
                                idx_ind = all_words.index("triggerword")
                                if len(all_words) > idx_ind +1:
                                    p_text = "<|triggerword|> " + all_words[idx_ind+1] + " " + tag_pe.group()
                                elif len(all_words) > 1:
                                    p_text = "<|triggerword|> " + all_words[idx_ind-1] + " " + tag_pe.group()
                            else:
                                if len(all_words) > 0:
                                    p_text = "<|triggerword|> " + all_words[0] + " " + tag_pe.group()
                                else:
                                    p_text = "<|triggerword|> None " + tag_pe.group()
                        else:
                            if "triggerword" in p_text:
                                idx_ind = all_words.index("triggerword")
                                if len(all_words) > idx_ind+2:
                                    p_text = "<|triggerword|> " + all_words[idx_ind+1] + " [" + all_words[idx_ind+2] + "]"
                                elif len(all_words) > idx_ind+1:
                                    p_text = "<|triggerword|> " + all_words[idx_ind+1] + " [None]"
                                elif len(all_words) > 1:
                                    p_text = "<|triggerword|> " + all_words[idx_ind-1] + " [None]"
                                else:
                                    p_text = "<|triggerword|> None [None]"
                            else:
                                if len(all_words) > 1:
                                    p_text = "<|triggerword|> " + all_words[0] + " [" + all_words[1] + "]"
                                elif len(all_words) > 0:
                                    p_text = "<|triggerword|> " + all_words[0] + " [None]"
                                else:
                                    p_text = "<|triggerword|> None [None]"
                        p_text = p_text.replace("/", "")
                        # if not tag_pe:
                        #     cur_candidates.append(p_text + " [None]")
                        #     continue
                        # pred_event_type = tag_pe.group()[1:-1]
                        # pred_trigs.append(p_text[16:tag_pe.start()-1])
                        pred_trig_event.append(p_text[16:])
                        cur_candidates.append(p_text)
                    trig_candidates.append(cur_candidates)
                
                # print(f"shape of ref_event: {np.shape(ref_event)}, shape of trig_candidates: {np.shape(trig_candidates)}, shape of content: {np.shape(content)}")
            assert np.shape(ref_event)[0] == np.shape(trig_candidates)[0] == np.shape(content)[0]
            # create data inputs and output for argument extraction
            # use the pred_trigs generated in the first stage to construct the EE input
            find_idx = []
            for i, event_temp in enumerate(batch_event_templates):
                if args.single_only:
                    if len(event_temp.get_training_data()) > 1:
                        # only considering the single event examples
                        continue
                for data in event_temp.get_training_data():
                    for beam_i in range(args.num_return):
                        # try:
                        tag_pe = re.search('\[[^ />][^>]*\]', trig_candidates[count][beam_i])
                        if not tag_pe:
                            print('\n', trig_candidates[count][beam_i],"\n", '-'*50)
                        eae_inputs.append("EventExtract: " + data[0].split(' <|triggerword|>')[0] + ' ' + trig_candidates[count][beam_i][:tag_pe.start()-1])
                        # except:
                        #     print(tag_pe, "num trig_candidates[count]", len(trig_candidates[count]))
                        #     [print( "num trig_candidates[count][beam_i]", len(trig_candidates[count][bi])) for bi in range(args.num_return)]
                        #     print(f"shape of ref_event: {np.shape(ref_event)}, shape of trig_candidates: {np.shape(trig_candidates)}, shape of content: {np.shape(content)}")
                        #     print(f"count: {count}, beam_i: {beam_i}")
                        #     print(eae_inputs[-1])
                    eae_gold_outputs.append(data[1])
                    ref_event[count] = ref_event[count] + " " + data[1]
                    find_idx.append(count)
                    count += 1
            # print("shape eae_inputs: ", np.shape(eae_inputs))
            eae_inputs_all = np.reshape(eae_inputs, (-1, args.num_return)).tolist()
            # if there are triggers in this batch, predict argument roles
            if len(eae_inputs_all) > 0:
                # each eae_inputs is the one event with four trigger beams
                for num_e, eae_inputs in enumerate(eae_inputs_all):
                    eae_inputs = tokenizer(eae_inputs, return_tensors='pt', padding=True, max_length=config.max_length+2)
                    enc_idxs = eae_inputs['input_ids']
                    enc_idxs = enc_idxs.cuda()
                    enc_attn = eae_inputs['attention_mask'].cuda()

                    if config.beam_size == 1:
                        model.model._cache_input_ids = enc_idxs
                    else:
                        expanded_return_idx = (
                            torch.arange(enc_idxs.shape[0]).view(-1, 1).repeat(1, config.beam_size).view(-1).to(enc_idxs.device)
                        )
                        input_ids = enc_idxs.index_select(0, expanded_return_idx)
                        model.model._cache_input_ids = input_ids
                    
                    # inference
                    with torch.no_grad():
                        if args.constrained_decode:
                            prefix_fn_obj = Prefix_fn_cls(tokenizer, ["[and]"], enc_idxs)
                            outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn, 
                                    num_beams=config.beam_size, 
                                    max_length=config.max_output_length,
                                    forced_bos_token_id=None,
                                    prefix_allowed_tokens_fn=lambda batch_id, sent: prefix_fn_obj.get(batch_id, sent)
                                    )
                        else:
                            # outputs= model.model.predict(batch, num_beams=config.beam_size, max_length=config.max_output_length)
                            outputs = model.model.generate(
                                input_ids=enc_idxs,
                                attention_mask=enc_attn,
                                num_beams=config.beam_size, 
                                max_length=config.max_output_length,
                                forced_bos_token_id=None, 
                                num_return_sequences=args.num_return, 
                                num_beam_groups=args.beam_group, 
                                diversity_penalty=1.0) # diverse beam search

                    eae_pred_outputs = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
                    eae_pred_outputs = np.reshape(eae_pred_outputs, (args.num_return, -1)) # 4*4
                    cur_candidates = []
                    for trig_i, trig_c in enumerate(trig_candidates[find_idx[num_e]]):
                        for eae_trig in eae_pred_outputs[trig_i]:
                            cur_candidates.append(trig_c + " " + eae_trig)
                    
                    # cur_candidates = [(x, compute_rouge(x, ref_event[find_idx[num_e]])) for x in cur_candidates]
                    cur_candidates = [(x, compute_bleu(x, ref_event[find_idx[num_e]])) for x in cur_candidates]
                    candidates.append(cur_candidates)
                # print(f"shape of ref_event: {np.shape(ref_event)}, shape of candidates: {np.shape(candidates)}, shape of content: {np.shape(content)}")
                assert np.shape(ref_event)[0] == np.shape(candidates)[0] == np.shape(content)[0]

        count = 0
        for cn_i, r_i, can_i in zip(content, ref_event, candidates):
            output = {
                "article": cn_i, 
                "abstract": r_i,
                "candidates": can_i,
                }
            # write outputs
            with open(os.path.join(output_dir, f'{count}.json'), 'w') as fp:
                json.dump(output, fp, indent=2)
            count += 1
        print("num sample: ", count)

elif args.type == "ranktrig":
    #TODO: work save multiple triggers and rank on triggers
    # compute bleu
    def compute_bleu(can, ref):
        smoothie = SmoothingFunction().method2
        bleu_s = sentence_bleu([ref.split()], can.split(), weights=(1,0,0,0), smoothing_function=smoothie)
        return bleu_s
    
    # compute rouge score
    def compute_rouge(hyp, ref):
        score = all_scorer.score(ref, hyp)
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3

    all_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    # TODO: genearte train|dev|test files for ReRanking
    # in stage 1: DBS and return 4 triggers
    # in stage 2: DBS and return 4 candidates for each trigger -> in total 16 candidates
    for data_set, batch_num, data_type in zip([train_set, dev_set, test_set], [train_batch_num, dev_batch_num, test_batch_num], ['train', 'dev', 'test']):
        output_dir = args.output_dir + data_type
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        progress = tqdm.tqdm(total=batch_num, ncols=75, desc=data_type)
        gold_triggers, gold_roles, pred_roles = [], [], []
        pred_wnd_ids, gold_outputs, pred_outputs, inputs = [], [], [], []
        pred_trigs, pred_trig_event, gold_trigs, gold_trig_event = [], [], [], []
        count = 0
        ref_event, candidates, trig_candidates, content = [], [], [], []
        # only considering single event examples
        # content: with event_type and without event_type
        # candidates: <|triggerword|> triggerword <role> arg <role> arg # combine the prediction from TE and EE
        # ref_event: ground truth 

        # evaluate batch
        for batch in DataLoader(data_set, batch_size=config.eval_batch_size, shuffle=False, collate_fn=data_set.collate_fn):
            progress.update(1)
            # if count > 10: break
            batch_pred_roles = [[] for _ in range(config.eval_batch_size)]
            batch_pred_outputs = [[] for _ in range(config.eval_batch_size)]
            batch_gold_outputs = [[] for _ in range(config.eval_batch_size)]
            batch_inputs = [[] for _ in range(config.eval_batch_size)]
            batch_event_templates = []
            for tokens, triggers, roles in zip(batch.tokens, batch.triggers, batch.roles):
                batch_event_templates.append(event_template_generator(tokens, triggers, roles, config.input_style, config.output_style, vocab, config.lang))
            
            ## Stage1: Extract Trigger and Event_type
            # convert EE instances to EAE instances
            trig_inputs, trig_gold_outputs= [], []
            eae_inputs, eae_gold_outputs = [], []
            # create data inputs and output for trigger extraction
            for i, event_temp in enumerate(batch_event_templates):
                if args.single_only:
                    if len(event_temp.get_training_data()) > 1:
                        # only considering the single event examples
                        continue
                for data in event_temp.get_training_data():
                    trig_inputs.append("TriggerExtract: " + data[0].split(' <|triggerword|>')[0]) # + " [" +data[4].replace(":", "_")+ "]")
                    trig_gold_outputs.append('<|triggerword|> ' + data[0].split('<|triggerword|> ')[1].split(" <|template|")[0] + " [" +data[4].replace(":", "_")+ "]")

                    ## generate data for ranking
                    content.append(data[0].split(' <|triggerword|>')[0]) # without event_type
                    # content.append(data[0].split(' <|triggerword|>')[0] + " [" +data[4].replace(":", "_")+ "]") # with event_type
                    count += 1
            
            # if there is triggers in this batch, predict triggerword and event type
            if len(trig_inputs) > 0:
                trig_inputs = tokenizer(trig_inputs, return_tensors='pt', padding=True, max_length=config.max_length+2)
                enc_idxs = trig_inputs['input_ids']
                enc_idxs = enc_idxs.cuda()
                enc_attn = trig_inputs['attention_mask'].cuda()

                if config.beam_size == 1:
                    model.model._cache_input_ids = enc_idxs
                else:
                    expanded_return_idx = (
                        torch.arange(enc_idxs.shape[0]).view(-1, 1).repeat(1, config.beam_size).view(-1).to(enc_idxs.device)
                    )
                    input_ids = enc_idxs.index_select(0, expanded_return_idx)
                    model.model._cache_input_ids = input_ids
                
                with torch.no_grad():
                    if args.constrained_decode:
                        prefix_fn_obj = Prefix_fn_cls(tokenizer, ["[and]"], enc_idxs)
                        outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn, 
                                num_beams=config.beam_size, 
                                max_length=config.max_output_length,
                                forced_bos_token_id=None,
                                prefix_allowed_tokens_fn=lambda batch_id, sent: prefix_fn_obj.get(batch_id, sent)
                                )
                    else:
                        # outputs= model.model.predict(batch, num_beams=config.beam_size, max_length=config.max_output_length)
                        outputs = model.model.generate(
                            input_ids=enc_idxs, 
                            attention_mask=enc_attn,
                            num_beams=config.beam_size, 
                            max_length=config.max_output_length,
                            forced_bos_token_id=None, 
                            num_return_sequences=args.num_return, 
                            num_beam_groups=args.beam_group, 
                            diversity_penalty=1.0) # diverse beam search
                
                trig_pred_outputs = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
                trig_pred_outputs = np.reshape(trig_pred_outputs, (len(trig_gold_outputs), -1)).tolist()

                # extract triggerword and event type from the generated outputs
                for p_texts, g_text in zip(trig_pred_outputs, trig_gold_outputs):
                    # tag_ge = re.search('\[[^ />][^>]*\]', g_text)
                    # gold_event_type = tag_ge.group()[1:-1]
                    # gold_trigs.append(g_text[16:tag_ge.start()-1])
                    gold_trig_event.append(g_text[16:])
                    # ref_event: ground truth
                    ref_event.append(g_text)
                    p_texts = [(x, compute_bleu(x, g_text)) for x in p_texts]
                    candidates.append(p_texts)

                    # cur_candidates = []
                    # for i_p, p_text in enumerate(p_texts):
                    #     # if not p_text.startswith("<|triggerword|>"):
                    #     #     print("\n not start with <|triggerword|> :", p_text)
                    #     #     cur_candidates.append(cur_candidates[-1])
                    #     #     continue
                    #     all_words = re.findall("[^ <|\[>]\w*-*\w*[^ >|\]]", p_text)

                    #     tag_all = re.search("\<\|triggerword\|\> \w* \[[^ />][^>]*\]", p_text)
                    #     tag_trig = re.search("\<\|triggerword\|\> \w* ", p_text)
                    #     tag_pe = re.search('\[[^ />][^>]*\]', p_text)
                    #     tag_trigpe = re.search("\w* \[[^ />][^>]*\]",p_text)
                        
                    #     if tag_all:
                    #         p_text = tag_all.group()
                    #     elif tag_trig:
                    #         if tag_pe:
                    #             p_text = tag_trig.group() + tag_pe.group()
                    #         else:
                    #             idx_ind = all_words.index("triggerword")
                    #             if len(all_words) > idx_ind+2:
                    #                 p_text = tag_trig.group() + " [" + all_words[idx_ind+2] + "]"
                    #                 p_text.replace("[[", "[").replace("]]","]")
                    #             else:
                    #                 p_text = tag_trig.group() + " [None]"
                    #     elif tag_trigpe:
                    #         p_text = "<|triggerword|> " + tag_trigpe.group()
                    #     elif tag_pe:
                    #         if "triggerword" in p_text:
                    #             idx_ind = all_words.index("triggerword")
                    #             if len(all_words) > idx_ind +1:
                    #                 p_text = "<|triggerword|> " + all_words[idx_ind+1] + " " + tag_pe.group()
                    #             elif len(all_words) > 1:
                    #                 p_text = "<|triggerword|> " + all_words[idx_ind-1] + " " + tag_pe.group()
                    #         else:
                    #             if len(all_words) > 0:
                    #                 p_text = "<|triggerword|> " + all_words[0] + " " + tag_pe.group()
                    #             else:
                    #                 p_text = "<|triggerword|> None " + tag_pe.group()
                    #     else:
                    #         if "triggerword" in p_text:
                    #             idx_ind = all_words.index("triggerword")
                    #             if len(all_words) > idx_ind+2:
                    #                 p_text = "<|triggerword|> " + all_words[idx_ind+1] + " [" + all_words[idx_ind+2] + "]"
                    #             elif len(all_words) > idx_ind+1:
                    #                 p_text = "<|triggerword|> " + all_words[idx_ind+1] + " [None]"
                    #             elif len(all_words) > 1:
                    #                 p_text = "<|triggerword|> " + all_words[idx_ind-1] + " [None]"
                    #             else:
                    #                 p_text = "<|triggerword|> None [None]"
                    #         else:
                    #             if len(all_words) > 1:
                    #                 p_text = "<|triggerword|> " + all_words[0] + " [" + all_words[1] + "]"
                    #             elif len(all_words) > 0:
                    #                 p_text = "<|triggerword|> " + all_words[0] + " [None]"
                    #             else:
                    #                 p_text = "<|triggerword|> None [None]"
                    #     p_text = p_text.replace("/", "")
                    #     # if not tag_pe:
                    #     #     cur_candidates.append(p_text + " [None]")
                    #     #     continue
                    #     # pred_event_type = tag_pe.group()[1:-1]
                    #     # pred_trigs.append(p_text[16:tag_pe.start()-1])
                    #     pred_trig_event.append(p_text[16:])
                    #     cur_candidates.append(p_text)
                    # trig_candidates.append(cur_candidates)
                
                # print(f"shape of ref_event: {np.shape(ref_event)}, shape of trig_candidates: {np.shape(trig_candidates)}, shape of content: {np.shape(content)}")
            assert np.shape(ref_event)[0] == np.shape(candidates)[0] == np.shape(content)[0]
            
        count = 0
        for cn_i, r_i, can_i in zip(content, ref_event, candidates):
            output = {
                "article": cn_i, 
                "abstract": r_i,
                "candidates": can_i,
                }
            # write outputs
            with open(os.path.join(output_dir, f'{count}.json'), 'w') as fp:
                json.dump(output, fp, indent=2)
            count += 1
        print("num sample: ", count)
else:
    print("############ wrong evaluation type #############")

