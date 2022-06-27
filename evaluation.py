import json
from copy import deepcopy
import re
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

result_folder = "./generated/"

class Metric:
    def __init__(self):
        self.tp = 0.
        self.gold_num = 0.
        self.pred_num = 0.

    @staticmethod
    def safe_div(a, b):
        if b == 0.:
            return 0.
        else:
            return a / b

    def compute_f1(self, prefix=''):
        tp = self.tp
        pred_num = self.pred_num
        gold_num = self.gold_num
        p, r = self.safe_div(tp, pred_num), self.safe_div(tp, gold_num)
        return {prefix + 'tp': tp,
                prefix + 'gold': gold_num,
                prefix + 'pred': pred_num,
                prefix + 'P': p * 100,
                prefix + 'R': r * 100,
                prefix + 'F1': self.safe_div(2 * p * r, p + r) * 100
                }

    def count_instance(self, gold_list, pred_list, verbose=False):
        if verbose:
            print("Gold:", gold_list)
            print("Pred:", pred_list)
        self.gold_num += len(gold_list)
        self.pred_num += len(pred_list)

        dup_gold_list = deepcopy(gold_list)
        for pred in pred_list:
            if pred in dup_gold_list:
                self.tp += 1
                dup_gold_list.remove(pred)


def load_data(file_path, data_type):
    file = json.load(open(file_path))
    single_file = [i for i in file if '<|sep|>' not in i['reference_text']]
    # gt_continuation = [i['reference_continuation_text'] for i in single_file if i['data_type']==data_type]
    # pred_continuation = [t_i['generated_result']["0"]['continuation'] for i, t_i in enumerate(single_file) if t_i['data_type']==data_type]
    gt_continuation = [i['reference_continuation_text'] for i in file if i['data_type']==data_type]
    pred_continuation = [t_i['generated_result']['0']['continuation'] for i, t_i in enumerate(file) if t_i['data_type']==data_type]
    return gt_continuation, pred_continuation


def compute_bleu(gt_continuation, pred_continuation):
    bleu_s = 0
    smoothie = SmoothingFunction().method2
    for ref, can in zip(pred_continuation, gt_continuation):
        bleu_s += sentence_bleu([ref.split()], can.split(), weights=(1,0,0,0), smoothing_function=smoothie)
    print(' bleu score: {}'.format(bleu_s/len(gt_continuation)))


def compute_f1(gt_continuation, pred_continuation):
    all_results = {}
    def get_tokens(out):
        # event type
        out = '>' + out
        # out = out.replace("<|endoftext|>", "")
        event_type = re.findall(r"\[(.*?)\]", out)
        # special tokens
        special_tokens = re.findall(r"<(.*?)>", out)
        # special_tokens = [special_tokens[i:i+2] for i in range(0, len(special_tokens)-1, 2)]
        # word in between
        extracted = re.findall(r"> (.*?) <", out)
        extracted = [i[:] for i in extracted if (i != '' and i != ' ')]
        return special_tokens, extracted, event_type

    def update(to_update, keys):
        name1, name2, name3, name4 = keys
        # try:
        #     to_update[name1].append(extracted[0].split(" ")[0])
        #     to_update[name2].append({extracted[0].split(" ")[1]: extracted[0].split(" ")[0]})
        # except:
        #     pass
        #     # print(keys, extracted)
        
        to_update[name1].append('a')
        to_update[name2].append('a')
        to_update[name3].append([])
        to_update[name4].append([])


        if 'pred' in name1:
            if len(special_tokens) != len(extracted):
                if len(to_update['format']) == 0:
                    to_update['format'].append(False)
                else:
                    to_update['format'][-1] = False
        elif 'gt' in name1:
            to_update['format'].append(True)

        for role_pair, tokens in zip(special_tokens[:], extracted[:]):
            all_args = tokens.split('[and]')
            if all_args == ['[None]']:
                continue
            to_update[name3][-1] += all_args
            # if role_pair[0][1:-1] == role_pair[1][6:-1]:
            #     for j in all_args:
            #         to_update[name4][-1].append({"role": role_pair[0][1:-1], "arg": j})
            # else:
            #     for j in all_args:
            #         to_update[name4][-1].append({"role": role_pair[0][1:-1], "arg": j})
            #         to_update[name4][-1].append({"role": role_pair[1][6:-1], "arg": j})
            for j in all_args:
                to_update[name4][-1].append({"role": role_pair[:], "arg": j})
        return to_update

    index = 0
    for gt_i, pred_i in zip(gt_continuation, pred_continuation):
        # gt_triggers = [i['reference_continuation_text'].split("<|endoftrigger|>")[0].split(" ")[2] for i in file]
        # gt_event = [i['reference_continuation_text'].split("<|endoftrigger|>")[0].split(" ")[2] for i in file]
        all_results[index] = []
        ### for ground truth
        result_i = {'format': [],
                   'gt_triggerwords': [], 'gt_trigger_event': [], 'gt_arguments': [], 'gt_arg_role': [],
                   'pred_triggerwords': [], 'pred_trigger_event': [], 'pred_arguments': [], 'pred_arg_role': []}
        
        gt_i = gt_i.replace(" <|endoftext|>", "")
        gt_i_e = gt_i.split(' <|sep|>')
        for i in gt_i_e:
            special_tokens, extracted, _ = get_tokens(i)
            if extracted:
                result_i = update(result_i, list(result_i.keys())[1:5])

        ### for generated continuation
        pred_i = pred_i.replace(" <|endoftext|>", "")
        pred_i_e = pred_i.split(' <|sep|>')
        for i in pred_i_e:
            special_tokens, extracted, _ = get_tokens(i)
            if extracted:
                result_i = update(result_i, list(result_i.keys())[5:9])

        all_results[index] = result_i
        index += 1
    return all_results

# use seqeval to compute F1, BLEU, ROUGE


def compute_trigger_f1(results, data_type):
    trig_idf_metric = Metric()
    trig_cls_metric = Metric()
    for _, d_i in results.items():
        # for gt_i, pred_i in zip(d_i['gt_triggerwords'], d_i['pred_triggerwords']):
        trig_idf_metric.count_instance(d_i['gt_triggerwords'], d_i['pred_triggerwords'])

    # for classification, both the word it self and its role should match with the ground truth
    for _, d_i in results.items():
        # for gt_i, pred_i in zip(d_i['gt_trigger_event'], d_i['pred_trigger_event']):
        trig_cls_metric.count_instance(d_i['gt_trigger_event'], d_i['pred_trigger_event'])

    trig_idf_score = trig_idf_metric.compute_f1(prefix="trig-idf-")
    trig_cls_score = trig_cls_metric.compute_f1(prefix="trig-cls-")
    print(data_type, " -> trigger identification F1: {}, trigger classification F1: {}".format(trig_idf_score['trig-idf-F1'], trig_cls_score['trig-cls-F1']))
    # print(data_type, " -> trigger identification R: {}, trigger classification R: {}".format(trig_idf_score['trig-idf-R'], trig_cls_score['trig-cls-R']))
    # print(data_type, " -> trigger identification P: {}, trigger classification P: {}".format(trig_idf_score['trig-idf-P'], trig_cls_score['trig-cls-P']))
    return trig_idf_score['trig-idf-F1'], trig_cls_score['trig-cls-F1']


def compute_argument_f1(results, data_type):
    arg_idf_metric = Metric()
    arg_cls_metric = Metric()
    for _, d_i in results.items():
        for gt_i, pred_i in zip(d_i['gt_arguments'], d_i['pred_arguments']):
            arg_idf_metric.count_instance(gt_i, pred_i)

    # for classification, both the word it self and its role should match with the ground truth
    for _, d_i in results.items():
        for gt_i, pred_i in zip(d_i['gt_arg_role'], d_i['pred_arg_role']):
            arg_cls_metric.count_instance(gt_i, pred_i)

    arg_idf_score = arg_idf_metric.compute_f1(prefix="arg-idf-")
    arg_cls_score = arg_cls_metric.compute_f1(prefix="arg-cls-")
    print(data_type, " -> argument identification F1: {}, role classification F1: {}".format(arg_idf_score['arg-idf-F1'], arg_cls_score['arg-cls-F1']))
    return arg_idf_score['arg-idf-F1'], arg_cls_score['arg-cls-F1']


def main(filepath, data_type=''):
    gt_data, pred_data = load_data(filepath, data_type)
    rouge = Rouge()
    rouge_results = rouge.get_scores(pred_data, gt_data, avg=True)
    # print(rouge_results)
    # compute_bleu(gt_data, pred_data)
    all_results = compute_f1(gt_data, pred_data)
    # conpute trigger related F1
    # print('correct format {} over {}'.format(sum(all_results['format']), len(all_results['gt_triggerwords'])))
    compute_trigger_f1(all_results, data_type)
    compute_argument_f1(all_results, data_type)
    print('-'*50)
    return


if __name__ == "__main__":
    # files = ['ace_all_tod_cor_shifted_beam.json',
    #          'ace_all_tod_cor_shifted_greedy.json',
    #          'ace_all_tod_cor_shifted_top_k.json',
    #          'ace_all_tod_cor_shifted_top_p.json']
    files = ['ace_all_tod_cor_shifted_eval_greedy.json']
             #'ace_all_tod_cor_shifted_eval_greedy.json']
             # 'ace_all_tod_cor_shifted_eval_top_k.json']
    for file in files:
        print(file)
        main(result_folder + file, 'dev')
        main(result_folder + file, 'test')
    # main(result_folder+"ace_single_tod_cor_shifted_special_new.json")
    # main(result_folder+"ace_single_tod_cor_shifted_lr1_w2.json")
    # main(result_folder+"ace_single_tod_cor_shifted_special_oracle.json")