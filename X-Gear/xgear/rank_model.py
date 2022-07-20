# modified from https://github.com/maszhongming/MatchSum
import torch
from torch import nn
from transformers import RobertaModel
import os
# DBS: num_return_sequences=16, num_beam_groups=16, diversity_penalty=1.0, num_beams=16
# Transformers 4.1.1 and PyTorch 1.4

def RankingLoss(score, summary_score=None, margin=0, gold_margin=0, gold_weight=1, no_gold=False, no_cand=False):
    ones = torch.ones_like(score)
    loss_func = torch.nn.MarginRankingLoss(0.0)
    TotalLoss = loss_func(score, score, ones)
    # candidate loss
    n = score.size(1)
    if not no_cand:
        # a summation step that corresponds to a specific value of i in the equation is only finished when the loop in the code is completed.
        for i in range(1, n):
            # scores in descending order
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            # i serves the same purpose as j-i in the equation listed in the paper.
            loss_func = torch.nn.MarginRankingLoss(margin * i) # shouldn't use the reduction=sum?
            # If y=1 then it assumed the first input should be ranked higher (have a larger value) than the second input, and vice-versa 

            # The loss function for each pair of samples in the mini-batch is:
            # loss(x1,x2,y)=max(0,−y∗(x1−x2)+margin)

            # in the paper, it is negative minus positive, s1...sn is ranking in descending order, sj-si where j>i
            # if y is ones, the equation loss(x1, x2, y) = max(0, x2-x1+margin)
            # x2 should be negatives, x1 should be positives
            
            # if scores = [8,5,3,2,1]
            # i = 1, loss([8,5,3,2], [5,3,2,1], ones), margin = 1*lambda
            # i = 2, loss([8,5,3], [3,2,1], ones), margin = 2*lambda
            # i = 3, loss([8,5], [2,1], ones), margin = 3*lambda
            # i = 4, loss([8], [1], ones), margin = 4*lambda

            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss
    if no_gold:
        return TotalLoss
    # gold summary loss
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)
    return TotalLoss


class ReRanker(nn.Module):
    def __init__(self, encoder, pad_token_id):
        super(ReRanker, self).__init__()
        self.encoder = RobertaModel.from_pretrained(encoder)
        self.pad_token_id = pad_token_id

    def forward(self, text_id, candidate_id, summary_id=None, require_gold=True):
        
        batch_size = text_id.size(0)
        
        input_mask = text_id != self.pad_token_id
        out = self.encoder(text_id, attention_mask=input_mask)[0]
        doc_emb = out[:, 0, :]
        
        if require_gold:
            # get reference score
            input_mask = summary_id != self.pad_token_id
            out = self.encoder(summary_id, attention_mask=input_mask)[0]
            summary_emb = out[:, 0, :]
            summary_score = torch.cosine_similarity(summary_emb, doc_emb, dim=-1)

        candidate_num = candidate_id.size(1)
        candidate_id = candidate_id.view(-1, candidate_id.size(-1))
        input_mask = candidate_id != self.pad_token_id
        out = self.encoder(candidate_id, attention_mask=input_mask)[0]
        candidate_emb = out[:, 0, :].view(batch_size, candidate_num, -1)
        
        # get candidate score
        doc_emb = doc_emb.unsqueeze(1).expand_as(candidate_emb)
        score = torch.cosine_similarity(candidate_emb, doc_emb, dim=-1)

        output = {'score': score}
        if require_gold:
            output['summary_score'] = summary_score
        return output