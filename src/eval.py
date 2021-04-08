import json
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import RobertaTokenizerFast, PreTrainedTokenizerFast

from src import utils as utils


def eval_bert(model, data_loader, device, eval_file, max_len, use_squad_v2):
    ce_meter = utils.AverageMeter()

    # get tokenizer to decode (pred) indices of Wordpiece tokens
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json.load(fh)
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for q_c_ids, attn_mask, y1, y2, ids in data_loader:
            # Setup for forward
            q_c_ids = q_c_ids.to(device)
            attn_mask = attn_mask.to(device)

            batch_size = q_c_ids.size(0)

            # Forward
            start_logits, end_logits = model(q_c_ids, attn_mask)
            y1, y2 = y1.to(device), y2.to(device)
            # ignore indices outside of model input
            ignored_index = start_logits.size(1)
            y1.clamp_(0, ignored_index)
            y2.clamp_(0, ignored_index)

            loss_fnc = nn.CrossEntropyLoss(ignore_index=ignored_index)
            loss = loss_fnc(start_logits, y1) + loss_fnc(end_logits, y2)
            ce_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            softmax_fnc = nn.Softmax(dim=1)
            p1, p2 = softmax_fnc(start_logits), softmax_fnc(end_logits)
            starts, ends = utils.discretize(p1, p2, max_len, use_squad_v2) # get start-end pair with max joint probability

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(CE=ce_meter.avg)

            # map predicted start-end indices to tokens (from context)
            preds = utils.convert_bert_tokens(tokenizer,
                                                 q_c_ids.tolist(),
                                                 ids.tolist(),
                                                 starts.tolist(),
                                                 ends.tolist(),
                                                 use_squad_v2)
            pred_dict.update(preds)
            #
            # if len(pred_dict) > 100:
            #     break

    model.train()

    results = utils.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [('CE', ce_meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)


    return results, pred_dict


def eval_bidaf(model, data_loader, device, eval_file, max_len, use_squad_v2):
    nll_meter = utils.AverageMeter()

    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json.load(fh)
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            log_p1, log_p2 = model(cw_idxs, qw_idxs) # return log-softmax
            y1, y2 = y1.to(device), y2.to(device)

            loss_fnc = nn.NLLLoss()
            loss = loss_fnc(log_p1, y1) + loss_fnc(log_p2, y2)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = utils.discretize(p1, p2, max_len, use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

            preds, _ = utils.convert_tokens(gold_dict,
                                                 ids.tolist(),
                                                 starts.tolist(),
                                                 ends.tolist(),
                                                 use_squad_v2)
            pred_dict.update(preds)

    model.train()

    results = utils.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [('NLL', nll_meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, pred_dict