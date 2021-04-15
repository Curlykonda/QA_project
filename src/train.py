import os
from pathlib import Path

import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from json import dumps
import random

from tqdm import tqdm

import src.utils as utils
from src.eval import eval_bert, eval_bidaf
from src.options import *

from src.modules.models import *
from src.modules.utils import *
from src.datasets.squad import SQuAD, squad_collate_fn, SquadBERT


def train_bert(args):
    device, log, tbx, saver = setup_train(args)

    # word embeddings

    log.info('Using pt Roberta embeddings')
    # else:
    #     log.info('Loading some other embeddings...')

    # build model
    log.info('Building model...')
    # args.freeze_bert_encoder, args.freeze_we_embs
    model = RobertaQA()
    model = model.to(device)
    model.train()
    log.info(f'Trainable params: {utils.count_model_params(model)}')
    log.info(f'Est. memory for model: {utils.estimate_model_memory(model)}')

    optimizer, scheduler, ema = get_optim_schedule(args, model)
    # criterion = nn.CrossEntropyLoss()
    # get dataloader
    train_loader, dev_loader = get_dataloader(args, log)
    dev_eval_file = utils.get_file_path(args.data_root, args.dataset_name, args.dev_eval_file)

    # Training loop
    step = 0
    steps_till_eval = args.eval_steps
    epoch = step // len(train_loader)

    # evaluate at step 0
    log.info(f'Evaluating at step {step}...')

    results, pred_dict = eval_bert(model, dev_loader, device,
                                   dev_eval_file,
                                   args.max_ans_len,
                                   args.use_squad_v2)

    # Log to console
    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
    log.info(f'Dev {results_str}')

    log_to_tbx(tbx, dev_eval_file, log, pred_dict, results, step, args.num_visuals)

    log.info('Training...')
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
             tqdm(total=len(train_loader)) as progress_bar:
            for q_c_ids, attn_mask, y1, y2, ids in train_loader:
                # Setup for forward
                q_c_ids = q_c_ids.to(device)  # context-question-pair as word IDs
                attn_mask = attn_mask.to(device)

                batch_size = q_c_ids.size(0)
                optimizer.zero_grad()

                # Forward
                start_logits, end_logits = model(q_c_ids, attn_mask)
                y1, y2 = y1.to(device), y2.to(device)

                # sometimes the start/end positions lie outside of model inputs -> ignore these terms
                ignored_index = start_logits.size(1)
                y1.clamp_(0, ignored_index)
                y2.clamp_(0, ignored_index)

                criterion = nn.CrossEntropyLoss(ignore_index=ignored_index)
                loss = criterion(start_logits, y1) + criterion(end_logits, y2)
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         CE=loss_val)
                tbx.add_scalar('train/CE', loss_val, step)
                tbx.add_scalar('train/LR', optimizer.param_groups[0]['lr'], step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    # log internal or first iteration
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    results, pred_dict = eval_bert(model, dev_loader, device,
                                                   dev_eval_file,
                                                   args.max_ans_len,
                                                   args.use_squad_v2)

                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    log_to_tbx(tbx, dev_eval_file, log, pred_dict, results, step, args.num_visuals)


def log_to_tbx(tbx: SummaryWriter, dev_eval_file, log, pred_dict, results: dict, step, n_visuals):
    # Log to TensorBoard
    log.info('Visualizing in TensorBoard...')
    for k, v in results.items():
        tbx.add_scalar(f'dev/{k}', v, step)
    utils.visualize(tbx,
                    pred_dict=pred_dict,
                    eval_path=dev_eval_file,
                    step=step,
                    split='dev',
                    num_visuals=n_visuals)


def get_dataloader(args, log):
    """
    Create (torch.utils.data.Dataloader) from preprocessed train and dev data
    """

    log.info('Building dataset...')
    data_path = Path(args.data_root)

    if args.use_roberta_token:

        train_dataset = SquadBERT(data_path.joinpath(args.dataset_name, 'roberta_' + args.train_record_file),
                                  args.use_squad_v2)

        dev_dataset = SquadBERT(data_path.joinpath(args.dataset_name, 'roberta_' + args.dev_record_file),
                                args.use_squad_v2)

        col_fnc = None
    else:

        train_dataset = SQuAD(data_path.joinpath(args.dataset_name, args.train_record_file),
                              args.use_squad_v2)
        dev_dataset = SQuAD(data_path.joinpath(args.dataset_name, args.dev_record_file),
                            args.use_squad_v2)

        col_fnc = squad_collate_fn

    if args.debug:
        train_dataset.shrink_dataset(1000)
        dev_dataset.shrink_dataset(100)

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=col_fnc)

    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=col_fnc)

    return train_loader, dev_loader


def get_optim_schedule(args, model):
    if args.use_ema:
        ema = EMA(model, args.ema_decay)  # Exponential Moving Average (over model parameters)
    else:
        ema = None

    # Get optimizer and scheduler
    if "adadelta" == args.optim.lower():
        optimizer = optim.Adadelta(model.parameters(), args.lr,
                                   weight_decay=args.l2_wd)
    elif "adam" == args.optim.lower():
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.l2_wd)
    else:
        raise ValueError()

    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    return optimizer, scheduler, ema


def setup_train(args):
    # Set up logging and devices
    args.save_dir = utils.get_save_dir(args.save_dir, args.model_name, training=True)
    log = utils.get_logger(args.save_dir, args.model_name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = utils.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)  # rnd = random.Random(seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False  # no real need for reproducibility atm
    torch.backends.cudnn.deterministic = True

    # Get saver
    saver = utils.CheckpointSaver(args.save_dir,
                                  max_checkpoints=args.max_checkpoints,
                                  metric_name=args.metric_name,
                                  maximize_metric=args.maximize_metric,
                                  log=log)

    return device, log, tbx, saver


def train_bidaf(args):
    device, log, tbx, saver = setup_train(args)

    # Get embeddings
    log.info('Loading embeddings...')

    word_vectors = utils.torch_from_json(os.path.join(args.data_root, args.dataset_name, args.word_emb_file))
    # char_vectors = utils.torch_from_json()

    # Get model
    log.info('Building model...')

    # select model
    if "bidaf" == args.model_name.lower():
        model = BiDAF(args, word_vectors=word_vectors,
                      char_vectors=None,
                      p_drop=args.drop_prob)
    else:
        raise ValueError()

    log.info(f'Trainable params: {utils.count_model_params(model)}')
    log.info(f'Est. memory for model: {utils.estimate_model_memory(model)}')
    # model = nn.DataParallel(model, args.gpu_ids)

    # load checkpoint
    # if args.load_path:
    #     log.info(f'Loading checkpoint from {args.load_path}...')
    #     model, step = utils.load_model(model, args.load_path, args.gpu_ids)
    # else:
    #     step = 0
    step = 0

    model = model.to(device)
    model.train()
    optimizer, scheduler, ema = get_optim_schedule(args, model)
    criterion = nn.NLLLoss()
    # Prep Dataloader
    train_loader, dev_loader = get_dataloader(args, log)

    # Training loop
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_loader.dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
             tqdm(total=len(train_loader.dataset)) as progress_bar:
            for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
                # Setup for forward
                cw_idxs = cw_idxs.to(device)
                qw_idxs = qw_idxs.to(device)
                batch_size = cw_idxs.size(0)
                optimizer.zero_grad()

                # Forward
                log_p1, log_p2 = model(cw_idxs, qw_idxs)
                y1, y2 = y1.to(device), y2.to(device)
                loss = criterion(log_p1, y1) + criterion(log_p2, y2)
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # step // batch_size
                ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                tbx.add_scalar('train/NLL', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    dev_eval_file = utils.get_file_path(args.data_root, args.dataset_name, args.dev_eval_file)
                    results, pred_dict = eval_bidaf(model, dev_loader, device,
                                                    dev_eval_file,
                                                    args.max_ans_len,
                                                    args.use_squad_v2)
                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)
                    utils.visualize(tbx,
                                    pred_dict=pred_dict,
                                    eval_path=dev_eval_file,
                                    step=step,
                                    split='dev',
                                    num_visuals=args.num_visuals)


if __name__ == '__main__':
    args = get_train_args()

    if 'bidaf' == args.model_name:
        train_bidaf(args)
    elif 'roberta-qa' == args.model_name:
        train_bert(args)
    else:
        raise ValueError(args.model_name)
