import torch
import torch.utils.data as data
import numpy as np
import json


from src.datasets.squad import SQuAD, squad_collate_fn

def main(args):

    # load stuff


    # get dataloader
    train_dataset = SQuAD(args.train_record_file, args.use_squad_v2)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=squad_collate_fn)
    dev_dataset = SQuAD(args.dev_record_file, args.use_squad_v2)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=squad_collate_fn)

    return



# for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
# cw_idxs : context word indices (?)
# cc_idxs : context chars
# qw_idxs : question word indices
# qc_idxs : question chars
# y1 : answer start (?)
# y2 : answer span (?)

#                 # Setup for forward
#                 cw_idxs = cw_idxs.to(device)
#                 qw_idxs = qw_idxs.to(device)
#                 batch_size = cw_idxs.size(0)
#                 optimizer.zero_grad()
#
#                 # Forward
#                 log_p1, log_p2 = model(cw_idxs, qw_idxs)
#                 y1, y2 = y1.to(device), y2.to(device)
#                 loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
#                 loss_val = loss.item()
