#!/usr/bin/env python3
# Author: Joel Ye

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

def binary_mask_to_attn_mask(x):
    return x.float().masked_fill(x == 0, float('-inf')).masked_fill(x == 1, float(0.0))

# Verbatim from LFADS_TF2
def merge_train_valid(train_data, valid_data, train_ixs, valid_ixs):
    """Merges training and validation numpy arrays using indices.

    This function merges training and validation numpy arrays
    in the appropriate order using arrays of their indices. The
    lengths of the indices must be the same as the first dimension
    of the corresponding data.

    Parameters
    ----------
    train_data : np.ndarray
        An N-dimensional numpy array of training data with
        first dimension T.
    valid_data : np.ndarray
        An N-dimensional numpy array of validation data with
        first dimension V.
    train_ixs : np.ndarray
        A 1-D numpy array of training indices with length T.
    valid_ixs : np.ndarray
        A 1-D numpy array of validation indices with length V.

    Returns
    -------
    np.ndarray
        An N-dimensional numpy array with dimension T + V.

    """

    if train_data.shape[0] == train_ixs.shape[0] \
        and valid_data.shape[0] == valid_ixs.shape[0]:
        # if the indices match up, then we can use them to merge
        data = np.full_like(np.concatenate([train_data, valid_data]), np.nan)
        if min(min(train_ixs), min(valid_ixs)) > 0:
            # we've got matlab data...
            train_ixs -= 1
            valid_ixs -= 1
        data[train_ixs.astype(int)] = train_data
        data[valid_ixs.astype(int)] = valid_data
    else:
        # if the indices do not match, train and
        # valid data may be the same (e.g. for priors)
        if np.all(train_data == valid_data):
            data = train_data
        else:
            raise ValueError("shape mismatch: "
                f"Index shape {train_ixs.shape} does not "
                f"match the data shape {train_data.shape}.")
    return data

def get_inverse_sqrt_schedule(optimizer, warmup_steps=1000, lr_init=1e-8, lr_max=5e-4):
    """
    https://github.com/pytorch/fairseq/blob/master/fairseq/optim/lr_scheduler/inverse_square_root_schedule.py
    Decay the LR based on the inverse square root of the update number.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.
    During warmup::
      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]
    After warmup::
      decay_factor = args.lr * sqrt(args.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """
    def lr_lambda(current_step):
        lr_step = (lr_max - lr_init) / warmup_steps
        decay_factor = lr_max * warmup_steps ** 0.5

        if current_step < warmup_steps:
            lr = lr_init + current_step * lr_step
        else:
            lr = decay_factor * current_step ** -0.5
        return lr

    return LambdaLR(optimizer, lr_lambda)