import os
import os.path as osp

import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import time
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as f
from torch.utils import data

from src.run import prepare_config
from src.runner import Runner
from src.dataset import SpikesDataset, DATASET_MODES
from src.mask import Masker, UNMASKED_LABEL

def make_runner(variant, ckpt, base="", prefix=""):
    run_type = "eval"
    exp_config = osp.join("../configs", prefix, f"{variant}.yaml")
    if base != "":
        exp_config = [osp.join("../configs", f"{base}.yaml"), exp_config]
    ckpt_path = f"{variant}.{ckpt}.pth"
    config, ckpt_path = prepare_config(
            exp_config, run_type, ckpt_path, [
                "USE_TENSORBOARD", False,
                "SYSTEM.NUM_GPUS", 1,
            ], suffix=prefix
        )
    return Runner(config), ckpt_path

def setup_dataset(runner, mode):
    test_set = SpikesDataset(runner.config, runner.config.DATA.VAL_FILENAME, mode=mode, logger=runner.logger)
    runner.logger.info(f"Evaluating on {len(test_set)} samples.")
    test_set.clip_spikes(runner.max_spikes)
    spikes, rates, heldout_spikes, forward_spikes = test_set.get_dataset()
    if heldout_spikes is not None:
        heldout_spikes = heldout_spikes.to(runner.device)
    if forward_spikes is not None:
        forward_spikes = forward_spikes.to(runner.device)
    return spikes.to(runner.device), rates.to(runner.device), heldout_spikes, forward_spikes

def init_by_ckpt(ckpt_path, mode=DATASET_MODES.val):
    runner = Runner(checkpoint_path=ckpt_path)
    runner.model.eval()
    torch.set_grad_enabled(False)
    spikes, rates, heldout_spikes, forward_spikes = setup_dataset(runner, mode)
    return runner, spikes, rates, heldout_spikes, forward_spikes

def init(variant, ckpt, base="", prefix="", mode=DATASET_MODES.val):
    runner, ckpt_path = make_runner(variant, ckpt, base, prefix)
    return init_by_ckpt(ckpt_path, mode)


# Accumulates multiplied attentions - examine at lower layers to see where information is sourced in early processing.
# Takes in layer weights
def get_multiplicative_weights(weights_list):
    weights = weights_list[0]
    multiplied_weights  = [weights]
    for layer_weights in weights_list[1:]:
        weights = torch.bmm(layer_weights, weights)
        multiplied_weights.append(weights)
    return multiplied_weights
