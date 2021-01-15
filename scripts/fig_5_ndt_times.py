#!/usr/bin/env python3
# Author: Joel Ye

# Run from scripts directory.
# python timing_tests.py -l {1, 2, 6}

#%%
import os
import os.path as osp
import argparse
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import time
import gc
gc.disable()

import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.utils import data
import torch.nn.functional as f

from src.dataset import DATASET_MODES, SpikesDataset
from src.run import prepare_config
from src.runner import Runner
from analyze_utils import make_runner, get_multiplicative_weights, init_by_ckpt

prefix="arxiv"
base=""
variant = "chaotic"

run_type = "eval"
exp_config = osp.join("../configs", prefix, f"{variant}.yaml")
if base != "":
    exp_config = [osp.join("../configs", f"{base}.yaml"), exp_config]

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-layers", "-l",
        type=int,
        required=True,
    )
    return parser

parser = get_parser()
args = parser.parse_args()
layers = vars(args)["num_layers"]

def make_runner_of_bins(bin_count=100, layers=None):
    config, _ = prepare_config(
        exp_config, run_type, "", [
            "USE_TENSORBOARD", False,
            "SYSTEM.NUM_GPUS", 1,
        ], suffix=prefix
    )
    config.defrost()
    config.MODEL.TRIAL_LENGTH = bin_count
    if layers is not None:
        config.MODEL.NUM_LAYERS = layers
    config.MODEL.LEARNABLE_POSITION = False # Not sure why...
    config.freeze()
    return Runner(config)

def time_length(trials=1300, bin_count=100, **kwargs):
    # 100 as upper bound
    runner = make_runner_of_bins(bin_count=bin_count, **kwargs)
    runner.logger.mute()
    runner.load_device()
    runner.max_spikes = 9 # from chaotic ckpt
    runner.num_neurons = 50 # from chaotic ckpt
    # runner.num_neurons = 202 # from chaotic ckpt
    runner.setup_model(runner.device)
    # whole_set = SpikesDataset(runner.config, runner.config.DATA.TRAIN_FILENAME, mode="trainval")
    # whole_set.clip_spikes(runner.max_spikes)
    # # print(f"Evaluating on {len(whole_set)} samples.")
    # data_generator = data.DataLoader(whole_set,
    #     batch_size=1, shuffle=False
    # )
    loop_times = []
    with torch.no_grad():
        probs = torch.full((1, bin_count, runner.num_neurons), 0.1)
        # probs = torch.full((1, bin_count, runner.num_neurons), 0.01)
        while len(loop_times) < trials:
            spikes = torch.bernoulli(probs).long()
            spikes = spikes.to(runner.device)
            start = time.time()
            runner.model(spikes, mask_labels=spikes, passthrough=True)
            delta = time.time() - start
            loop_times.append(delta)
    p_loop_times = np.array(loop_times) * 1e3
    print(f"{p_loop_times.mean():.4f}ms for {bin_count} bins")

    # A note about memory: It's a bit unclear why `empty_cache` is failing and memory still shows as used on torch, but the below diagnostic indicates the memory is not allocated, and will not cause OOM. So, a minor inconvenience for now.
    # device = runner.device
    # runner.model.to('cpu')
    # t = torch.cuda.get_device_properties(device).total_memory
    # c = torch.cuda.memory_cached(device)
    # a = torch.cuda.memory_allocated(device)
    # print(device, t, c, a)
    # del runner
    # t = torch.cuda.get_device_properties(device).total_memory
    # c = torch.cuda.memory_cached(device)
    # a = torch.cuda.memory_allocated(device)
    # print(device, t, c, a)
    # del data_generator
    # del whole_set
    # del spikes
    # torch.cuda.empty_cache()
    return p_loop_times

times = []
for i in range(5, 15, 5):
    p_loop_times = time_length(trials=2000, bin_count=i, layers=layers)
    times.append(p_loop_times)

times = np.stack(times, axis=0)
np.save(f'ndt_times_layer_{layers}', times)

#%%

