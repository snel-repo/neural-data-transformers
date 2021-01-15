#!/usr/bin/env python3
# Author: Joel Ye

# Notebook for interactive model evaluation/analysis
# Allows us to interrogate model on variable data (instead of masked sample again)

#%%
import os
import os.path as osp
from pathlib import Path
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
from src.mask import UNMASKED_LABEL

from analyze_utils import init_by_ckpt

grid = False
grid = True

variant = "m700_2296-s1"

val_errs = {}
def save_rates(ckpt_path, handle):
    runner, spikes, rates = init_by_ckpt(ckpt_path, mode=DATASET_MODES.val)
    if "maze" in variant or "m700" in variant:
        runner.config.defrost()
        runner.config.DATA.DATAPATH = "/snel/share/data/ndt_paper/m1_maze/heldout_trial/2296_trials/0_seed"
        runner.config.freeze()
        rate_output_pth = f"/snel/share/joel/ndt_rates/psth_match"
        rate_output_pth = osp.join(rate_output_pth, "grid" if grid else "pbt")
        rate_output_fn = f"{handle}_{variant}.h5"
        val_errs[handle] = runner.best_val['value'].cpu().numpy()


sweep_dir = Path("/snel/home/joely/ray_results/ndt/")
if grid:
    sweep_dir = sweep_dir.joinpath("gridsearch")
sweep_dir = sweep_dir.joinpath(variant, variant)
for run_dir in sweep_dir.glob("tuneNDT*/"):
    run_ckpt_path = run_dir.joinpath(f'ckpts/{variant}.lve.pth')
    handle = run_dir.parts[-1][:10]
    save_rates(run_ckpt_path, handle)

#%%
torch.save(val_errs, f'{variant}_val_errs_sweep.pth')