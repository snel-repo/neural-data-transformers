import os
import os.path as osp
import pdb
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

def try_savedown(variant, runner, ckpt_path, data_path, save_path):
    # Override the config dataset so as to predict on the whole dataset even for subsets
    # Port for rds analysis
    if "maze" in variant or "m700" in variant:
        runner.config.defrost()
        runner.config.DATA.DATAPATH = data_path
        runner.config.freeze()
        rate_output_pth = save_path
        rate_output_fn = f"{variant}.h5"
        pred_rates, layer_outputs = runner.get_rates(
            checkpoint_path=ckpt_path,
            save_path=osp.join(rate_output_pth, rate_output_fn)
        )
        return pred_rates, layer_outputs
    return None, None




run_name = "sr06"
data_name = "I152_R91_seed0_SR0.6"

variant = "maze_sbtt"
#variant = "m700_sbtt"
ckpt_path = f"/snel/share/share/derived/selective_backprop/runs/ndt/{run_name}/{variant}/{variant}.lve.pth"
data_path = f"/snel/share/share/derived/selective_backprop/runs/ndt/data/{data_name}/"
save_path = f"/snel/share/share/derived/selective_backprop/runs/ndt/{run_name}/"

runner, spikes, rates = init_by_ckpt(ckpt_path, mode=DATASET_MODES.val)
print(runner.config.VARIANT)
pred_rates, layer_outputs = try_savedown(variant, runner, ckpt_path, data_path, save_path)

pdb.set_trace()
