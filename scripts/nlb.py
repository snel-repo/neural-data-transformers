#%%

# 1. Load model and get rate predictions
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

from nlb_tools.evaluation import evaluate

from src.run import prepare_config
from src.runner import Runner
from src.dataset import SpikesDataset, DATASET_MODES
from src.mask import UNMASKED_LABEL

from analyze_utils import init_by_ckpt

variant = "area2_bump"


is_ray = True
is_ray = False

if is_ray:
    best_model = "best_model"
    # best_model = "best_model_unmasked"
    lve = "lfve" if "unmasked" in best_model else "lve"

    def to_path(variant):
        grid_var = f"{variant}_lite"
        ckpt_path = f"/snel/home/joely/ray_results/ndt/gridsearch/{grid_var}/best_model/ckpts/{grid_var}.{lve}.pth"
        runner, spikes, rates, heldout_spikes, forward_spikes = init_by_ckpt(ckpt_path, mode=DATASET_MODES.val)
        print(runner.config.VARIANT)
        return runner, spikes, rates, heldout_spikes, ckpt_path

    runner, spikes, rates, heldout_spikes, ckpt_path = to_path(variant)
else:
    ckpt_dir = Path(f"/snel/share/joel/transformer_modeling/{variant}/")
    ckpt_path = ckpt_dir.joinpath(f"{variant}.lve.pth")
    runner, spikes, rates, heldout_spikes, forward_spikes = init_by_ckpt(ckpt_path, mode=DATASET_MODES.val)

eval_rates, _ = runner.get_rates(
    checkpoint_path=ckpt_path,
    save_path = None,
    mode = DATASET_MODES.val
)
train_rates, _ = runner.get_rates(
    checkpoint_path=ckpt_path,
    save_path = None,
    mode = DATASET_MODES.train
)
# * Val
eval_rates, eval_rates_forward = torch.split(eval_rates, [spikes.size(1), eval_rates.size(1) - spikes.size(1)], 1)
eval_rates_heldin_forward, eval_rates_heldout_forward = torch.split(eval_rates_forward, [spikes.size(-1), heldout_spikes.size(-1)], -1)
train_rates, _ = torch.split(train_rates, [spikes.size(1), train_rates.size(1) - spikes.size(1)], 1)
eval_rates_heldin, eval_rates_heldout = torch.split(eval_rates, [spikes.size(-1), heldout_spikes.size(-1)], -1)
train_rates_heldin, train_rates_heldout = torch.split(train_rates, [spikes.size(-1), heldout_spikes.size(-1)], -1)

#%%

output_dict = {
    variant: {
        'train_rates_heldin': train_rates_heldin.cpu().numpy(),
        'train_rates_heldout': train_rates_heldout.cpu().numpy(),
        'eval_rates_heldin': eval_rates_heldin.cpu().numpy(),
        'eval_rates_heldout': eval_rates_heldout.cpu().numpy(),
        'eval_rates_heldin_forward': eval_rates_heldin_forward.cpu().numpy(),
        'eval_rates_heldout_forward': eval_rates_heldout_forward.cpu().numpy()
    }
}

target_dict = torch.load('/snel/home/joely/tmp/area2_bump_target.pth')

print(evaluate(target_dict, output_dict))

#%%
# * Test

runner.config.defrost()
runner.config.DATA.TRAIN_FILENAME = 'area2_bump_test.h5'
runner.config.freeze()
train_rates, _ = runner.get_rates(
    checkpoint_path=ckpt_path,
    save_path = None,
    mode = DATASET_MODES.train
)
test_rates, _ = runner.get_rates(
    checkpoint_path=ckpt_path,
    save_path = None,
    mode = DATASET_MODES.val,
)
eval_rates_heldin, eval_rates_heldout = torch.split(test_rates, [spikes.size(-1), heldout_spikes.size(-1)], -1)
train_rates_heldin, train_rates_heldout = torch.split(train_rates, [spikes.size(-1), heldout_spikes.size(-1)], -1)

output_dict = {
    variant: {
        'train_rates_heldin': train_rates_heldin.cpu().numpy(),
        'train_rates_heldout': train_rates_heldout.cpu().numpy(),
        'eval_rates_heldin': eval_rates_heldin.cpu().numpy(),
        'eval_rates_heldout': eval_rates_heldout.cpu().numpy()
    }
}

print(evaluate('/snel/share/data/nlb/test_data_do_not_share/eval_data_test.h5', output_dict))
