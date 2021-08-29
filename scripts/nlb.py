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
variant = "mc_maze"
# variant = "mc_maze_large"
# variant = "mc_maze_medium"
# variant = "mc_maze_small"
variant = 'dmfc_rsg'
variant = 'dmfc_rsg2'
# variant = 'mc_rtt'

is_ray = True
# is_ray = False

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

# target_dict = torch.load(f'/snel/home/joely/tmp/{variant}_target.pth')
target_dict = np.load(f'/snel/home/joely/tmp/{variant}_target.npy', allow_pickle=True).item()

print(evaluate(target_dict, output_dict))

#%%
# * Test

variant = 'dmfc_rsg'
runner.config.defrost()
runner.config.DATA.TRAIN_FILENAME = f'{variant}_test.h5'
runner.config.freeze()
train_rates, _ = runner.get_rates(
    checkpoint_path=ckpt_path,
    save_path = None,
    mode = DATASET_MODES.train
)
eval_rates, _ = runner.get_rates(
    checkpoint_path=ckpt_path,
    save_path = None,
    mode = DATASET_MODES.val,
)

eval_rates, eval_rates_forward = torch.split(eval_rates, [spikes.size(1), eval_rates.size(1) - spikes.size(1)], 1)
eval_rates_heldin_forward, eval_rates_heldout_forward = torch.split(eval_rates_forward, [spikes.size(-1), heldout_spikes.size(-1)], -1)
train_rates, _ = torch.split(train_rates, [spikes.size(1), train_rates.size(1) - spikes.size(1)], 1)
eval_rates_heldin, eval_rates_heldout = torch.split(eval_rates, [spikes.size(-1), heldout_spikes.size(-1)], -1)
train_rates_heldin, train_rates_heldout = torch.split(train_rates, [spikes.size(-1), heldout_spikes.size(-1)], -1)

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

print(evaluate('/snel/share/data/nlb/test_data_do_not_share/eval_data_test.h5', output_dict))

#%%
import h5py
with h5py.File('ndt_maze_preds.h5', 'w') as f:
    group = f.create_group('mc_maze')
    for key in output_dict['mc_maze']:
        group.create_dataset(key, data=output_dict['mc_maze'][key])
#%%
# Viz some trials
trials = [1, 2, 3]
neuron = 10
trial_rates = train_rates_heldout[trials, :, neuron].cpu()
trial_spikes = heldout_spikes[trials, :, neuron].cpu()
trial_time = heldout_spikes.size(1)
"""  """
spike_level = 0.05
# one_tenth_point = 0.1 * (50 if key == "lorenz" else 100)
# ax.plot([0, one_tenth_point], [(first_n + 1) * spike_level, (first_n + 1) * spike_level], 'k-', lw=3) # 5 / 50 = 0.1

for trial_index in range(trial_spikes.size(0)):
    print(trial_spikes[trial_index])
    # plt.plot(trial_rates[trial_index].exp())
    spike_times, = np.where(trial_spikes[trial_index].numpy())
    plt.scatter(spike_times, spike_level * (trial_index + 1)*np.ones_like(spike_times), c='k', marker='|', label='Spikes', s=30)
# labels['Spikes'] = ""

print(train_rates_heldout.size())

# %%
print(heldout_spikes.sum(0).sum(0).argmax())
# print(heldout_spikes[:,:,15])
for i in range(0, heldout_spikes.size(0), 6):
    plt.plot(heldout_spikes[i, :, 15].cpu())