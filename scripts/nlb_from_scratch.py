#%%
"""
- Let's setup NDT for train/eval from scratch.
- (Assumes your device is reasonably pytorch/GPU compatible)
- This is an interactive python script run via vscode. If you'd like to run as a notebook

Run to setup requirements:
```
    Making a new env
    - python 3.7
    - conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    - conda install seaborn
    - pip install yacs pytorch_transformers tensorboard "ray[tune]" sklearn
    - pip install dandi "pynwb>=2.0.0"
    Or from nlb.yaml

    Then,
    - conda develop ~/path/to/nlb_tools
```

Install NLB dataset(s), and create the h5s for training. (Here we install MC_Maze_Small)
```
    pip install dandi
    dandi download DANDI:000140/0.220113.0408
```

This largely follows the `basic_example.ipynb` in `nlb_tools`. The only distinction is that we save to an h5.
"""

from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import (
    make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5
)
from nlb_tools.evaluation import evaluate

import numpy as np
import pandas as pd
import h5py

import logging
logging.basicConfig(level=logging.INFO)

dataset_name = 'mc_maze_small'
datapath = '/home/joelye/user_data/nlb/000140/sub-Jenkins/'
dataset = NWBDataset(datapath)

# Prepare dataset
phase = 'val'

# Choose bin width and resample
bin_width = 5
dataset.resample(bin_width)

# Create suffix for group naming later
suffix = '' if (bin_width == 5) else f'_{int(bin_width)}'

train_split = 'train' if (phase == 'val') else ['train', 'val']
train_dict = make_train_input_tensors(
    dataset, dataset_name=dataset_name, trial_split=train_split, save_file=False,
    include_behavior=True,
    include_forward_pred = True,
)

# Show fields of returned dict
print(train_dict.keys())

# Unpack data
train_spikes_heldin = train_dict['train_spikes_heldin']
train_spikes_heldout = train_dict['train_spikes_heldout']

# Print 3d array shape: trials x time x channel
print(train_spikes_heldin.shape)

## Make eval data (i.e. val)

# Split for evaluation is same as phase name
eval_split = phase
# eval_dict = make_eval_input_tensors(
#     dataset, dataset_name=dataset_name, trial_split=eval_split, save_file=False,
# )
# Make data tensors - use all chunks including forward prediction for training NDT
eval_dict = make_train_input_tensors(
    dataset, dataset_name=dataset_name, trial_split=['val'], save_file=False, include_forward_pred=True,
)
eval_dict = {
    f'eval{key[5:]}': val for key, val in eval_dict.items()
}
eval_spikes_heldin = eval_dict['eval_spikes_heldin']

print(eval_spikes_heldin.shape)

h5_dict = {
    **train_dict,
    **eval_dict
}

h5_target = '/home/joelye/user_data/nlb/mc_maze_small.h5'
save_to_h5(h5_dict, h5_target, overwrite=True)


#%%
"""
- At this point we should be able to train a basic model.
- In CLI, run a training call, replacing the appropriate paths
```
    ./scripts/train.sh mc_maze_small_from_scratch

    OR

    python ray_random.py -e ./configs/mc_maze_small_from_scratch.yaml
    (CLI overrides aren't available here, so make another config file)
```
- Once this is done training (~0.5hr for non-search), let's load the results...
"""
import os
import os.path as osp
from pathlib import Path
import sys

# Add ndt src if not in path
module_path = osp.abspath(osp.join('..'))
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
from ray import tune

from nlb_tools.evaluation import evaluate

from src.run import prepare_config
from src.runner import Runner
from src.dataset import SpikesDataset, DATASET_MODES
from analyze_utils import init_by_ckpt

variant = "mc_maze_small_from_scratch"

is_ray = True
# is_ray = False

if is_ray:
    tune_dir = f"/home/joelye/user_data/nlb/ndt_runs/ray/{variant}"
    df = tune.ExperimentAnalysis(tune_dir).dataframe()
    # ckpt_path = f"/home/joelye/user_data/nlb/ndt_runs/ray/{variant}/best_model/ckpts/{variant}.lve.pth"
    ckpt_dir = df.loc[df["best_masked_loss"].idxmin()].logdir
    ckpt_path = f"{ckpt_dir}/ckpts/{variant}.lve.pth"
    runner, spikes, rates, heldout_spikes, forward_spikes = init_by_ckpt(ckpt_path, mode=DATASET_MODES.val)
else:
    ckpt_dir = Path("/home/joelye/user_data/nlb/ndt_runs/")
    ckpt_path = ckpt_dir / variant / f"{variant}.lve.pth"
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
# * Viz some trials
trials = [1, 2, 3]
neuron = 10
trial_rates = train_rates_heldout[trials, :, neuron].cpu()
trial_spikes = heldout_spikes[trials, :, neuron].cpu()
trial_time = heldout_spikes.size(1)

spike_level = 0.05
f, axes = plt.subplots(2, figsize=(6, 4))
times = np.arange(0, trial_time * 0.05, 0.05)
for trial_index in range(trial_spikes.size(0)):
    spike_times, = np.where(trial_spikes[trial_index].numpy())
    spike_times = spike_times * 0.05
    axes[0].scatter(spike_times, spike_level * (trial_index + 1)*np.ones_like(spike_times), marker='|', label='Spikes', s=30)
    axes[1].plot(times, trial_rates[trial_index].exp())

#%%
# Submission e.g. as in `basic_example.ipynb`
# Looks like this model is pretty terrible :/
output_dict = {
    dataset_name + suffix: {
        'train_rates_heldin': train_rates_heldin.cpu().numpy(),
        'train_rates_heldout': train_rates_heldout.cpu().numpy(),
        'eval_rates_heldin': eval_rates_heldin.cpu().numpy(),
        'eval_rates_heldout': eval_rates_heldout.cpu().numpy(),
        # 'eval_rates_heldin_forward': eval_rates_heldin_forward.cpu().numpy(),
        # 'eval_rates_heldout_forward': eval_rates_heldout_forward.cpu().numpy()
    }
}

# Reset logging level to hide excessive info messages
logging.getLogger().setLevel(logging.WARNING)

# If 'val' phase, make the target data
if phase == 'val':
    # Note that the RTT task is not well suited to trial averaging, so PSTHs are not made for it
    target_dict = make_eval_target_tensors(dataset, dataset_name=dataset_name, train_trial_split='train', eval_trial_split='val', include_psth=True, save_file=False)

    # Demonstrate target_dict structure
    print(target_dict.keys())
    print(target_dict[dataset_name + suffix].keys())

# Set logging level again
logging.getLogger().setLevel(logging.INFO)

if phase == 'val':
    print(evaluate(target_dict, output_dict))

# e.g. with targets to compare to
# target_dict = torch.load(f'/snel/home/joely/tmp/{variant}_target.pth')
# target_dict = np.load(f'/snel/home/joely/tmp/{variant}_target.npy', allow_pickle=True).item()

# print(evaluate(target_dict, output_dict))

# e.g. to upload to EvalAI
# with h5py.File('ndt_maze_preds.h5', 'w') as f:
#     group = f.create_group('mc_maze')
#     for key in output_dict['mc_maze']:
#         group.create_dataset(key, data=output_dict['mc_maze'][key])