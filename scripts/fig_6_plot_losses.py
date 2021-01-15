#%%
from pathlib import Path
import sys
module_path = str(Path('..').resolve())
if module_path not in sys.path:
    sys.path.append(module_path)
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as f
from torch.utils import data

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.run import prepare_config
from src.runner import Runner
from src.dataset import SpikesDataset, DATASET_MODES
from src.mask import UNMASKED_LABEL

from analyze_utils import init_by_ckpt

tf_size_guidance = {'scalars': 1000}
#%%

plot_folder = Path('~/ray_results/ndt/gridsearch/lograte_tb').expanduser().resolve()
variants = {
    "m700_no_log-s1": "Rates",
    "m700_2296-s1": "Logrates"
}

all_info = defaultdict(dict) # key: variant, value: dict per variant, keyed by run (value will dict of step and value)
for variant in variants:
    v_dir = plot_folder.joinpath(variant)
    for run_dir in v_dir.iterdir():
        if not run_dir.is_dir():
            continue
        tb_dir = run_dir.joinpath('tb')
        all_info[variant][tb_dir.parts[-2][:10]] = defaultdict(list) # key: "step" or "value", value: info
        for tb_file in tb_dir.iterdir():
            event_acc = EventAccumulator(str(tb_file), tf_size_guidance)
            event_acc.Reload()
            # print(event_acc.Tags())
            if 'val_loss' in event_acc.Tags()['scalars']:
                val_loss = event_acc.Scalars('val_loss')
                all_info[variant][tb_dir.parts[-2][:10]]['step'].append(val_loss[0].step)
                all_info[variant][tb_dir.parts[-2][:10]]['value'].append(val_loss[0].value)

#%%
SMALL_SIZE = 12
MEDIUM_SIZE = 15
LARGE_SIZE = 18

def prep_plt():
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    # plt.rc('title', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels

    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.style.use('seaborn-muted')
    # plt.figure(figsize=(6,4))

    spine_alpha = 0.5
    plt.gca().spines['right'].set_alpha(0.0)
    # plt.gca().spines['bottom'].set_alpha(spine_alpha)
    plt.gca().spines['bottom'].set_alpha(0)
    # plt.gca().spines['left'].set_alpha(spine_alpha)
    plt.gca().spines['left'].set_alpha(0)
    plt.gca().spines['top'].set_alpha(0.0)

    plt.tight_layout()

plt.figure(figsize=(6, 4))

prep_plt()
palette = sns.color_palette(palette='muted', n_colors=len(variants), desat=0.9)
colors = {}
color_ind = 0
for variant, label in variants.items():
    colors[variant] = palette[color_ind]
    color_ind += 1
    legend_str = label
    for run, run_info in all_info[variant].items():
        # Sort
        sort_ind = np.argsort(run_info['step'])
        steps = np.array(run_info['step'])[sort_ind]
        vals = np.array(run_info['value'])[sort_ind]
        plt.plot(steps, vals, color=colors[variant], label=legend_str)
        # plt.hlines(vals.min(),0, 25000, color=colors[variant])
        legend_str = ""
plt.legend(fontsize=MEDIUM_SIZE, frameon=False)
plt.xticks(np.arange(0, 25001, 12500))
plt.yticks(np.arange(0.1, 1.21, 0.5))
plt.ylim(0.1, 1.21)
# plt.yticks(np.arange(0.2, 1.21, 0.5))
plt.ylabel("Loss (NLL)")
plt.yscale('log')
plt.xlabel("Epochs")
plt.savefig("6_losses.pdf", bbox_inches="tight")