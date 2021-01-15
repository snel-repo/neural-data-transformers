#!/usr/bin/env python3
# Author: Joel Ye

# Notebook for interactive model evaluation/analysis
# Allows us to interrogate model on variable data (instead of masked sample again)

#%%
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
from src.mask import UNMASKED_LABEL

from analyze_utils import init_by_ckpt
# Note that a whole lot of logger items will still be dumped into ./scripts/logs

grid = True
ckpt = "Grid" if grid else "PBT"

# Lorenz
variant = "lorenz-s1"

# Chaotic
variant = "chaotic-s1"

def get_info(variant):
    ckpt_path = f"/snel/home/joely/ray_results/ndt/gridsearch/{variant}/best_model/ckpts/{variant}.lve.pth"
    runner, spikes, rates = init_by_ckpt(ckpt_path, mode=DATASET_MODES.val)

    train_set = SpikesDataset(runner.config, runner.config.DATA.TRAIN_FILENAME, mode=DATASET_MODES.train, logger=runner.logger)
    train_spikes, train_rates = train_set.get_dataset()
    full_spikes = torch.cat([train_spikes.to(spikes.device), spikes], dim=0)
    full_rates = torch.cat([train_rates.to(rates.device), rates], dim=0)
    # full_spikes = spikes
    # full_rates = rates

    (
        unmasked_loss,
        pred_rates,
        layer_outputs,
        attn_weights,
        attn_list,
    ) = runner.model(full_spikes, mask_labels=full_spikes, return_weights=True)

    return full_spikes.cpu(), full_rates.cpu(), pred_rates, runner

chaotic_spikes, chaotic_rates, chaotic_ndt, chaotic_runner = get_info('chaotic-s1')
lorenz_spikes, lorenz_rates, lorenz_ndt, lorenz_runner = get_info('lorenz-s1')


#%%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

palette = sns.color_palette(palette='muted', n_colors=3, desat=0.9)
SMALL_SIZE = 12
MEDIUM_SIZE = 15
LARGE_SIZE = 18

def prep_plt(ax=plt.gca()):
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    # plt.rc('title', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels

    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.style.use('seaborn-muted')
    # plt.figure(figsize=(6,4))

    spine_alpha = 0.5
    ax.spines['right'].set_alpha(0.0)
    # plt.gca().spines['bottom'].set_alpha(spine_alpha)
    ax.spines['bottom'].set_alpha(0)
    # plt.gca().spines['left'].set_alpha(spine_alpha)
    ax.spines['left'].set_alpha(0)
    ax.spines['top'].set_alpha(0.0)

    plt.tight_layout()

prep_plt()

# np.random.seed(5)

# plt.text("NDT Predictions", color=palette[0])
# plt.text("Ground Truth", color=palette[2])
def plot_axis(
    spikes,
    true_rates,
    ndt_rates, # ndt
    ax,
    key="lorenz",
    condition_idx=0,
    # trial_idx=0,
    neuron_idx=0,
    seed=3, # legacy,
    first_n=8
):
    prep_plt(ax)
    # np.random.seed(seed)
    # neuron_idx = np.random.randint(0, true_rates.size(-1))
    # trial_idx = np.random.randint(0, true_rates.size(0))
    _, unique, counts = np.unique(true_rates.numpy(), axis=0, return_inverse=True, return_counts=True)
    trial_idx = (unique == condition_idx)

    true_trial_rates = true_rates[trial_idx][0, :, neuron_idx].cpu()
    pred_trial_rates = ndt_rates[trial_idx][0:first_n, :, neuron_idx].cpu()
    trial_spikes = spikes[trial_idx][0:first_n, :, neuron_idx].cpu()

    time = true_rates.size(1)
    if True: # lograte
        true_trial_rates = true_trial_rates.exp()
        pred_trial_rates = pred_trial_rates.exp()
    lfads_preds = np.load(f"/snel/home/joely/data/{key}.npy")
    lfads_trial_rates = lfads_preds[trial_idx][0:first_n, :, neuron_idx]

    if key == "lorenz":
        spike_level = -0.06
    else:
        spike_level = -0.03
    one_tenth_point = 0.1 * (50 if key == "lorenz" else 100)

    ax.plot([0, one_tenth_point], [(first_n + 1) * spike_level, (first_n + 1) * spike_level], 'k-', lw=3) # 5 / 50 = 0.1

    ax.plot(np.arange(time), true_trial_rates.numpy(), color='#111111', label="Ground Truth") #, linestyle="dashed")

    # Only show labels once
    labels = {
        "NDT": "NDT",
        'LFADS': "AutoLFADS",
        'Spikes': 'Spikes'
    }

    for trial_idx in range(pred_trial_rates.numpy().shape[0]):
        # print(trial_idx)
        # print(pred_trial_rates.size())
        ax.plot(np.arange(time), pred_trial_rates[trial_idx].numpy(), color=palette[0], label=labels['NDT'], alpha=0.4)
        labels['NDT'] = ""
        ax.plot(np.arange(time), lfads_trial_rates[trial_idx], color=palette[1], label=labels['LFADS'], alpha=0.4)
        labels["LFADS"] = ""
        spike_times, = np.where(trial_spikes[trial_idx].numpy())
        ax.scatter(spike_times, spike_level * (trial_idx + 1)*np.ones_like(spike_times), c='k', marker='|', label=labels['Spikes'], s=30)
        labels['Spikes'] = ""

    ax.set_xticks([])
    # ax.set_xticks(np.linspace(one_tenth_point, one_tenth_point, 1))

    if key == "lorenz":
        ax.set_ylim(-0.6, 0.65)
        ax.set_yticks([0.0, 0.4])
        ax.plot([-1, -1], [0, 0.4], 'k-', lw=3)

        ax.annotate("",
            xy=(-1.5, -0.5),
            xycoords="data",
            xytext=(-1.5, -0.3),
            textcoords="data",
            arrowprops=dict(
                arrowstyle="<-",
                connectionstyle="arc3,rad=0",
                linewidth="2",
                # color=(0.2, 0.2, 0.2)
            ),
            size=14
        )

        ax.text(
            -5.5, -0.5,
            # -3.5, -0.5,
            "Trials",
            fontsize=14,
            rotation=90
        )

    else:
        ax.set_ylim(-0.3, 0.5)
        ax.set_yticks([0, 0.2])
        ax.plot([-2, -2], [0, 0.2], 'k-', lw=3)

        ax.annotate("",
            xy=(-3, -0.25),
            xycoords="data",
            xytext=(-3, -0.15),
            textcoords="data",
            arrowprops=dict(
                arrowstyle="<-",
                connectionstyle="arc3,rad=0",
                linewidth="2",
                # color=(0.2, 0.2, 0.2)
            ),
            size=14
        )

        ax.text(
            -11, -0.25,
            # -7, -0.25,
            "Trials",
            fontsize=14,
            rotation=90
        )

    # ax.set_title(f"Trial {trial_idx}, Neuron {neuron_idx}")

f, axes = plt.subplots(
    nrows=2, ncols=2, sharex=False, sharey=False, figsize=(8, 6)
)

plot_axis(
    lorenz_spikes,
    lorenz_rates,
    lorenz_ndt,
    axes[0, 0],
    condition_idx=0,
    key="lorenz"
)

plot_axis(
    lorenz_spikes,
    lorenz_rates,
    lorenz_ndt,
    axes[0, 1],
    condition_idx=1,
    key="lorenz"
)

plot_axis(
    chaotic_spikes,
    chaotic_rates,
    chaotic_ndt,
    axes[1, 0],
    condition_idx=0,
    key="chaotic"
)

plot_axis(
    chaotic_spikes,
    chaotic_rates,
    chaotic_ndt,
    axes[1, 1],
    condition_idx=1,
    key="chaotic"
)

# plt.suptitle(f"{variant} {ckpt}", y=1.05)
axes[0, 0].text(15, 0.45, "Lorenz", size=18, rotation=0)
# axes[0, 0].text(-20, 0.2, "Lorenz", size=18, rotation=45)
axes[1, 0].text(30, 0.1, "Chaotic", size=18, rotation=0)
# axes[1, 0].text(20, 0.25, "Chaotic", size=18, rotation=0)

# plt.tight_layout()
f.subplots_adjust(
    # left=0.15,
    # bottom=-0.1,
    hspace=0.0,
    wspace=0.0
)
# axes[0,0].set_xticks([])
axes[0,1].set_yticks([])
# axes[0,1].set_xticks([])
axes[1,1].set_yticks([])

legend = axes[1, 1].legend(
    # loc=(-.95, 0.97),
    loc=(-1.05, 0.85),
    fontsize=14,
    frameon=False,
    ncol=4,
)

# for line in legend.get_lines():
#     line.set_linewidth(3.0)

# plt.savefig("lorenz_rates.png", dpi=300, bbox_inches="tight")
# plt.savefig("lorenz_rates_2.png", dpi=300, bbox_inches="tight")
plt.setp(legend.get_texts()[1], color=palette[0])
plt.setp(legend.get_texts()[2], color=palette[1])
plt.setp(legend.get_texts()[0], color="#111111")
# plt.setp(legend.get_texts()[3], color=palette[2])

plt.savefig("3a_synth_qual.pdf")

#%%
# Total 1300 (130 x 10 per)
# Val 260 (130 x 2 per)
_, unique, counts = np.unique(chaotic_rates.numpy(), axis=0, return_inverse=True, return_counts=True)
# unique, counts = chaotic_rates.unique(dim=0, return_counts=True)
trial_idx = (unique == 0)
# Oh, a lot of people look like
print(np.where(trial_idx))
true_trial_rates = chaotic_rates[trial_idx][0, :, 0].cpu()
print(true_trial_rates[:10])