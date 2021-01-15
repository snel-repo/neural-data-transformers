#!/usr/bin/env python3
# Author: Joel Ye

# Notebook for interactive model evaluation/analysis (catch all analysis.)
# This notebook saves down inferred rates for maze analysis.

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

grid = False
grid = True
ckpt = "Grid" if grid else "PBT"

# Lorenz
variant = "lorenz-s1"

# Chaotic
variant = "chaotic-s1"

# Reaching
# variant = "m700_2296-s1"

# stems = ["m700_no_reg"]
# variants = []
# for stem in stems:
#     variants.extend(f"{stem}-s{i}" for i in [1,2,3])
# variants = ['m700_2296-s1', 'm700_2296-s2', 'm700_2296-s3']

best_model = "best_model"
best_model = "best_model_unmasked"
lve = "lfve" if "unmasked" in best_model else "lve"

def to_path(variant):
    if grid:
        ckpt_path = f"/snel/home/joely/ray_results/ndt/gridsearch/{variant}/best_model/ckpts/{variant}.{lve}.pth"
    else:
        ckpt_path = f"/snel/home/joely/ray_results/ndt/{variant}/best_model/ckpts/{variant}.{lve}.pth"
    runner, spikes, rates = init_by_ckpt(ckpt_path, mode=DATASET_MODES.val)
    print(runner.config.VARIANT)
    # print(runner.config.MODEL.CONTEXT_FORWARD)
    # print(runner.config.MODEL.CONTEXT_BACKWARD)
    # print(runner.config.TRAIN.MASK_RANDOM_RATIO)
    # print(runner.config.TRAIN.MASK_TOKEN_RATIO)
    # print(runner.config.TRAIN.MASK_MAX_SPAN)
    # print(runner.config.MODEL.DROPOUT_RATES)
    return runner, spikes, rates, ckpt_path

def try_savedown(variant, runner, ckpt_path):
# Override the config dataset so as to predict on the whole dataset even for subsets
# Port for rds analysis
    if "maze" in variant or "m700" in variant:
        runner.config.defrost()
        runner.config.DATA.DATAPATH = "/snel/share/data/ndt_paper/m1_maze/heldout_trial/2296_trials/0_seed"
        # runner.config.DATA.DATAPATH = "/snel/share/data/ndt_paper/m1_maze_2ms/heldout_trial/2296_trials/0_seed"
        runner.config.freeze()
        rate_output_pth = f"/snel/share/joel/ndt_rates/"
        rate_output_pth = osp.join(rate_output_pth, "grid" if grid else "pbt")
        rate_output_fn = f"{variant}.h5"
        pred_rates, layer_outputs = runner.get_rates(
            checkpoint_path=ckpt_path,
            save_path=osp.join(rate_output_pth, rate_output_fn)
        )
        return pred_rates, layer_outputs
    return None, None

def savedown_variant(variant):
    runner, _, _, ckpt_path = to_path(variant)
    try_savedown(variant, runner, ckpt_path)
# [savedown_variant(v) for v in variants]

runner, spikes, rates, ckpt_path = to_path(variant)
try_savedown(variant, runner, ckpt_path)


#%%

(
    unmasked_loss,
    pred_rates,
    layer_outputs,
    attn_weights,
    attn_list,
) = runner.model(spikes, mask_labels=spikes, return_weights=True)
prinst(f"Unmasked: {unmasked_loss}")

if "maze" not in variant and "m700" not in variant:
    r2 = runner.neuron_r2(rates, pred_rates, flatten=True)
    # r2 = runner.neuron_r2(rates, pred_rates)
    vaf = runner.neuron_vaf(rates, pred_rates, flatten=True)
    # vaf = runner.neuron_vaf(rates, pred_rates)
    print(f"R2:\t{r2}, VAF:\t{vaf}")

trials, time, num_neurons = rates.size()
print(f"Best Masked Val: {runner.best_val}") # `  best val is .380 at 1302...
print(f"Best Unmasked Val: {runner.best_unmasked_val}") # .37763
print(f"Best R2: {runner.best_R2}") # best val is .300 at 1302...
print(runner.count_updates)

# print(runner.config.DATA)
# print(runner.config.TRAIN.MASK_RANDOM_RATIO)



#%%
# Analyze weights (bxtxt)
def norm_attention_weights(weights, use_average=True, trial=0):
    if use_average:
        weight = weights.mean(dim=0)
    else:
        weight = weights[trial]
    # Normalize along attended time
    norm_weights = f.normalize(weight, p=1, dim=1)
    return norm_weights.cpu()

def plot_ax_attention(ax, weights, use_average=False, trial=0):
    norm_weights = norm_attention_weights(weights, use_average=use_average, trial=trial)
    # im = plt.pcolormesh(norm_weights, edgecolors='k', linewidth=1, cmap="Reds")
    im = ax.imshow(norm_weights, cmap='hot', interpolation='nearest', aspect='auto')
    # target/source naming is tricky
    # it's transformer "source" encoding into "target"
    ax.set_ylabel("Tgt-Attender")
    ax.set_xlabel("Src-Attended")
    plt.colorbar(im ,ax=ax)

    # Highlight diagonal
    t = np.arange(time)
    max_back = runner.config.MODEL.CONTEXT_BACKWARD * runner.config.MODEL.NUM_LAYERS

plt.title(f"{variant} {ckpt}")
# plot_ax_attention(plt.gca(), attn_weights, use_average=False, trial=10)
plot_ax_attention(plt.gca(), attn_list[1], use_average=True, trial=10)
plot_ax_attention(plt.gca(), attn_list[0], use_average=True, trial=1)

def plot_avg(layer=0):
    plot_ax_attention(plt.gca(), attn_list[layer], use_average=True, trial=10)
    plt.title(f"{variant} Layer {layer}")
# plot_avg(5)

#%%
num_plots = len(attn_list)
fig = plt.figure(figsize=(8, 5))
trial = 10
for i in range(num_plots):
    rows = num_plots // 3
    ax = fig.add_subplot(rows * 100 + 31 + i)
    ax.set_title(f"Layer {i}")
    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)
    plot_ax_attention(ax, attn_list[i], use_average=False, trial=trial)
fig.tight_layout(rect=[0.06, 0.06, 1, 0.95])
fig.text(0.5, 0.04, 'Src-Attended', ha='center')
fig.text(0.04, 0.5, 'Target-Attender', va='center', rotation='vertical')
plt.suptitle(f"{variant} | ckpt {ckpt} trial {trial} | R2: {r2:.3f}")

#%%
per_neuron_r2s = []
for i in range(pred_rates.size(-1)):
    neuron_r2 = runner.neuron_r2(rates[..., i], pred_rates[..., i], flatten=True)
    per_neuron_r2s.append(neuron_r2)
plt.title("R2 v Neuron")
plt.plot(np.arange(num_neurons), per_neuron_r2s)
#%%
plt.style.use('seaborn-muted')
spine_alpha = 0.3
plt.gca().spines['right'].set_alpha(spine_alpha)
plt.gca().spines['bottom'].set_alpha(spine_alpha)
plt.gca().spines['left'].set_alpha(spine_alpha)
plt.gca().spines['top'].set_alpha(spine_alpha)
plt.grid(alpha=0.25)
per_time_r2s = []
for i in range(time):
    neuron_r2 = runner.neuron_r2(rates[:, i], pred_rates[:, i])
    per_time_r2s.append(neuron_r2)
plt.title("R2 v Time Fwd: 0")
plt.plot(np.arange(time), per_time_r2s)
#%%
per_trial_r2s = []
for i in range(trials):
    neuron_r2 = runner.neuron_r2(rates[i], pred_rates[i])
    per_trial_r2s.append(neuron_r2)
plt.plot(np.arange(trials), per_trial_r2s)
plt.title("R2 v Trial")
#%%s
# * Plot firing rates for selected neurons
np.random.seed(3)
fig = plt.figure()
NUM_PLOTS = 9

neuron_idx = 0
for i in range(NUM_PLOTS):
    neuron_idx = np.random.randint(0, rates.size(-1))
    # trial_idx = i+108
    # trial_idx = np.random.randint(0, rates.size(0))
    trial_idx = 0
    true_trial_rates = rates[trial_idx, :, neuron_idx].cpu()
    pred_trial_rates = pred_rates[trial_idx, :, neuron_idx].cpu()
    if runner.config.MODEL.LOGRATE:
        true_trial_rates = true_trial_rates.exp()
        pred_trial_rates = pred_trial_rates.exp()
    ax = fig.add_subplot(331 + i)
    ax.plot(np.arange(time), true_trial_rates.numpy(), label="GT")
    ax.plot(np.arange(time), pred_trial_rates.numpy(), label="Pred")
    ax.set_title(f"T {trial_idx}, N {neuron_idx}")
plt.suptitle(f"{variant} {ckpt}", y=1.05)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
# prep_plt()
plt.show()

#%%
from src.dataset import SpikesDataset, DATASET_MODES

train_set = SpikesDataset(runner.config, runner.config.DATA.TRAIN_FILENAME, mode=DATASET_MODES.train, logger=runner.logger)
train_rates = train_set.get_dataset()[1]
full_rates = torch.cat([train_rates, rates.cpu()], dim=0)
# (rates[5] == rates[10]).all()
# (rates[311] == ratses[308]).all()
# (rates[0] == rates[1]).all()
#%%
print(full_rates.size())
unique, counts = full_rates.unique(dim=0, return_counts=True)
print(unique)
print(counts)
# print(len(full_rates[:, 0, 0].unique()))
# print(len(full_rates[:, 1, 1].unique()))
# print(len(rates[:, 0, 0].unique()))
# print(len(train_rates[:, 0, 0].unique()))
#%%
neuron_idx = np.random.randint(0, rates.size(-1))
neuron_idx = 148
# trial_idx = i+108
trial_idx = np.random.randint(0, rates.size(0))
trial_idx = 12
# trial_idx = 0
# true_trial_rates = rates[trial_idx, :, neuron_idx].cpu()
pred_trial_rates = pred_rates[trial_idx, :, neuron_idx].cpu()
if runner.config.MODEL.LOGRATE:
    # true_trial_rates = true_trial_rates.exp()
    pred_trial_rates = pred_trial_rates.exp()
ax = plt.gca()
# ax.plot(np.arange(time), true_trial_rates.numpy(), label="GT")
ax.plot(np.arange(time), pred_trial_rates.numpy(), label="Pred")
ax.set_title(f"T {trial_idx}, N {neuron_idx}")

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

plt.figure(figsize=(6, 4))

prep_plt()

# np.random.seed(5)

# plt.text("NDT Predictions", color=palette[0])
# plt.text("Ground Truth", color=palette[2])
dataset_name = variant.split('-')[0]
def plot_axis(
    spikes,
    true_rates,
    ndt_rates, # ndt
    ax,
    key="lorenz",
    seed=3
):
    np.random.seed(seed)
    neuron_idx = np.random.randint(0, rates.size(-1))
    trial_idx = np.random.randint(0, rates.size(0))
    true_trial_rates = true_rates[trial_idx, :, neuron_idx].cpu()
    pred_trial_rates = pred_rates[trial_idx, :, neuron_idx].cpu()
    trial_spikes = spikes[trial_idx, :, neuron_idx].cpu()
    if runner.config.MODEL.LOGRATE:
        true_trial_rates = true_trial_rates.exp()
        pred_trial_rates = pred_trial_rates.exp()

    ax.plot(np.arange(time), pred_trial_rates.numpy(), color=palette[0], label="NDT Predicted")

    lfads_preds = np.load(f"/snel/home/joely/data/{dataset_name}.npy")
    lfads_pred_rates = lfads_preds[trial_idx, :, neuron_idx]
    ax.plot(np.arange(time), lfads_pred_rates, color=palette[1], label="LFADS Predicted")

    ax.plot(np.arange(time), true_trial_rates.numpy(), color='#33333388', label="Ground Truth") #, linestyle="dashed")

    ax.plot(np.arange(time), trial_spikes.numpy(), color=palette[2], label="Spikes", linestyle="dashed")
    ax.set_xticks([])
    # plt.xticks(np.linspace(5, 5, 1), labels=np.linspace(0.1, 0.1, 1))
    ax.set_yticks(np.linspace(0, 0.5, 2))
    ax.plot([0, 0], [0, 0.5], 'k-', lw=3)
    # ax.plot([0, 5], [0, 0], 'k-', lw=3)
    ax.set_title(f"Trial {trial_idx}, Neuron {neuron_idx}")


f, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8, 8))

plot_axis(
    spikes,
    rates,
    pred_rates,
    axes[0, 0],
    seed=3,
    key="chaotic"
)
prep_plt(axes[0,0])
plot_axis(
    spikes,
    rates,
    pred_rates,
    axes[0, 1],
    seed=7,
    key="chaotic"
)
prep_plt(axes[0,1])

#%%
# Plot input spikes for selected trial/neurons
np.random.seed(9)
fig = plt.figure()
NUM_PLOTS = 9
t = rates.size(1)

neuron_idx = 8
for i in range(NUM_PLOTS):
    # neuron_idx = np.random.randint(0, rates.size(-1))
    trial_idx = np.random.randint(0, rates.size(0))
    trial_spikes = spikes[trial_idx, :, neuron_idx].cpu()
    trial_rate = pred_rates[trial_idx, :, neuron_idx].cpu()
    if runner.config.MODEL.LOGRATE:
        trial_rate = trial_rate.exp()
    ax = fig.add_subplot(331 + i)
    ax.plot(np.arange(t), trial_spikes, label="spike")
    ax.plot(np.arange(t), trial_rate, label="rate")
    ax.set_title(f"Trial {trial_idx}, Neuron {neuron_idx}")
plt.suptitle(f"{variant} {ckpt}", y=1.05)
fig.tight_layout()
plt.legend()
plt.show()

#%%
np.random.seed(9)
fig = plt.figure()
NUM_PLOTS = 1
t = rates.size(1)

neuron_idx = 8
neuron_idx = 2
trial_idx = np.random.randint(0, rates.size(0))
trial_spikes = spikes[trial_idx, :, neuron_idx].cpu()
trial_rate = pred_rates[trial_idx, :, neuron_idx].cpu()
if runner.config.MODEL.LOGRATE:
    trial_rate = trial_rate.exp()
ax = plt.gca()
ax.plot(np.arange(t), trial_spikes, label="spike")
ax.plot(np.arange(t), trial_rate, label="rate")
ax.set_title(f"Trial {trial_idx}, Neuron {neuron_idx}")

plt.suptitle(f"{variant} {ckpt}", y=1.05)
fig.tight_layout()
plt.legend()
plt.show()

