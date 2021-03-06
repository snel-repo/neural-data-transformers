# ! Reference script. Not supported out of the box.

#%%
# param vs nll (3b) and match to vaf vs nll (4b)
# 3b requires a CSV downloaded from TB Hparams page (automatically generated by Ray)
# 4b requires saving down of rate predictions from each model (for NDT, this step is in `record_all_rates.py`)

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
import pandas as pd
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
# Extract the NLL and R2
plot_path = Path('~/projects/transformer-modeling/scripts/hparams.csv').expanduser().resolve()
df = pd.read_csv(plot_path)

#%%
SMALL_SIZE = 12
MEDIUM_SIZE = 20
LARGE_SIZE = 24

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
    plt.gca().spines['bottom'].set_alpha(spine_alpha)
    # plt.gca().spines['bottom'].set_alpha(0)
    plt.gca().spines['left'].set_alpha(spine_alpha)
    # plt.gca().spines['left'].set_alpha(0)
    plt.gca().spines['top'].set_alpha(0.0)

    plt.tight_layout()

plt.figure(figsize=(4, 6))
# plt.figure(figsize=(6, 4))
prep_plt()
# So. I select the model with the least all-time MVE, and report metrics on that MVE ckpt.
# Here I'm reporting numbers from the final ckpt instead.

x_axis = 'unmasked_loss'
# x_axis = 'masked_loss'
sns.scatterplot(data=df, x=f"ray/tune/{x_axis}", y="ray/tune/r2", s=120)
# sns.scatterplot(data=df, x="ray/tune/best_masked_loss", y="ray/tune/r2", s=120)
plt.yticks(np.arange(0.8, 0.96, 0.15))
# plt.yticks(np.arange(0.8, 0.96, 0.05))
plt.xticks([0.37, 0.39])
plt.xlim(0.365, 0.39)
plt.xlabel("Match to Spikes (NLL)")
plt.ylabel("Rate Prediction $R^2$", labelpad=-20)
from scipy.stats import pearsonr
r = pearsonr(df[f'ray/tune/{x_axis}'], df['ray/tune/r2'])
print(r)
plt.text(0.378, 0.92, f"$\it{{\\rho}}$ : {r[0]:.3f}", size=LARGE_SIZE)

plt.savefig("3_match_spikes.pdf", bbox_inches="tight")
#%%
palette = sns.color_palette(palette='muted', n_colors=2, desat=0.9)
variant = 'm700_2296-s1'
nlls = torch.load(f'{variant}_val_errs_sweep.pth')
matches = torch.load(f'/snel/home/joely/projects/rds/{variant}_psth_match_sweep.pth')
decoding = torch.load(f'/snel/home/joely/projects/rds/{variant}_deocding_sweep.pth')
nll_arr = []
match_arr = []
decoding_arr = []
for key in nlls:
    nll_arr.append(nlls[key])
    match_arr.append(matches[key])
    decoding_arr.append(decoding[key])
plt.figure(figsize=(4, 5))
prep_plt()
plt.scatter(nll_arr, match_arr, color = palette[0])
plt.xticks([0.139, 0.144], rotation=20)
plt.xlim(0.139, 0.144)
plt.yticks([0.55, 0.75])
# plt.yticks(np.linspace(0.5, 0.75, 2))
plt.xlabel("Match to Spikes (NLL)", labelpad=0, fontsize=MEDIUM_SIZE)
plt.ylabel("Match to Empirical PSTH ($R^2$)", labelpad=0, fontsize=MEDIUM_SIZE)
r = pearsonr(nll_arr, match_arr)
print(r)

plt.text(0.1392, 0.53, f"$\it{{\\rho}}$ : {r[0]:.3f}", size=LARGE_SIZE, color=palette[0])
# plt.text(0.141, 0.72, f"$\it{{\\rho}}$ : {r[0]:.3f}", size=LARGE_SIZE)
plt.hlines(0.7078, 0.139, 0.144, linestyles="--", color=palette[1])
plt.text(0.141, 0.715, f"AutoLFADS", size=MEDIUM_SIZE, color=palette[1])
# plt.text(0.142, 0.715, f"LFADS", size=MEDIUM_SIZE, color=palette[1])

plt.savefig("4b_match_psth.pdf", bbox_inches="tight")
#%%
plt.figure(figsize=(4, 5))
prep_plt()
plt.scatter(nll_arr, decoding_arr)
plt.xticks([0.139, 0.144], rotation=20)
plt.xlim(0.139, 0.144)

#%%
plt.figure(figsize=(4, 5))
prep_plt()
plt.scatter(match_arr, decoding_arr)
# plt.xticks([0.139, 0.144], rotation=20)
# plt.xlim(0.139, 0.144)
