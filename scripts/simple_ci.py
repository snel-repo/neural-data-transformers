# Calculate some confidence intervals

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

grid = True
ckpt = "Grid"
best_model = "best_model"
best_model = "best_model_unmasked"
lve = "lfve" if "unmasked" in best_model else "lve"
r2s = []
mnlls = []
for i in range(3):
    variant = f"lorenz-s{i+1}"
    # variant = f"lorenz_lite-s{i+1}"
    # variant = f"chaotic-s{i+1}"
    # variant = f"chaotic_lite-s{i+1}"
    ckpt_path = f"/snel/home/joely/ray_results/ndt/gridsearch/{variant}/{best_model}/ckpts/{variant}.{lve}.pth"
    runner, spikes, rates = init_by_ckpt(ckpt_path, mode=DATASET_MODES.val)

    # print(runner.config.MODEL.CONTEXT_FORWARD)
    # print(runner.config.MODEL.CONTEXT_BACKWARD)
    # print(runner.config.TRAIN.MASK_MAX_SPAN)
    (
        unmasked_loss,
        pred_rates,
        layer_outputs,
        attn_weights,
        attn_list,
    ) = runner.model(spikes, mask_labels=spikes, return_weights=True)
    print(f"Best Unmasked Val: {runner.best_unmasked_val}") # .37763
    print(f"Best Masked Val: {runner.best_val}") # `  best val is .380 at 1302...
    mnlls.append(runner.best_val['value'])
    print(f"Best R2: {runner.best_R2}") # best val is .300 at 1302...

    print(f"Unmasked: {unmasked_loss}")

    if "maze" not in variant:
        r2 = runner.neuron_r2(rates, pred_rates, flatten=True)
        # r2 = runner.neuron_r2(rates, pred_rates)
        r2s.append(r2)
        vaf = runner.neuron_vaf(rates, pred_rates)
        print(f"R2:\t{r2}, VAF:\t{vaf}")

    trials, time, num_neurons = rates.size()
    print(runner.count_updates)
    print(runner.config.MODEL.EMBED_DIM)

#%%
import math
# print(sum(mnlls) / 3)
# print(mnlls)
def print_ci(r2s):
    r2s = np.array(r2s)
    mean = r2s.mean()
    ci = r2s.std() * 1.96 / math.sqrt(3)
    print(f"{mean:.3f} \pm {ci:.3f}")
print_ci(r2s)
# print_ci([0.2079, 0.2077, 0.2091])
# print_ci([
#     0.9225, 0.9113, 0.9183
# ])
# print_ci([0.8712, 0.8687, 0.8664])
# print_ci([0.9255, 0.9271, 0.9096])
# print_ci([.924, .914, .9174])
# print_ci([.0496, .0095, .0054])
# print_ci([.4003, .0382, .4014])
# print_ci([.52, .4469, .6242])
print_ci([.416, .0388, .4221])
print_ci([.0516, .0074, .0062])
print_ci([
    0.5184, 0.4567, 0.6724
])

#%%
# Port for rds analysis
if "maze" in variant:
    rate_output_pth = f"/snel/share/joel/ndt_rates/"
    rate_output_pth = osp.join(rate_output_pth, "grid" if grid else "pbt")
    rate_output_fn = f"{variant}.h5"
    pred_rates, layer_outputs = runner.get_rates(
        checkpoint_path=ckpt_path,
        save_path=osp.join(rate_output_pth, rate_output_fn)
    )
    trials, time, num_neurons = pred_rates.size()


#%%

(
    unmasked_loss,
    pred_rates,
    layer_outputs,
    attn_weights,
    attn_list,
) = runner.model(spikes, mask_labels=spikes, return_weights=True)
print(f"Unmasked: {unmasked_loss}")

if "maze" not in variant:
    r2 = runner.neuron_r2(rates, pred_rates)
    vaf = runner.neuron_vaf(rates, pred_rates)
    print(f"R2:\t{r2}, VAF:\t{vaf}")

trials, time, num_neurons = rates.size()
print(f"Best Masked Val: {runner.best_val}") # `  best val is .380 at 1302...
print(f"Best Unmasked Val: {runner.best_unmasked_val}") # .37763
print(f"Best R2: {runner.best_R2}") # best val is .300 at 1302...
print(runner.count_updates)
