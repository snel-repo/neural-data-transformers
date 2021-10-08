# Run with RDS dev, pyglmnet, and tqdm
import h5py
import logging
import multiprocessing
#from tqdm import tqdm
import pickle as pkl
from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from scipy import stats
import pdb

from rds.decoding import prepare_decoding_data

#from gpfa_utils import resample_and_merge_gpfa

logging.basicConfig(level=logging.INFO)

RUN_NAME = "sr02"
#VARIANT = "m700_sbtt"
VARIANT = "maze_sbtt"

#DATA_HOME = '/snel/share/share/derived/selective_backprop/real/maze/bandwidth_ratio'
DATA_HOME = '/snel/share/share/derived/selective_backprop/runs/ndt/data'
RUN_HOME = f"/snel/share/share/derived/selective_backprop/runs/ndt/{RUN_NAME}/"

# Load the fully observed data and extract important variables
full_obs_file = path.join(DATA_HOME, 'lfads_full_obs.h5')
with h5py.File(full_obs_file, 'r') as hf:
  full_data_dict = {k: np.array(v) for k, v in hf.items()}
array_lookup = full_data_dict['array_lookup']
N = full_data_dict['train_data'].shape[-1]

# load the saved rates h5
rates_file_name = f"{VARIANT}.h5"
rates_file = path.join(RUN_HOME, rates_file_name)
with h5py.File(rates_file, 'r') as hf:
  rates_dict = {k: np.array(v) for k, v in hf.items()}

# Select set of heldout neurons, fixed across entire experiment
# heldin_ixs, heldout_ixs = train_test_split(
#   np.arange(N), test_size=100, random_state=0, stratify=array_lookup)
heldin_ixs, heldout_ixs = train_test_split(
  np.arange(N), test_size=50, random_state=0, stratify=array_lookup)

# Load the trial_data and interface for decoding
def load_from_data_dir(fname):
    fpath = path.join(DATA_HOME, fname)
    with open(fpath, 'rb') as file:
        obj = pkl.load(file)
    return obj
trial_data = load_from_data_dir('trial_data.pkl')
interface = load_from_data_dir('interface.pkl')

#
# Tell the interface to only load factors
interface.merge_fields_map = {'rates': 'rates'}
rates_df = interface.merge(rates_dict)
if 'trial_id' in trial_data:
    # Temporarily use clock_time as the index
    concat_df = trial_data.set_index('clock_time')
    full_df = pd.concat([concat_df, rates_df], axis=1)
    full_df = full_df.reset_index()
else:
    full_df = pd.concat([trial_data, rates_df], axis=1)

(x_train, y_train, groups_train), (x_valid, y_valid, groups_valid) = \
    prepare_decoding_data(
        full_df, 'rates', 'kin_v', valid_ratio=0.5, ms_lag=80, return_groups=True)
# Train and evaluate a decoder
print("Training and evaluating decoder")
#decoder = LinearRegression().fit(x_train, y_train)

# Regularization turns out not to be important here
decoder = GridSearchCV(
    estimator=Ridge(),
    param_grid={'alpha': np.logspace(-5, 4, base=10, num=100)},
    scoring='r2',
    n_jobs=-1,
    cv=GroupKFold(n_splits=5),
).fit(x_train, y_train, groups=groups_train)

# Compute decoder accuracy
y_pred = decoder.predict(x_train)
decode_train_r2 = r2_score(y_train, y_pred)
y_pred = decoder.predict(x_valid)
decode_valid_r2 = r2_score(y_valid, y_pred)
pdb.set_trace()
