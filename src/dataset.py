#!/usr/bin/env python3
# Author: Joel Ye
import os.path as osp

import h5py
import numpy as np
import torch
from torch.utils import data

from src.utils import merge_train_valid

class DATASET_MODES:
    train = "train"
    val = "val"
    test = "test"
    trainval = "trainval"

class SpikesDataset(data.Dataset):
    r"""
        Dataset for single file of spike times (loads into memory)
        Lorenz data is NxTxH (H being number of neurons) - we load to T x N x H
    """

    def __init__(self, config, filename, mode=DATASET_MODES.train, logger=None):
        r"""
            args:
                config: dataset config
                filename: excluding path
                mode: used to extract the right indices from LFADS h5 data
        """
        super().__init__()
        self.logger = logger
        if self.logger is not None:
            self.logger.info(f"Loading {filename} in {mode}")
        self.config = config.DATA
        self.use_lograte = config.MODEL.LOGRATE
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.datapath = osp.join(config.DATA.DATAPATH, filename)
        split_path = self.datapath.split(".")

        self.has_rates = False
        if len(split_path) == 1 or split_path[-1] == "h5":
            spikes, rates = self.get_data_from_h5(mode, self.datapath)

            spikes = torch.tensor(spikes).long()
            if rates is not None:
                rates = torch.tensor(rates)
        elif split_path[-1] == "pth":
            dataset_dict = torch.load(self.datapath)
            spikes = dataset_dict["spikes"]
            if "rates" in dataset_dict:
                self.has_rates = True
                rates = dataset_dict["rates"]
        else:
            raise Exception(f"Unknown dataset extension {ext}")

        self.num_trials, _, self.num_neurons = spikes.size()
        self.trial_length = config.MODEL.TRIAL_LENGTH if config.MODEL.TRIAL_LENGTH > 0 else spikes.size(1)
        self.spikes = self.batchify(spikes)
        # Fake rates so we can skip None checks everywhere. Use `self.has_rates` when desired
        self.rates = self.batchify(rates) if self.has_rates else torch.zeros_like(spikes)

        if config.DATA.OVERFIT_TEST:
            if self.logger is not None:
                self.logger.warning("Overfitting..")
            self.spikes = self.spikes[:2]
            self.rates = self.rates[:2]
            self.num_trials = 2
            # self.spikes = torch.ones_like(self.spikes) * 5 # Yes, this works...
            # self.spikes = self.spikes[:10]
            # self.rates = self.rates[:10]
        elif hasattr(config.DATA, "RANDOM_SUBSET_TRIALS") and config.DATA.RANDOM_SUBSET_TRIALS < 1.0 and mode == DATASET_MODES.train:
            if self.logger is not None:
                self.logger.warning(f"!!!!! Training on {config.DATA.RANDOM_SUBSET_TRIALS} of the data with seed {config.SEED}.")
            reduced = int(self.num_trials * config.DATA.RANDOM_SUBSET_TRIALS)
            torch.random.manual_seed(config.SEED)
            random_subset = torch.randperm(self.num_trials)[:reduced]
            self.num_trials = reduced
            self.spikes = self.spikes[random_subset]
            self.rates = self.rates[random_subset]

    def batchify(self, x):
        r"""
            Chops data into uniform sizes as configured by trial_length.

            Returns:
                x reshaped as num_samples x trial_length x neurons
        """
        trial_time = x.size(1)
        samples_per_trial = trial_time // self.trial_length
        if trial_time % self.trial_length != 0:
            if self.logger is not None:
                self.logger.debug(f"Trimming dangling trial info. Data trial length {trial_time} \
                is not divisible by asked length {self.trial_length})")
        x = x.narrow(1, 0, samples_per_trial * self.trial_length)

        # ! P sure this can be a view op
        # num_samples x trial_length x neurons
        return torch.cat(torch.split(x, self.trial_length, dim=1), dim=0)

    def get_num_neurons(self):
        return self.num_neurons

    def __len__(self):
        return self.spikes.size(0)

    def __getitem__(self, index):
        r"""
            Return spikes and rates, shaped T x N (num_neurons)
        """
        return self.spikes[index], None if self.rates is None else self.rates[index]

    def get_dataset(self):
        return self.spikes, self.rates

    def get_max_spikes(self):
        return self.spikes.max().item()

    def get_num_batches(self):
        return self.spikes.size(0) // self.batch_size

    def clip_spikes(self, max_val):
        self.spikes = torch.clamp(self.spikes, max=max_val)

    def get_data_from_h5(self, mode, filepath):
        r"""
            Data being spikes + rates if available (returns None if unavailable)
        """
        with h5py.File(filepath, 'r') as h5file:
            h5dict = {key: h5file[key][()] for key in h5file.keys()}
            train_data = h5dict['train_data'].astype(np.float32).squeeze()
            valid_data = h5dict['valid_data'].astype(np.float32).squeeze()
            train_rates = None
            valid_rates = None
            if "train_truth" and "valid_truth" in h5dict:
                self.has_rates = True
                train_rates = h5dict['train_truth'].astype(np.float32)
                valid_rates = h5dict['valid_truth'].astype(np.float32)
                train_rates = train_rates / h5dict['conversion_factor']
                valid_rates = valid_rates / h5dict['conversion_factor']
                if self.use_lograte:
                    train_rates = torch.log(torch.tensor(train_rates) + self.config.LOG_EPSILON)
                    valid_rates = torch.log(torch.tensor(valid_rates) + self.config.LOG_EPSILON)
        if mode == DATASET_MODES.train:
            return train_data, train_rates
        elif mode == DATASET_MODES.val:
            return valid_data, valid_rates
        elif mode == DATASET_MODES.trainval:
            # merge training and validation data
            if 'train_inds' in h5dict and 'valid_inds' in h5dict:
                # if there are index labels, use them to reassemble full data
                train_inds = h5dict['train_inds'].squeeze()
                valid_inds = h5dict['valid_inds'].squeeze()
                file_data = merge_train_valid(
                    train_data, valid_data, train_inds, valid_inds)
                if self.has_rates:
                    merged_rates = merge_train_valid(
                        train_rates, valid_rates, train_inds, valid_inds
                    )
            else:
                if self.logger is not None:
                    self.logger.info("No indices found for merge. "
                    "Concatenating training and validation samples.")
                file_data = np.concatenate([train_data, valid_data], axis=0)
                if self.has_rates:
                    merged_rates = np.concatenate([train_rates, valid_rates], axis=0)
            return file_data, merged_rates if self.has_rates else None
        else: # test unsupported
            return None, None