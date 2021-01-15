# Src: Andrew's tune_tf2

import os
import os.path as osp
import numpy as np
import ray
from ray import tune
from yacs.config import CfgNode as CN
import torch

from src.config.default import get_cfg_defaults, unflatten
from src.runner import Runner

class tuneNDT(tune.Trainable):
    """ A wrapper class that allows `tune` to interface with NDT.
    """

    def setup(self, config):
        yacs_cfg = self.convert_tune_cfg(config)
        self.epochs_per_generation = yacs_cfg.TRAIN.TUNE_EPOCHS_PER_GENERATION
        self.warmup_epochs = yacs_cfg.TRAIN.TUNE_WARMUP
        self.runner = Runner(config=yacs_cfg)
        self.runner.load_device()
        self.runner.load_train_val_data_and_masker()
        num_hidden = self.runner.setup_model(self.runner.device)
        self.runner.load_optimizer(num_hidden)

    def step(self):
        num_epochs = self.epochs_per_generation
        # the first generation always completes ramping (warmup)
        if self.runner.count_updates < self.warmup_epochs:
            num_epochs += self.warmup_epochs
        for i in range(num_epochs):
            metrics = self.runner.train_epoch()
        return metrics

    def save_checkpoint(self, tmp_ckpt_dir):
        path = osp.join(tmp_ckpt_dir, f"{self.runner.config.VARIANT}.{self.runner.count_checkpoints}.pth")
        self.runner.save_checkpoint(path)
        return path

    def load_checkpoint(self, path):
        self.runner.load_checkpoint(path)

    def reset_config(self, new_config):
        new_cfg_node = self.convert_tune_cfg(new_config)
        self.runner.update_config(new_cfg_node)
        return True

    def convert_tune_cfg(self, flat_cfg_dict):
        """Converts the tune config dictionary into a CfgNode for LFADS.
        """
        cfg_node = get_cfg_defaults()

        flat_cfg_dict['CHECKPOINT_DIR'] = osp.join(self.logdir, 'ckpts')
        flat_cfg_dict['TENSORBOARD_DIR'] = osp.join(self.logdir, 'tb')
        flat_cfg_dict['LOG_DIR'] = osp.join(self.logdir, 'logs')
        flat_cfg_dict['TRAIN.TUNE_MODE'] = True
        cfg_update = CN(unflatten(flat_cfg_dict))
        cfg_node.merge_from_other_cfg(cfg_update)

        return cfg_node