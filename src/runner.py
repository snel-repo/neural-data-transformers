#!/usr/bin/env python3
# Author: Joel Ye

import os
import os.path as osp

import time
from typing import Any, Dict, List, Optional

from sklearn.metrics import r2_score, explained_variance_score
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_transformers import AdamW, WarmupCosineSchedule
from torch.utils import data

from src import (
    get_model_class, is_learning_model, is_input_masked_model,
    TensorboardWriter,
    create_logger,
)
from src.utils import get_inverse_sqrt_schedule
from src.dataset import DATASET_MODES, SpikesDataset
from src.mask import Masker, UNMASKED_LABEL, DEFAULT_MASK_VAL

"""
Runner class for NDT
"""

def get_lightest_gpus(num_gpus):
    # TODO update with better CUDA_VISIBLE_DEVICES support (or just use ray)
    if torch.cuda.device_count() == 1:
        return [0]
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argsort(memory_available)[-num_gpus:].tolist()

def exp_smooth(new_metric, old_metric, mu=0.5):
    r""" Higher mu is smoother """
    return (1.0 - mu) * new_metric + mu * old_metric

def exp_smooth_dict(new_metrics, rolling_metrics, mu=0.5):
    for m in new_metrics:
        if m in rolling_metrics:
            rolling_metrics[m] = exp_smooth(new_metrics[m], rolling_metrics[m], mu)

class Runner:
    r"""
        Two paths to inference.
        A:
            Have a config file.
            Load device.
            Load a checkpoint.
        B:
            Pass a checkpoint path (other steps automated)
        We prioritize path A.
    """
    def __init__(self, config=None, checkpoint_path=None):
        assert config is not None or checkpoint_path is not None
        self.flush_secs = 10
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.device = None
        self.num_neurons = 0
        self.pth_time = 0
        self.count_updates = 0
        self.count_checkpoints = 0
        self.num_gpus = 0
        self.masker = None
        self.rolling_metrics = {} # For PBT

        if checkpoint_path is not None:
            ckpt_dict = torch.load(checkpoint_path)
            config = ckpt_dict["config"]
        self.config = config
        if not osp.exists(config.LOG_DIR):
            os.makedirs(config.LOG_DIR, exist_ok=True)
        logfile_path = osp.join(config.LOG_DIR, f"{config.VARIANT}.log")
        # if osp.exists(logfile_path):
        #     os.remove(logfile_path)
        self.logger = create_logger()
        self.logger.clear_filehandlers()
        self.logger.add_filehandler(logfile_path)
        if hasattr(config.TRAIN, "TUNE_MODE") and config.TRAIN.TUNE_MODE:
            self.logger.clear_streamhandlers()

        self.best_val = {
            "value": 100,
            "update": -1,
        }
        self.best_unmasked_val = {
            "value": 100,
            "update": -1,
        }
        self.best_R2 = {
            "value": -100,
            "update": -1,
        }

        if checkpoint_path is not None:
            self.load_device()
            self.load_checkpoint(checkpoint_path)

    def setup_model(self, device):
        r""" Creates model and assigns to device """
        self.model = get_model_class(self.config.MODEL.NAME)(
            self.config.MODEL,
            self.trial_length,
            self.num_neurons,
            device,
            max_spikes=self.max_spikes
        )
        num_hidden = self.model.get_hidden_size()
        if self.num_gpus > 1:
            if self.config.SYSTEM.GPU_AUTO_ASSIGN:
                gpu_indices = get_lightest_gpus(self.num_gpus)
            else:
                gpu_indices = list(range(self.num_gpus))
            if self.device_gpu in gpu_indices:
                gpu_indices.remove(self.device_gpu)
            else:
                gpu_indices = gpu_indices[:-1]
            gpu_indices = [self.device_gpu] + gpu_indices # Make sure our primary gpu is first
            self.model = nn.DataParallel(self.model, device_ids=gpu_indices)
        self.model = self.model.to(device)
        return num_hidden

    def _get_parameters(self):
        return list(self.model.parameters())

    def _do_log(self, update):
        return (
            update > 0 and update % self.config.TRAIN.LOG_INTERVAL == 0
        )

    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "optim_state": None if self.optimizer is None else self.optimizer.state_dict(),
            "lr_scheduler": None if self.lr_scheduler is None else self.lr_scheduler.state_dict(),
            "config": self.config,
            "best_val": self.best_val,
            "best_unmasked_val": self.best_unmasked_val,
            "best_r2": self.best_R2,
            "max_spikes": self.max_spikes,
            "num_neurons": self.num_neurons,
            "trial_length": self.trial_length,
        }
        checkpoint["extra_state"] = dict( # metadata
            update=self.count_updates,
            checkpoint=self.count_checkpoints,
            pth_time=self.pth_time,
            max_spikes=self.max_spikes
        )

        if extra_state is not None:
            checkpoint["extra_state"].update(extra_state)

        if len(osp.split(file_name)[0]) > 0:
            full_path = file_name
        else:
            os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
            full_path = osp.join(self.config.CHECKPOINT_DIR, file_name)
        #self.logger.info("Saving {} with val {}, dropout {}. Decoder weights: {}".format(
        #     full_path,
        #     self.best_val,
        #     self.config.MODEL.DROPOUT,
        #     self.model.state_dict()['decoder.0.bias'][:5]
        # ))
        torch.save(
            checkpoint, full_path
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.
        Will fully load model if not already configured. Expects runner devices to be set.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        ckpt_dict = torch.load(checkpoint_path, *args, **kwargs)
        if "num_neurons" in ckpt_dict:
            self.num_neurons = ckpt_dict["num_neurons"]
        if "trial_length" in ckpt_dict:
            self.trial_length = ckpt_dict["trial_length"]
        if "max_spikes" in ckpt_dict:
            self.max_spikes = ckpt_dict["max_spikes"]
        if self.model is None:
            self.setup_model(self.device)
        self.model.load_state_dict(ckpt_dict["state_dict"])
        if "optim_state" in ckpt_dict and self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt_dict["optim_state"])
        if "lr_scheduler" in ckpt_dict and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(ckpt_dict["lr_scheduler"])
        if "best_val" in ckpt_dict:
            self.best_val = ckpt_dict["best_val"]
        if "best_unmasked_val" in ckpt_dict:
            self.best_unmasked_val = ckpt_dict["best_unmasked_val"]
        if "best_r2" in ckpt_dict:
            self.best_R2 = ckpt_dict["best_r2"]
        if "extra_state" in ckpt_dict:
            self.count_updates = ckpt_dict["extra_state"]["update"]
            self.logger.info("Update loaded -- {}".format(self.count_updates))
            self.count_checkpoints = ckpt_dict["extra_state"]["checkpoint"]
            self.pth_time = ckpt_dict["extra_state"]["pth_time"]
        #self.logger.info("Loading {} with val {}, dropout {}. Decoder weight {}".format(
        #     checkpoint_path,
        #     self.best_val,
        #     self.config.MODEL.DROPOUT,
        #     self.model.state_dict()['decoder.0.bias'][:5]
        # ))
        return ckpt_dict

    def load_device(self):
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.num_gpus = min(self.config.SYSTEM.NUM_GPUS, torch.cuda.device_count())
            self.logger.info(f"Using {self.num_gpus} GPUs")
            gpu_id = self.config.SYSTEM.TORCH_GPU_ID
            if self.config.SYSTEM.GPU_AUTO_ASSIGN:
                gpu_id = get_lightest_gpus(1)[0]
            self.device = (
                torch.device("cuda", gpu_id)
            )
            self.device_gpu = gpu_id

        self.logger.info(f"Using {self.device}")

    def update_config(self, config):
        r""" Update config node and propagate through model. Used for pbt.
        """
        # Diff LR
        #self.logger.info(f"\n\n Updating config! {config.TRAIN.LR.SCHEDULE} \n\n")
        if self.config.TRAIN.LR.INIT != config.TRAIN.LR.INIT and self.optimizer is not None:
            for g in self.optimizer.param_groups:
                g['lr'] = config.TRAIN.LR.INIT # Manualy override of LR
        self.config = config
        if self.masker is not None:
            self.masker.config = config.TRAIN
        self.model.update_config(config.MODEL)

    def load_train_val_data_and_masker(self):
        training_set = SpikesDataset(self.config, self.config.DATA.TRAIN_FILENAME, mode=DATASET_MODES.train, logger=self.logger)
        self.training_generator = data.DataLoader(training_set,
            batch_size=self.config.TRAIN.BATCH_SIZE, shuffle=True
        )
        # We'll need this to embed spikes. Hoping max spikes for val/train isn't too far off
        self.max_spikes = training_set.get_max_spikes() + 3
        self.logger.info(f"Clipping all spikes to {self.max_spikes}.")
        self.logger.info(f"Training on {len(training_set)} samples.")

        if self.config.TRAIN.DO_VAL:
            self.validation_set = SpikesDataset(self.config, self.config.DATA.VAL_FILENAME, mode=DATASET_MODES.val, logger=self.logger)
            self.validation_set.clip_spikes(self.max_spikes)
            # Typically this is small enough
            # validation_generator = data.DataLoader(validation_set,
            #     batch_size=len(validation_set), shuffle=False,
            # )

        self.num_neurons = training_set.get_num_neurons()
        self.trial_length = training_set.trial_length
        self.masker = Masker(self.config.TRAIN, self.device)

    def load_optimizer(self, num_hidden):
        train_cfg = self.config.TRAIN
        if is_learning_model(self.config.MODEL.NAME):
            self.optimizer = AdamW(
                list(filter(lambda p: p.requires_grad, self._get_parameters())),
                lr=train_cfg.LR.INIT,
                weight_decay=train_cfg.WEIGHT_DECAY,
                eps=train_cfg.EPS,
            )

            self.logger.info(
                "number of trainable parameters: {}".format(
                    sum(
                        param.numel()
                        for param in self.model.parameters()
                        if param.requires_grad
                    )
                )
            )

        if self.optimizer is not None and train_cfg.LR.SCHEDULE:
            if train_cfg.LR.SCHEDULER == "cosine":
                self.lr_scheduler = WarmupCosineSchedule(
                    self.optimizer,
                    warmup_steps=train_cfg.LR.WARMUP,
                    t_total=train_cfg.NUM_UPDATES
                )
            else:
                self.lr_scheduler = get_inverse_sqrt_schedule(
                    self.optimizer,
                    warmup_steps=train_cfg.LR.WARMUP,
                    lr_max=train_cfg.LR.INIT
                )

    def train(self, checkpoint_path=None) -> None:
        r"""Main method for training model.

        Args:
            checkpoint_path: path of checkpoint to load
        Returns:
            None
        """
        self.load_device()
        train_cfg = self.config.TRAIN

        self.load_train_val_data_and_masker()
        num_hidden = self.setup_model(self.device)
        self.load_optimizer(num_hidden)

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path, map_location="cpu")

        start_updates = self.count_updates

        for update in range(start_updates, train_cfg.NUM_UPDATES):
            metrics = self.train_epoch()
            if metrics["done"]:
                break
        if not metrics["done"]:
           self.logger.info("Reached max updates without early stopping. Consider training some more.")

        if not train_cfg.TUNE_MODE:
            metrics_dict = {
                "Loss": self.best_val["value"],
                "Unmasked Loss": self.best_unmasked_val["value"],
            }
            if train_cfg.DO_R2:
                metrics_dict.update({ "R2": self.best_R2["value"] })
            with TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            ) as writer:
                writer.add_hparams(self.extract_hps_dict(), metrics_dict)
        torch.cuda.empty_cache()

    def train_epoch(self):
        r"""
            One (PBT) epoch of training. Model and data should be set up and on device at this point.

            Note: LFADS runs an epoch every pass through the data. This may be too frequently for transformers.
            i.e. we may need to do multiple passes through the data. For now, we're changing to report every pass through data.

            Returns:
                metrics: Information about the epoch.
                    "done" -- should stop this run (e.g. due to early stopping). Keyword for Tune PBT.
        """
        if self.training_generator is None:
            raise Exception("No dataset generator set")

        update = self.count_updates
        #self.logger.info(f"update {update}")
        train_cfg = self.config.TRAIN

        expand_prob = min((update - train_cfg.MASK_SPAN_RAMP_START) / (train_cfg.MASK_SPAN_RAMP_END - train_cfg.MASK_SPAN_RAMP_START), 1)

        self.model.train()

        t_start = time.time()
        for spikes, rates, heldout_spikes, forward_spikes in self.training_generator:
            spikes = spikes.to(self.device)
            rates = rates.to(self.device) if self.config.MODEL.REQUIRES_RATES else None
            if self.training_generator.dataset.has_heldout:
                heldout_spikes = heldout_spikes.to(self.device)
            else:
                heldout_spikes = None
            if self.training_generator.dataset.has_forward:
                forward_spikes = forward_spikes.to(self.device)
            else:
                forward_spikes = None
            masked_spikes, labels = self.masker.mask_batch(
                spikes,
                max_spikes=self.max_spikes,
                should_mask=is_input_masked_model(self.config.MODEL.NAME),
                expand_prob=expand_prob,
                heldout_spikes=heldout_spikes,
                forward_spikes=forward_spikes
            )
            mlm_loss, _, layer_outputs, *_ = self.model(
                masked_spikes,
                mask_labels=labels,
                rates=rates,
                return_outputs=False,
            )
            loss = mlm_loss.mean()

            if self.optimizer is not None:

                self.optimizer.zero_grad()
                loss.backward()
                params = self._get_parameters()

                nn.utils.clip_grad_norm_(
                    params, train_cfg.MAX_GRAD_NORM
                )
                self.optimizer.step()

        self.pth_time += time.time() - t_start
        self.count_updates += 1
        update = self.count_updates

        if self.optimizer is not None and train_cfg.LR.SCHEDULE:
            self.lr_scheduler.step()

        if self._do_log(update):
            # * Note we're only logging the loss of the last train step
            with TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            ) as writer:
                if self.optimizer is not None and train_cfg.LR.SCHEDULE:
                    writer.add_scalar("lr", self.lr_scheduler.get_last_lr()[0])
                    self.logger.queue_stat("LR", self.lr_scheduler.get_last_lr()[0])
                writer.add_scalar(
                    "loss", # train loss
                    loss,
                    update,
                )

            self.logger.queue_stat("loss", loss.item())

        metrics_dict = dict(
            done = False,
            epoch = self.count_updates,
            # r2 = self.best_r2["value"],
            best_masked_loss = self.best_val["value"] # Tune will reference this value to select best model.
        )

        if (train_cfg.DO_VAL and update % train_cfg.VAL_INTERVAL == 0):
            self.model.eval()
            with torch.no_grad():
                spikes, rates, heldout_spikes, forward_spikes = self.validation_set.get_dataset()
                spikes = spikes.to(self.device)
                rates = rates.to(self.device)
                if self.validation_set.has_heldout:
                    heldout_spikes = heldout_spikes.to(self.device)
                else:
                    heldout_spikes = None
                if self.validation_set.has_forward:
                    forward_spikes = forward_spikes.to(self.device)
                else:
                    forward_spikes = None
                feed_rates = rates if self.config.MODEL.REQUIRES_RATES else None
                masked_spikes, labels = self.masker.mask_batch(
                    spikes,
                    max_spikes=self.max_spikes,
                    should_mask=is_input_masked_model(self.config.MODEL.NAME),
                    heldout_spikes=heldout_spikes,
                    forward_spikes=forward_spikes,
                )

                loss, pred_rates, *_ = self.model(
                    masked_spikes,
                    mask_labels=labels,
                    rates=feed_rates,
                )

                val_loss = loss.mean()

                # no_mask evaluation should still exclude heldout neurons
                if heldout_spikes is not None:
                    spikes = torch.cat([spikes, torch.zeros_like(heldout_spikes)], -1)
                if forward_spikes is not None:
                    spikes = torch.cat([spikes, torch.zeros_like(forward_spikes)], 1)
                no_mask_labels = spikes.clone()
                if heldout_spikes is not None:
                    no_mask_labels[..., -heldout_spikes.size(-1)] = -100 # unmasked_label
                if forward_spikes is not None:
                    no_mask_labels[:, -forward_spikes.size(1):,:] = -100 # unmasked_label
                no_mask_loss, pred_rates, *_ = self.model(
                    spikes,
                    mask_labels=no_mask_labels,
                    passthrough=True,
                    rates=rates
                )

                no_mask_loss = no_mask_loss.mean()

                metrics_dict["unmasked_loss"] = no_mask_loss.item()
                metrics_dict["masked_loss"] = val_loss.item()

                if "smth_masked_loss" not in self.rolling_metrics:
                    self.rolling_metrics["smth_masked_loss"] = metrics_dict["masked_loss"]
                else:
                    self.rolling_metrics["smth_masked_loss"] = exp_smooth(metrics_dict["masked_loss"], self.rolling_metrics["smth_masked_loss"])
                metrics_dict["smth_masked_loss"] = self.rolling_metrics["smth_masked_loss"]

                if self._do_log(update):
                    with TensorboardWriter(
                        self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
                    ) as writer:
                        writer.add_scalar(
                            "val_loss",
                            val_loss,
                            update,
                        )

                        writer.add_scalar(
                            "unmasked_loss",
                            no_mask_loss,
                            update,
                        )
                        self.logger.queue_stat("val loss", val_loss.item())
                        self.logger.queue_stat("unmasked val loss", no_mask_loss.item())
                        if train_cfg.DO_R2 and self.validation_set.has_rates:
                            r2 = self.neuron_r2(rates, pred_rates)
                            writer.add_scalar("r2", r2, update)
                            self.logger.queue_stat("r2", r2)
                            if self.best_R2["value"] < r2:
                                self.best_R2["value"] = r2
                                self.best_R2["update"] = update
                                self.save_checkpoint(f'{self.config.VARIANT}.gr2.pth') # greatest r2
                            metrics_dict["r2"] = r2

                if no_mask_loss.item() < self.best_unmasked_val["value"]:
                    self.logger.info(f"Overwriting best unmasked val {self.best_unmasked_val['value']} from {self.best_unmasked_val['update']} with {no_mask_loss} at {update}.")
                    self.best_unmasked_val["value"] = no_mask_loss.item()
                    self.best_unmasked_val["update"] = update
                    self.save_checkpoint(f'{self.config.VARIANT}.lfve.pth') # full validation

                if val_loss.item() < self.best_val["value"]:
                    self.logger.info(f"Overwriting best val {self.best_val['value']} from {self.best_val['update']} with {val_loss} at {update}.")
                    self.best_val["value"] = val_loss.item()
                    self.best_val["update"] = update
                    self.save_checkpoint(f'{self.config.VARIANT}.lve.pth')

                elif update - self.best_val["update"] > train_cfg.PATIENCE:
                    self.logger.info(f"Val loss has not improved for {train_cfg.PATIENCE} updates. Stopping...")
                    self.logger.info(f"Best val: {self.best_val['value']} at {self.best_val['update']} updates.")
                    if train_cfg.DO_R2 and self.validation_set.has_rates: # log down for hparams
                        self.logger.info(f"Best R2: {self.best_R2['value']} at {self.best_R2['update']}")
                        r2 = self.neuron_r2(rates, pred_rates)
                        metrics_dict["r2"] = r2
                    metrics_dict["done"] = True
                metrics_dict["best_masked_loss"] = self.best_val["value"]

        if self._do_log(update):
            self.logger.log_update(update)
            self.logger.info(
                "update: {}\tpth-time: {:.3f}s\t".format(
                    update, self.pth_time
                )
            )

        if update % train_cfg.CHECKPOINT_INTERVAL == 0 and not train_cfg.TUNE_MODE: # Don't save extra checkpoints when sweeping
            self.save_checkpoint(
                f"{self.config.VARIANT}.{self.count_checkpoints}.pth"
            )
            self.count_checkpoints += 1

        return metrics_dict

    def eval(
        self,
        checkpoint_path: str,
        save_path = ""
    ) -> None:
        # * The evaluation code path has legacy code (and will not run).
        # * Evaluation / analysis is done in analysis scripts.

        r"""Evaluates a single checkpoint.
            Runs masking identical to train and calculates PoissonNLL, R2 on masked neurons.
        Args:
            checkpoint_path: path of checkpoint

        Returns:
            None
        """
        self.logger.info(f"Starting evaluation")

        self.load_device()
        self.masker = Masker(self.config.TRAIN, self.device)

        # Not using a generator atm because we can fit the whole set onto GPU
        test_set = SpikesDataset(self.config, self.config.DATA.TEST_FILENAME, mode="test", logger=self.logger)
        self.logger.info(f"Evaluating on {len(test_set)} samples.")

        train_cfg = self.config.TRAIN
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        assert test_set.get_num_neurons() == self.num_neurons # Compatibility check
        update = ckpt_dict["extra_state"]["update"]
        test_set.clip_spikes(self.max_spikes)
        self.model.eval()

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            with torch.no_grad():
                spikes, rates, heldout_spikes, forward_spikes = test_set.get_dataset()
                spikes = spikes.to(self.device)
                rates = rates.to(self.device)
                if test_set.has_heldout:
                    heldout_spikes = heldout_spikes.to(self.device)
                else:
                    heldout_spikes = None
                if test_set.has_forward:
                    forward_spikes = forward_spikes.to(self.device)
                else:
                    forward_spikes = None
                masked_spikes, labels = self.masker.mask_batch(
                    spikes,
                    train_cfg,
                    max_spikes=self.max_spikes,
                    should_mask=is_input_masked_model(self.config.MODEL.NAME),
                    heldout_spikes=heldout_spikes,
                    forward_spikes=forward_spikes
                )
                loss, pred_rates, *_ = self.model(masked_spikes, mask_labels=labels)
                test_loss = loss.mean()

                writer.add_scalar(
                    "test_loss",
                    test_loss,
                    update,
                )

                # Ideally we could do this just on masked areas
                selected_mask = labels != UNMASKED_LABEL
                masked_rates = torch.masked_select(rates, selected_mask).cpu()
                masked_pred_rates = torch.masked_select(pred_rates, selected_mask).cpu()
                r2 = r2_score(masked_rates, masked_pred_rates, multioutput='uniform_average')
                writer.add_scalar("test_r2", r2, update)
                self.logger.queue_stat("test r2", r2)

                self.logger.queue_stat("test loss", test_loss.item())

                stat_str = "\t".join([f"{stat[0]}: {stat[1]:.3f}" for stat in self.logger.empty_queue()])
                self.logger.info("update: {}\t{}".format(update, stat_str))

    def get_rates(
        self,
        checkpoint_path = None,
        mode = DATASET_MODES.trainval,
        save_path = None,
        keep_layers = -1, # keep last layer
    ) -> None:
        r"""Evaluates model (with checkpoint loaded) on train/val data and retrieves rates and activations (features for downstream tasks).
        Matches LFADS structure - we thus use a single dataset (no train val differentiator).
        Args:
            checkpoint_path: path of checkpoint (will use model on runner if not provided)
            save_path: Path to save activations at (optional). Does not save if nothing provided

        Returns:
            rates: ! confirm shape
            layer_outputs: ! confirm shape
        """
        self.logger.info(f"Getting rates...")
        if self.device is None:
            self.load_device()
        train_cfg = self.config.TRAIN
        self.masker = Masker(train_cfg, self.device) # Unused

        whole_set = SpikesDataset(self.config, self.config.DATA.TRAIN_FILENAME, mode=mode, logger=self.logger)
        self.max_spikes = whole_set.get_max_spikes() + 3
        self.num_neurons = whole_set.get_num_neurons()
        self.logger.info(f"Evaluating on {len(whole_set)} samples.")
        data_generator = data.DataLoader(whole_set,
            batch_size=train_cfg.BATCH_SIZE, shuffle=False
        )

        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        if self.num_neurons is None:
            self.num_neurons(whole_set.get_num_neurons())
        update = ckpt_dict["extra_state"]["update"]
        if self.max_spikes is not None:
            whole_set.clip_spikes(self.max_spikes)
        self.model.eval()

        with torch.no_grad():
            losses = []
            pred_rates = []
            layer_outputs = []
            # all_attentions = []
            for spikes, _, heldout_spikes, forward_spikes in data_generator:
                spikes = spikes.to(self.device)
                # Do NOT provide privileged eval info
                if data_generator.dataset.has_heldout:
                    heldout_spikes = heldout_spikes.to(self.device)
                    spikes = torch.cat([spikes, torch.zeros_like(heldout_spikes)], -1)
                else:
                    heldout_spikes = None
                if data_generator.dataset.has_forward:
                    forward_spikes = forward_spikes.to(self.device)
                    spikes = torch.cat([spikes, torch.zeros_like(forward_spikes)], 1)
                else:
                    forward_spikes = None
                labels = spikes # i.e. predict everything
                loss, batch_rates, batch_layer_outputs, *_ = self.model(
                # loss, batch_rates, batch_layer_outputs, _, _, batch_attn_list, *_ = self.model(
                    spikes,
                    mask_labels=spikes,
                    passthrough=True,
                    return_outputs=True,
                    return_weights=True,
                )
                batch_layer_outputs = batch_layer_outputs[keep_layers:]
                losses.append(loss.mean().item())
                pred_rates.append(batch_rates)
                # batch_layer_outputs is Batch list of Layer list of modules * T x B x H (permuted due to transformer)
                layer_outputs.append(batch_layer_outputs)
                # layer x trial x time x time
                # all_attentions.append(batch_attn_list)
            # trial x time x h
            pred_rates = torch.cat(pred_rates, dim=0)
            if self.config.MODEL.LOGRATE:
                pred_rates = pred_rates.exp()
            # Note this a list
            outputs_per_layer = zip(*layer_outputs) # Now lists of all samples, grouped by layer
            all_layer_outputs = [torch.cat(layer, dim=1).permute(1, 0, 2) for layer in outputs_per_layer]
            # all_layer_outputs is Layer list of B x M*T x H

            # attention_per_layer = zip(*all_attentions) # Lists of all samples, grouped by layer
            # all_attentions = torch.stack([torch.cat(layer, dim=0) for layer in attention_per_layer], dim=0)
            self.logger.queue_stat("test loss", torch.tensor(losses).mean().item())
            self.logger.log_update(update)

        if save_path is not None:
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('rates', data=pred_rates.cpu().numpy())
                f.create_dataset('layer_outputs', data=all_layer_outputs[-1].cpu().numpy()) # Only final layer
                # f.create_dataset('attention', data=all_attentions.cpu().numpy())
        return pred_rates, all_layer_outputs # , all_attentions

    def _clean_rates(self, gt, pred, flatten=False):
        if gt.size() != pred.size():
            raise Exception(f"Incompatible r2 sizes, GT: {gt.size()}, Pred: {pred.size()}")

        if flatten or len(gt.size()) > 1:
            gt = gt.flatten(end_dim=1)
            pred = pred.flatten(end_dim=1)

        if self.config.MODEL.LOGRATE:
            gt = gt.exp()
            pred = pred.exp()
        return gt.cpu(), pred.cpu()

    def neuron_r2(self, gt, pred, **kwargs):
        gt, pred = self._clean_rates(gt, pred, **kwargs)
        return r2_score(gt, pred, multioutput='uniform_average')

    def neuron_vaf(self, gt, pred, **kwargs):
        gt, pred = self._clean_rates(gt, pred, **kwargs)
        return explained_variance_score(gt, pred, multioutput='uniform_average')

    # For HParams
    def extract_hps_dict(self):
        hp_dict = {}
        hp_dict.update(self._extract_flat_dict(self.config.MODEL, "MODEL"))
        hp_dict.update(self._extract_flat_dict(self.config.TRAIN, "TRAIN"))
        return hp_dict

    BLACKLIST = ['MODEL/LOSS']
    def _extract_flat_dict(self, config, prefix):
        flat_dict = {}
        if prefix in Runner.BLACKLIST:
            return flat_dict
        for key, value in config.items():
            if isinstance(value, dict):
                flat_dict.update(self._extract_flat_dict(value, f"{prefix}/{key}"))
            elif not isinstance(value, list): # drop lists
                flat_dict[f"{prefix}/{key}"] = value
        return flat_dict
