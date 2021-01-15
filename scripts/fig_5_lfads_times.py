# Author: Joel Ye

# LFADS timing
# ! This is a reference script. May not run out of the box with the NDT environment (e.g. needs LFADS dependencies)
#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import copy
import cProfile
import os
import time
import gc

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.utils import Progbar

from lfads_tf2.defaults import get_cfg_defaults
from lfads_tf2.models import LFADS
from lfads_tf2.tuples import DecoderInput, SamplingOutput
from lfads_tf2.utils import (load_data, load_posterior_averages,
                             restrict_gpu_usage)

tfd = tfp.distributions
gc.disable()  # disable garbage collection

# tf.debugging.set_log_device_placement(True)
restrict_gpu_usage(gpu_ix=0)


# %%
# restore the LFADS model


# model = LFADS(model_dir) # A hardcoded chaotic path
model_dir = '/snel/home/joely/ray_results/lfads/chaotic-s1/best_model'
cfg_path = os.path.join(model_dir, 'model_spec.yaml') # Don't directly load model

def sample_and_average(n_samples=50,
                       batch_size=64,
                       merge_tv=False,
                       seq_len_cap=100):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.SEQ_LEN = seq_len_cap
    cfg.freeze()
    model = LFADS(
        cfg_node=cfg,
        # model_dir='/snel/home/joely/ray_results/lfads/chaotic-s1/best_model'
    )
    model.restore_weights()
    # ! Modified for timing
    if not model.is_trained:
        model.lgr.warn("Performing posterior sampling on an untrained model.")

    # define merging and splitting utilities
    def merge_samp_and_batch(data, batch_dim):
        """ Combines the sample and batch dimensions """
        return tf.reshape(data, [n_samples * batch_dim] +
                          tf.unstack(tf.shape(data)[2:]))

    def split_samp_and_batch(data, batch_dim):
        """ Splits up the sample and batch dimensions """
        return tf.reshape(data, [n_samples, batch_dim] +
                          tf.unstack(tf.shape(data)[1:]))

    # ========== POSTERIOR SAMPLING ==========
    # perform sampling on both training and validation data
    loop_times = []
    for prefix, dataset in zip(['train_', 'valid_'],
                               [model._train_ds, model._val_ds]):
        data_len = len(model.train_tuple.data) if prefix == 'train_' else len(
            model.val_tuple.data)

        # initialize lists to store rates
        all_outputs = []
        model.lgr.info(
            "Posterior sample and average on {} segments.".format(data_len))
        if not model.cfg.TRAIN.TUNE_MODE:
            pbar = Progbar(data_len, width=50, unit_name='dataset')

        def process_batch():
            # unpack the batch
            data, _, ext_input = batch
            data = data[:,:seq_len_cap]
            ext_input = ext_input[:,:seq_len_cap]

            time_start = time.time()

            # for each chop in the dataset, compute the initial conditions
            # distribution
            ic_mean, ic_stddev, ci = model.encoder.graph_call(data)
            ic_post = tfd.MultivariateNormalDiag(ic_mean, ic_stddev)

            # sample from the posterior and merge sample and batch dimensions
            ic_post_samples = ic_post.sample(n_samples)
            ic_post_samples_merged = merge_samp_and_batch(
                ic_post_samples, len(data))

            # tile and merge the controller inputs and the external inputs
            ci_tiled = tf.tile(tf.expand_dims(ci, axis=0),
                               [n_samples, 1, 1, 1])
            ci_merged = merge_samp_and_batch(ci_tiled, len(data))
            ext_tiled = tf.tile(tf.expand_dims(ext_input, axis=0),
                                [n_samples, 1, 1, 1])
            ext_merged = merge_samp_and_batch(ext_tiled, len(data))

            # pass all samples into the decoder
            dec_input = DecoderInput(ic_samp=ic_post_samples_merged,
                                     ci=ci_merged,
                                     ext_input=ext_merged)
            output_samples_merged = model.decoder.graph_call(dec_input)

            # average the outputs across samples
            output_samples = [
                split_samp_and_batch(t, len(data))
                for t in output_samples_merged
            ]
            output = [np.mean(t, axis=0) for t in output_samples]

            time_elapsed = time.time() - time_start

            if not model.cfg.TRAIN.TUNE_MODE:
                pbar.add(len(data))
            return time_elapsed

        for batch in dataset.batch(batch_size):
            loop_times.append(process_batch())
        return loop_times
# %%
def time_bin(bins):
    n_samples = 1
    p_loop_times = sample_and_average(batch_size=1, n_samples=n_samples, seq_len_cap=bins)
    del p_loop_times[0]  # first iteration is slower due to graph initialization
    p_loop_times = np.array(p_loop_times) * 1e3
    print(f"{p_loop_times.mean():.3f}ms for {bins} bins")
    return p_loop_times

all_times = []
for bins in range(5, 10, 5):
for bins in range(5, 105, 5):
    all_times.append(time_bin(bins))
all_times = np.array(all_times)

with open('lfads_times.npy', 'wb') as f:
    np.save(f, all_times)
