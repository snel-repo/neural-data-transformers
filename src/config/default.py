#!/usr/bin/env python3
# Author: Joel Ye

from typing import List, Optional, Union

from yacs.config import CfgNode as CN

DEFAULT_CONFIG_DIR = "config/"
CONFIG_FILE_SEPARATOR = ","

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 100

# Name of experiment
_C.VARIANT = "experiment"
_C.USE_TENSORBOARD = True
_C.TENSORBOARD_DIR = "tb/"
_C.CHECKPOINT_DIR = "ckpts/"
_C.LOG_DIR = "logs/"

# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------
_C.SYSTEM = CN()
_C.SYSTEM.TORCH_GPU_ID = 0
_C.SYSTEM.GPU_AUTO_ASSIGN = True # Auto-assign
_C.SYSTEM.NUM_GPUS = 1

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.DATAPATH = 'data/'
_C.DATA.TRAIN_FILENAME = 'train.pth'
_C.DATA.VAL_FILENAME = 'val.pth'
_C.DATA.TEST_FILENAME = 'test.pth'
_C.DATA.OVERFIT_TEST = False
_C.DATA.RANDOM_SUBSET_TRIALS = 1.0 # Testing how NDT performs on a variety of dataset sizes

_C.DATA.LOG_EPSILON = 1e-7 # prevent -inf if we use logrates
# _C.DATA.IGNORE_FORWARD = True # Ignore forward prediction even if it's available in train. Useful if we don't have forward spikes in validation. (system misbehaves if we only have train...)
# Performance with above seems subpar, i.e. we need forward and heldout together for some reason
_C.DATA.IGNORE_FORWARD = False

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = "NeuralDataTransformer"
_C.MODEL.TRIAL_LENGTH = -1 # -1 represents "as much as available" # ! not actually supported in model yet...
_C.MODEL.CONTEXT_FORWARD = 4 # -1 represents "as much as available"
_C.MODEL.CONTEXT_BACKWARD = 8 # -1 represents "as much as available"
_C.MODEL.CONTEXT_WRAP_INITIAL = False
_C.MODEL.FULL_CONTEXT = False # Ignores CONTEXT_FORWARD and CONTEXT_BACKWARD if True (-1 for both)
_C.MODEL.UNMASKED_LOSS_SCALE = 0.0 # Relative scale for predicting unmasked spikes (kinda silly - deprecated)
_C.MODEL.HIDDEN_SIZE = 128 # Generic hidden size, used as default
_C.MODEL.DROPOUT = .1 # Catch all
_C.MODEL.DROPOUT_RATES = 0.2 # Specific for rates
_C.MODEL.DROPOUT_EMBEDDING = 0.2 # Dropout Population Activity pre-transformer
_C.MODEL.NUM_HEADS = 2
_C.MODEL.NUM_LAYERS = 6
_C.MODEL.ACTIVATION = "relu" # "gelu"
_C.MODEL.LINEAR_EMBEDDER = False # Use linear layer instead of embedding layer
_C.MODEL.EMBED_DIM = 2 # this greatly affects model size btw
_C.MODEL.LEARNABLE_POSITION = False
_C.MODEL.MAX_SPIKE_COUNT = 20
_C.MODEL.REQUIRES_RATES = False
_C.MODEL.LOGRATE = True # If true, we operate in lograte, and assume rates from data are logrates. Only for R2 do we exp
_C.MODEL.SPIKE_LOG_INIT = False # If true, init spike embeddings as a 0 centered linear sequence
_C.MODEL.FIXUP_INIT = False
_C.MODEL.PRE_NORM = False # per transformers without tears
_C.MODEL.SCALE_NORM = False # per transformers without tears

_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.LAYERS = 1

_C.MODEL.LOSS = CN()
_C.MODEL.LOSS.TYPE = "poisson" # ["cel", "poisson"]
_C.MODEL.LOSS.TOPK = 1.0 # In case we're neglecting some neurons, focus on them

_C.MODEL.POSITION = CN()
_C.MODEL.POSITION.OFFSET = True
# -----------------------------------------------------------------------------
# Train Config
# -----------------------------------------------------------------------------
_C.TRAIN = CN()

_C.TRAIN.DO_VAL = True # Run validation while training
_C.TRAIN.DO_R2 = True # Run validation while training

_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.NUM_UPDATES = 10000 # Max updates
_C.TRAIN.MAX_GRAD_NORM = 200.0
_C.TRAIN.USE_ZERO_MASK = True
_C.TRAIN.MASK_RATIO = 0.2
_C.TRAIN.MASK_TOKEN_RATIO = 1.0 # We don't need this if we use zero mask
_C.TRAIN.MASK_RANDOM_RATIO = 0.5 # Of the non-replaced, what percentage should be random?
_C.TRAIN.MASK_MODE = "timestep" # ["full", "timestep"]
_C.TRAIN.MASK_MAX_SPAN = 1
_C.TRAIN.MASK_SPAN_RAMP_START = 600
_C.TRAIN.MASK_SPAN_RAMP_END = 1200

_C.TRAIN.LR = CN()
_C.TRAIN.LR.INIT = 1e-3
_C.TRAIN.LR.SCHEDULE = True
_C.TRAIN.LR.SCHEDULER = "cosine" # invsqrt
_C.TRAIN.LR.WARMUP = 1000 # Mostly decay
_C.TRAIN.WEIGHT_DECAY = 0.0
_C.TRAIN.EPS = 1e-8
_C.TRAIN.PATIENCE = 750  # For early stopping (be generous, our loss steps)

_C.TRAIN.CHECKPOINT_INTERVAL = 1000
_C.TRAIN.LOG_INTERVAL = 50
_C.TRAIN.VAL_INTERVAL = 10 # Val less often so things run faster

_C.TRAIN.TUNE_MODE = False
_C.TRAIN.TUNE_EPOCHS_PER_GENERATION = 500
_C.TRAIN.TUNE_HP_JSON = "./lorenz_pbt.json"
_C.TRAIN.TUNE_WARMUP = 0
_C.TRAIN.TUNE_METRIC = "smth_masked_loss"
# JSON schema - flattened config dict. Each entry has info to construct a hyperparam.

def get_cfg_defaults():
  """Get default LFADS config (yacs config node)."""
  return _C.clone()

def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = get_cfg_defaults()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config


# The flatten and unflatten snippets are from an internal lfads_tf2 implementation.

def flatten(dictionary, level=[]):
    """ Flattens a dictionary by placing '.' between levels.

    This function flattens a hierarchical dictionary by placing '.'
    between keys at various levels to create a single key for each
    value. It is used internally for converting the configuration
    dictionary to more convenient formats. Implementation was
    inspired by `this StackOverflow post
    <https://stackoverflow.com/questions/6037503/python-unflatten-dict>`_.

    Parameters
    ----------
    dictionary : dict
        The hierarchical dictionary to be flattened.
    level : str, optional
        The string to append to the beginning of this dictionary,
        enabling recursive calls. By default, an empty string.

    Returns
    -------
    dict
        The flattened dictionary.

    See Also
    --------
    lfads_tf2.utils.unflatten : Performs the opposite of this operation.

    """

    tmp_dict = {}
    for key, val in dictionary.items():
        if type(val) == dict:
            tmp_dict.update(flatten(val, level + [key]))
        else:
            tmp_dict['.'.join(level + [key])] = val
    return tmp_dict


def unflatten(dictionary):
    """ Unflattens a dictionary by splitting keys at '.'s.

    This function unflattens a hierarchical dictionary by splitting
    its keys at '.'s. It is used internally for converting the
    configuration dictionary to more convenient formats. Implementation was
    inspired by `this StackOverflow post
    <https://stackoverflow.com/questions/6037503/python-unflatten-dict>`_.

    Parameters
    ----------
    dictionary : dict
        The flat dictionary to be unflattened.

    Returns
    -------
    dict
        The unflattened dictionary.

    See Also
    --------
    lfads_tf2.utils.flatten : Performs the opposite of this operation.

    """

    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict

