# Src: Andrew's run_random_search in tune_tf2

"""
Run grid search for NDT.
"""

from typing import List, Union
import os
from os import path
import json
import argparse
import ray, yaml, shutil
from ray import tune
import torch

from tune_models import tuneNDT

from defaults import DEFAULT_CONFIG_DIR
from src.config.default import flatten

PBT_HOME = path.expanduser('~/user_data/nlb/ndt_runs/ray/')
os.makedirs(PBT_HOME, exist_ok=True) # ray hangs if this isn't true...
OVERWRITE = True
PBT_METRIC = 'smth_masked_loss'
BEST_MODEL_METRIC = 'best_masked_loss'
LOGGED_COLUMNS = ['smth_masked_loss', 'masked_loss', 'r2', 'unmasked_loss']

DEFAULT_HP_DICT = {
    'TRAIN.WEIGHT_DECAY': tune.loguniform(1e-8, 1e-3),
    'TRAIN.MASK_RATIO': tune.uniform(0.1, 0.4)
}

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-config", "-e",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )

    parser.add_argument('--eval-only', '-ev', dest='eval_only', action='store_true')
    parser.add_argument('--no-eval-only', '-nev', dest='eval_only', action='store_false')
    parser.set_defaults(eval_only=False)

    parser.add_argument(
        "--name", "-n",
        type=str,
        default="",
        help="defaults to exp filename"
    )

    parser.add_argument(
        "--gpus-per-worker", "-g",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--cpus-per-worker", "-c",
        type=float,
        default=3.0
    )

    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=-1,
        help="-1 indicates -- use max possible workers on machine (assuming 0.5 GPUs per trial)"
    )

    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=20,
        help="samples for random search"
    )

    parser.add_argument(
        "--seed", "-d",
        type=int,
        default=-1,
        help="seed for config"
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    launch_search(**vars(args))

def build_hp_dict(raw_json: dict):
    hp_dict = {}
    for key in raw_json:
        info: dict = raw_json[key]
        sample_fn = info.get("sample_fn", "uniform")
        assert hasattr(tune, sample_fn)
        if sample_fn == "choice":
            hp_dict[key] = tune.choice(info['opts'])
        else:
            assert "low" in info, "high" in info
            sample_fn = getattr(tune, sample_fn)
            hp_dict[key] = sample_fn(info['low'], info['high'])
    return hp_dict

def launch_search(exp_config: Union[List[str], str], name: str, workers: int, gpus_per_worker: float, cpus_per_worker: float, eval_only: bool, samples: int, seed: int) -> None:
    # ---------- PBT I/O CONFIGURATION ----------
    # the directory to save PBT runs (usually '~/ray_results')

    if len(path.split(exp_config)[0]) > 0:
        CFG_PATH = exp_config
    else:
        CFG_PATH = path.join(DEFAULT_CONFIG_DIR, exp_config)
    variant_name = path.split(CFG_PATH)[1].split('.')[0]
    # Ok, now update the paths in the config
    if seed > 0:
        variant_name = f"{variant_name}-s{seed}"
    if name == "":
        name = variant_name

    pbt_dir = path.join(PBT_HOME, name)
    # the name of this PBT run (run will be stored at `pbt_dir`)

    # ---------- PBT RUN CONFIGURATION ----------
    # whether to use single machine or cluster
    SINGLE_MACHINE = True # Cluster not supported atm, don't know how to use it.

    NUM_WORKERS = workers if workers > 0 else int(torch.cuda.device_count() // gpus_per_worker)
    # the resources to allocate per model
    RESOURCES_PER_TRIAL = {"cpu": cpus_per_worker, "gpu": gpus_per_worker}

    # ---------------------------------------------

    def train():
        if path.exists(pbt_dir):
            print("Run exists!!! Overwriting.")
            if not OVERWRITE:
                print("overwriting disallowed, exiting..")
                exit(0)
            else:
                if path.exists(pbt_dir):
                    shutil.rmtree(pbt_dir)

        # load the configuration as a dictionary and update for this run
        flat_cfg_dict = flatten(yaml.full_load(open(CFG_PATH)))

        # Default behavior is to pull experiment name from config file
        # Bind variant name to directories
        flat_cfg_dict.update({'VARIANT': variant_name})
        if seed > 0:
            flat_cfg_dict.update({'SEED': seed})

        # the hyperparameter space to search
        assert 'TRAIN.TUNE_HP_JSON' in flat_cfg_dict, "please specify hp sweep (no default)"
        with open(flat_cfg_dict['TRAIN.TUNE_HP_JSON']) as f:
            raw_hp_json = json.load(f)
        cfg_samples = DEFAULT_HP_DICT
        cfg_samples.update(build_hp_dict(raw_hp_json))

        flat_cfg_dict.update(cfg_samples)

        # connect to Ray cluster or start on single machine
        address = None if SINGLE_MACHINE else 'localhost:6379'
        ray.init(address=address)

        reporter = tune.CLIReporter(metric_columns=LOGGED_COLUMNS)

        analysis = tune.run(
            tuneNDT,
            name=name,
            local_dir=pbt_dir,
            stop={'done': True},
            config=flat_cfg_dict,
            resources_per_trial=RESOURCES_PER_TRIAL,
            num_samples=samples,
            # sync_to_driver='# {source} {target}', # prevents rsync
            verbose=1,
            progress_reporter=reporter,
            # loggers=(tune.logger.JsonLogger, tune.logger.CSVLogger)
        )

    if not eval_only:
        train()
    # load the results dataframe for this run
    df = tune.Analysis(
        pbt_dir
    ).dataframe()
    df = df[df.logdir.apply(lambda path: not 'best_model' in path)]

    # Hm... we need to go through each model, and run the lfve ckpt.
    # And then record that in the dataframe?

    if df[BEST_MODEL_METRIC].dtype == 'O': # Accidentally didn't case to scalar, now we have a tensor string
        df = df.assign(best_masked_loss=lambda df: df[BEST_MODEL_METRIC].str[7:13].astype(float))
    best_model_logdir = df.loc[df[BEST_MODEL_METRIC].idxmin()].logdir
    # copy the  best model somewhere easy to find
    # best_model_src = path.join(best_model_logdir, 'model_dir')
    best_model_dest = path.join(pbt_dir, 'best_model')
    if path.exists(best_model_dest):
        shutil.rmtree(best_model_dest)
    shutil.copytree(best_model_logdir, best_model_dest)

if __name__ == "__main__":
    main()